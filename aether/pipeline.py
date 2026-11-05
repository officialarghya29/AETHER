"""End-to-end reliability pipeline.

This is the single orchestration path used by the API, the CLI and the benchmark harness, so
measured behaviour and served behaviour cannot diverge.

Per request (or batch):

1. validate and preprocess pixels (one shared preprocessing definition)
2. run the lightweight model
3. compute uncertainty, calibrated with a temperature fitted on the calibration split
4. update the drift window and evaluate shift evidence periodically
5. estimate error risk from inference-time evidence
6. ask the routing controller for an action
7. escalate to the advanced model when the policy says the expected benefit is worth the cost
8. diagnose failures with evidence-backed rules
9. return a fully justified result

No ground-truth label is ever consulted here.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from aether.adapters.base import Prediction
from aether.config import AetherConfig
from aether.diagnosis.image_quality import ImageQuality, analyze_image
from aether.diagnosis.rules import DiagnosisEngine
from aether.drift.monitor import DriftMonitor, DriftObservation, DriftReport, RollingWindow
from aether.drift.reference import load_reference
from aether.errors import InferenceError, ModelNotAvailableError
from aether.models.registry import ModelRegistry
from aether.routing.controller import RoutingController
from aether.routing.cost import Action
from aether.routing.policy import RoutingDecision, RoutingState
from aether.routing.risk_model import RiskModel, default_risk, features_from_evidence
from aether.uncertainty.calibration import CalibrationStore, apply_temperature
from aether.uncertainty.entropy import (
    max_probability,
    prediction_margin,
    predictive_entropy,
    softmax,
)
from aether.utils.logging import get_logger

logger = get_logger("aether.pipeline")


@dataclass
class InferenceResult:
    """A complete, auditable inference outcome."""

    prediction_id: str
    action: str
    predicted_index: int
    predicted_label: str
    probabilities: list[float]
    model_id: str
    model_version: str
    uncertainty: dict[str, float]
    calibrated_uncertainty: dict[str, float]
    temperature: float
    risk: dict[str, Any]
    drift: dict[str, Any]
    decision: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    quality: dict[str, Any]
    lightweight: dict[str, Any]
    advanced: dict[str, Any] | None
    timings: dict[str, float]
    input_size: int
    created_at: str = ""
    latency_ms: float = 0.0
    risk_budget: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "action": self.action,
            "predicted_index": self.predicted_index,
            "predicted_label": self.predicted_label,
            "probabilities": [round(value, 6) for value in self.probabilities],
            "model_id": self.model_id,
            "model_version": self.model_version,
            "uncertainty": self.uncertainty,
            "calibrated_uncertainty": self.calibrated_uncertainty,
            "temperature": self.temperature,
            "risk": self.risk,
            "drift": self.drift,
            "decision": self.decision,
            "diagnostics": self.diagnostics,
            "quality": self.quality,
            "lightweight": self.lightweight,
            "advanced": self.advanced,
            "timings": self.timings,
            "input_size": self.input_size,
            "created_at": self.created_at,
            "latency_ms": round(self.latency_ms, 3),
            "risk_budget": self.risk_budget,
        }


def tta_enabled(config: AetherConfig, severity: str) -> bool:
    """Whether an escalated request should pay for the mirror pass.

    TTA has to be switched on explicitly, and by default it is gated on the drift monitor so that
    clean traffic keeps the single-pass cost.
    """
    if not config.routing.tta:
        return False
    return not config.routing.tta_on_drift or severity != "none"


def _uncertainty(probabilities: np.ndarray, epsilon: float) -> dict[str, float]:
    return {
        "predictive_entropy": float(predictive_entropy(probabilities, epsilon)),
        "max_probability": float(max_probability(probabilities)),
        "prediction_margin": float(prediction_margin(probabilities)),
    }


class ReliabilityPipeline:
    """Model-agnostic reliability orchestration."""

    def __init__(
        self,
        config: AetherConfig,
        *,
        device: torch.device | str = "cpu",
        registry: ModelRegistry | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.registry = registry or ModelRegistry(config, self.device)
        self.controller = RoutingController(config.routing)
        self.calibration = CalibrationStore(config.resolve(config.uncertainty.temperature_artifact))
        self.diagnosis = DiagnosisEngine()
        self.window = RollingWindow(config.drift.evaluation_window_size)
        self._drift_monitor: DriftMonitor | None = None
        self._drift_error: str | None = None
        self._last_drift: DriftReport | None = None
        self._observations_since_eval = 0
        self._risk_light = self._load_risk_model("lightweight")
        self._risk_advanced = self._load_risk_model("advanced")
        self._risk_precall = self._load_risk_model("advanced_precall")
        self._advanced_latency_ms = config.routing.advanced_latency_estimate_ms
        # Checkpoint presence is a filesystem stat; cache it instead of paying for it per batch.
        self._advanced_present = bool(config.routing.enable_escalation) and config.resolve(
            config.models.advanced.checkpoint
        ).exists()
        # The first call to each model includes lazy weight loading and thread-pool warm-up. Those
        # samples are not steady-state latency, so they must not set the budget estimate: a single
        # cold start would otherwise make every later escalation look unaffordable.
        self._light_warm = False
        self._advanced_warm = False
        self._light_latency_ms = 0.0

    # ------------------------------------------------------------------ setup
    def _load_risk_model(self, role: str) -> RiskModel | None:
        path = self.config.resolve(self.config.routing.learned_policy.artifact)
        if not path.exists():
            if role == "lightweight":
                logger.warning(
                    "learned policy artifact absent; using heuristic risk estimates",
                    extra={"expected": str(path)},
                )
            return None
        try:
            from aether.routing.learned_policy import load_precall_risk_model, load_risk_models

            if role == "advanced_precall":
                precall = load_precall_risk_model(path)
                if precall is None:
                    logger.warning(
                        "artifact has no pre-call advanced head; escalation will use the "
                        "heuristic risk estimate",
                        extra={"artifact": str(path)},
                    )
                return precall
            light, advanced = load_risk_models(path)
            return light if role == "lightweight" else advanced
        except Exception as exc:
            logger.warning("risk model unavailable", extra={"role": role, "error": str(exc)})
            return None

    def drift_monitor(self) -> DriftMonitor:
        """Lazily build the drift monitor (the reference artifact is ~10 MB)."""
        if self._drift_monitor is not None:
            return self._drift_monitor
        if self._drift_error is not None:
            raise ModelNotAvailableError(f"drift reference unavailable: {self._drift_error}")
        try:
            reference = load_reference(self.config.resolve(self.config.drift.reference_artifact))
            self._drift_monitor = DriftMonitor(
                reference,
                self.config.drift,
                num_classes=self.config.models.lightweight.num_classes,
                seed=self.config.runtime.seed,
            )
        except Exception as exc:
            self._drift_error = str(exc)
            raise
        return self._drift_monitor

    def warm(self, roles: tuple[str, ...] = ("lightweight",)) -> None:
        self.registry.warmup_all(roles)

    @property
    def last_drift(self) -> DriftReport | None:
        """Most recent drift assessment for the live request window."""
        return self._last_drift

    @property
    def artifact_status(self) -> dict[str, bool]:
        return {
            "lightweight_checkpoint": self.config.resolve(
                self.config.models.lightweight.checkpoint
            ).exists(),
            "advanced_checkpoint": self.config.resolve(
                self.config.models.advanced.checkpoint
            ).exists(),
            "calibration": self.config.resolve(
                self.config.uncertainty.temperature_artifact
            ).exists(),
            "reference": self.config.resolve(self.config.drift.reference_artifact).exists(),
            "risk_model": self.config.resolve(
                self.config.routing.learned_policy.artifact
            ).exists(),
        }

    # ------------------------------------------------------------- inference
    def predict_image(self, pixels: np.ndarray) -> InferenceResult:
        """Convenience wrapper for a single HWC image."""
        if pixels.ndim != 3:
            raise ValueError(f"expected an HWC image, got shape {pixels.shape}")
        return self.predict_batch(pixels[None, ...])[0]

    def predict_batch(self, images: np.ndarray) -> list[InferenceResult]:
        """Run the full reliability pipeline over a batch of uint8 NHWC images."""
        started = time.perf_counter()
        batch, qualities = self._prepare(images)
        light_adapter = self.registry.get("lightweight")
        t_light = time.perf_counter()
        light_predictions = light_adapter.predict(batch)
        light_ms = (time.perf_counter() - t_light) * 1000.0
        self._assert_finite(light_predictions, "lightweight")

        temperatures = [self.calibration.temperature(p.model_id) for p in light_predictions]
        calibrated = [
            apply_temperature(p.logits, t) for p, t in zip(light_predictions, temperatures, strict=True)
        ]
        uncertainties = [
            _uncertainty(c, self.config.uncertainty.epsilon) for c in calibrated
        ]
        raw_uncertainties = [
            _uncertainty(p.probabilities, self.config.uncertainty.epsilon)
            for p in light_predictions
        ]
        drifts = self._evaluate_drift(
            light_predictions, calibrated, qualities, images, uncertainties=uncertainties
        )
        # Both first-call heads consume the same feature vector; building it once halves the
        # feature cost of the router without changing either estimate.
        risk_features = [
            features_from_evidence(
                entropy=uncertainty["predictive_entropy"],
                max_probability=uncertainty["max_probability"],
                margin=uncertainty["prediction_margin"],
                drift_score=drift.score,
                severity=drift.severity,
                disagreement=None,
            )
            for uncertainty, drift in zip(uncertainties, drifts, strict=True)
        ]
        risks = [
            self._risk_from_features("lightweight", features)[0] for features in risk_features
        ]
        # Estimate the advanced model's risk from lightweight evidence only (a dedicated pre-call
        # head). Without this estimate a cost-aware policy could never justify a second call, and
        # using the post-call head here would price escalation with information the caller has not
        # paid for.
        advanced_risk_estimates = [
            self._risk_from_features("advanced_precall", features)[0] for features in risk_features
        ]
        advanced_present = self._advanced_present
        elapsed_light = light_ms / max(1, len(light_predictions))
        if self._light_warm:
            self._light_latency_ms = elapsed_light
        else:
            self._light_warm = True
            elapsed_light = self._light_latency_ms
        affordable = self.controller.escalation_allowed(
            escalation_count=0,
            advanced_available=advanced_present,
            estimated_advanced_latency_ms=self._advanced_latency_ms,
            elapsed_ms=elapsed_light,
        )

        states = [
            self._state(
                risk_light=risk,
                uncertainty=uncertainty,
                drift=drift,
                elapsed_ms=elapsed_light,
                escalation_count=0,
                advanced_available=advanced_present,
                risk_advanced=estimate,
                escalation_affordable=affordable,
            )
            for risk, uncertainty, drift, estimate in zip(
                risks, uncertainties, drifts, advanced_risk_estimates, strict=True
            )
        ]
        decisions = [self.controller.decide(state) for state in states]

        escalated_indices = [
            index for index, decision in enumerate(decisions) if decision.action is Action.ESCALATE
        ]
        advanced_predictions = self._run_advanced(
            batch, escalated_indices, states, decisions, calibrated, risks, drifts
        )

        results: list[InferenceResult] = []
        for index, light_prediction in enumerate(light_predictions):
            results.append(
                self._finalise(
                    index=index,
                    light_prediction=light_prediction,
                    light_probabilities=calibrated[index],
                    light_uncertainty=uncertainties[index],
                    raw_uncertainty=raw_uncertainties[index],
                    temperature=temperatures[index],
                    quality=qualities[index],
                    drift=drifts[index],
                    risk_light=risks[index],
                    decision=decisions[index],
                    advanced=advanced_predictions.get(index),
                    light_ms=light_ms / max(1, len(light_predictions)),
                    started=started,
                )
            )
        return results

    # ----------------------------------------------------------- internals
    def _prepare(self, images: np.ndarray) -> tuple[torch.Tensor, list[ImageQuality]]:
        if not isinstance(images, np.ndarray) or images.ndim != 4:
            raise ValueError("expected a uint8 NHWC array")
        if images.shape[0] == 0:
            raise ValueError("empty batch")
        if images.shape[0] > self.config.runtime.max_batch_size:
            raise ValueError(
                f"batch of {images.shape[0]} exceeds the configured maximum "
                f"{self.config.runtime.max_batch_size}"
            )
        from aether.data.preprocessing import prepare_model_input

        qualities = [analyze_image(image) for image in images]
        batch = prepare_model_input(images, self.config.data.image_size)
        return batch, qualities

    def _evaluate_drift(
        self,
        predictions: list[Prediction],
        probabilities: list[np.ndarray],
        qualities: list[ImageQuality],
        images: np.ndarray,
        *,
        uncertainties: list[dict[str, float]] | None = None,
    ) -> list[DriftReport]:
        """Append observations to the window and periodically evaluate shift evidence."""
        monitor = None
        if self._drift_monitor is not None or self._drift_error is None:
            try:
                monitor = self.drift_monitor()
            except Exception:
                monitor = None
        # One vectorised reduction over the batch instead of a Python loop over pixels per image.
        channel_means = (
            images.reshape(images.shape[0], -1, images.shape[-1]).mean(axis=1) / 255.0
        )
        for index, (prediction, probs, quality) in enumerate(
            zip(predictions, probabilities, qualities, strict=True)
        ):
            if prediction.embedding is None:
                continue
            # Entropy and max-probability were already computed for calibration; recomputing them
            # here doubled the uncertainty cost of every request.
            if uncertainties is not None:
                entropy = uncertainties[index]["predictive_entropy"]
                confidence = uncertainties[index]["max_probability"]
            else:
                entropy = float(predictive_entropy(probs))
                confidence = float(max_probability(probs))
            self.window.add(
                DriftObservation(
                    input_stats=np.array(
                        [
                            quality.brightness,
                            quality.contrast,
                            quality.sharpness,
                            float(channel_means[index][0]),
                            float(channel_means[index][1]),
                            float(channel_means[index][2]),
                        ],
                        dtype=np.float64,
                    ),
                    embedding=prediction.embedding.astype(np.float64),
                    entropy=entropy,
                    max_probability=confidence,
                    predicted_class=prediction.class_index,
                )
            )
        self._observations_since_eval += len(predictions)

        if monitor is None:
            report = DriftReport(
                severity="none",
                score=0.0,
                window_size=len(self.window),
                status="reference_unavailable",
                warnings=[self._drift_error or "drift reference not loaded"],
            )
        elif (
            self._last_drift is None
            or self._observations_since_eval >= self.config.drift.evaluation_window_size
        ):
            report = monitor.evaluate(self.window.snapshot())
            self._last_drift = report
            self._observations_since_eval = 0
        else:
            report = self._last_drift
        return [report for _ in predictions]

    def _risk_for(
        self,
        role: str,
        uncertainty: dict[str, float],
        drift: DriftReport,
        disagreement: float | None = None,
    ) -> tuple[float, str]:
        """Estimate error risk. Disagreement is only meaningful for a completed second call."""
        evidence = features_from_evidence(
            entropy=uncertainty["predictive_entropy"],
            max_probability=uncertainty["max_probability"],
            margin=uncertainty["prediction_margin"],
            drift_score=drift.score,
            severity=drift.severity,
            disagreement=disagreement,
        )
        return self._risk_from_features(role, evidence)

    @staticmethod
    def _assert_finite(predictions: list[Prediction], role: str) -> None:
        """Fail loudly on non-finite model output instead of serving a silent NaN decision."""
        for prediction in predictions:
            if not np.all(np.isfinite(prediction.logits)):
                raise InferenceError(f"{role} model produced non-finite logits")

    @staticmethod
    def _merge_tta(
        primary: Prediction, mirrored: Prediction, classes: tuple[str, ...]
    ) -> Prediction:
        """Average the logits of an image and its horizontal mirror (test-time augmentation)."""
        logits = (
            np.asarray(primary.logits, dtype=np.float64)
            + np.asarray(mirrored.logits, dtype=np.float64)
        ) / 2.0
        probabilities = softmax(logits)
        index = int(np.argmax(probabilities))
        return Prediction(
            model_id=primary.model_id,
            model_version=primary.model_version,
            class_index=index,
            class_label=classes[index],
            logits=logits.astype(np.float32),
            probabilities=probabilities.astype(np.float32),
            latency_ms=primary.latency_ms + mirrored.latency_ms,
            embedding=primary.embedding,
        )

    def _risk_from_features(
        self, role: str, evidence: dict[str, float]
    ) -> tuple[float, str]:
        """Score a pre-built feature vector with the requested head."""
        model = {
            "lightweight": self._risk_light,
            "advanced": self._risk_advanced,
            "advanced_precall": self._risk_precall,
        }[role]
        if model is not None:
            return float(model.predict(evidence)), "learned"
        return float(default_risk(evidence)), "heuristic"

    def _state(
        self,
        *,
        risk_light: float,
        uncertainty: dict[str, float],
        drift: DriftReport,
        elapsed_ms: float,
        escalation_count: int,
        advanced_available: bool,
        risk_advanced: float | None = None,
        disagreement: float | None = None,
        escalation_affordable: bool = True,
        advanced_evaluated: bool = False,
    ) -> RoutingState:
        return RoutingState(
            risk_light=risk_light,
            risk_advanced=risk_advanced,
            entropy=uncertainty["predictive_entropy"],
            max_probability=uncertainty["max_probability"],
            margin=uncertainty["prediction_margin"],
            drift_score=drift.score,
            drift_severity=drift.severity,
            disagreement=disagreement,
            escalation_count=escalation_count,
            advanced_available=advanced_available,
            advanced_evaluated=advanced_evaluated,
            escalation_affordable=escalation_affordable,
            elapsed_ms=elapsed_ms,
        )

    def _run_advanced(
        self,
        batch: torch.Tensor,
        indices: list[int],
        states: list[RoutingState],
        decisions: list[RoutingDecision],
        light_probabilities: list[np.ndarray],
        light_risks: list[float],
        drifts: list[DriftReport],
    ) -> dict[int, dict[str, Any]]:
        """Execute the advanced model for escalated samples only."""
        if not indices:
            return {}
        advanced = self.registry.try_get("advanced")
        if advanced is None:
            # Escalation is impossible: re-decide so the action becomes review/abstain instead
            # of silently discarding an escalation request.
            fallback_drift = self._last_drift or DriftReport(
                severity="none", score=0.0, window_size=0
            )
            for index in indices:
                states[index] = self._state(
                    risk_light=light_risks[index],
                    uncertainty={
                        "predictive_entropy": states[index].entropy,
                        "max_probability": states[index].max_probability,
                        "prediction_margin": states[index].margin,
                    },
                    drift=fallback_drift,
                    elapsed_ms=states[index].elapsed_ms,
                    escalation_count=1,
                    advanced_available=False,
                )
                decisions[index] = self.controller.decide(states[index])
            return {}
        subset = batch[indices]
        started = time.perf_counter()
        predictions = advanced.predict(subset)
        self._assert_finite(predictions, "advanced")
        # Only the escalated rows whose drift report warrants it pay for the second pass; the rest
        # of the subset keeps single-pass latency.
        tta_positions = [
            position
            for position, index in enumerate(indices)
            if tta_enabled(self.config, drifts[index].severity)
        ]
        tta_applied = [False] * len(indices)
        if tta_positions:
            mirrored = advanced.predict(
                torch.flip(subset[tta_positions], dims=[3]), return_embeddings=False
            )
            self._assert_finite(mirrored, "advanced (test-time augmentation)")
            for position, mirror in zip(tta_positions, mirrored, strict=True):
                predictions[position] = self._merge_tta(
                    predictions[position], mirror, advanced.metadata.classes
                )
                tta_applied[position] = True
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        per_sample_ms = elapsed_ms / max(1, len(indices))
        if self._advanced_warm:
            self._advanced_latency_ms = per_sample_ms
        else:
            self._advanced_warm = True
        outputs: dict[int, dict[str, Any]] = {}
        for position, index in enumerate(indices):
            prediction = predictions[position]
            temperature = self.calibration.temperature(prediction.model_id)
            calibrated = apply_temperature(prediction.logits, temperature)
            uncertainty = _uncertainty(calibrated, self.config.uncertainty.epsilon)
            drift = self._last_drift or DriftReport(severity="none", score=0.0, window_size=0)
            # Total-variation distance between the two predictive distributions, in [0, 1].
            disagreement = float(
                0.5 * np.abs(np.asarray(light_probabilities[index]) - calibrated).sum()
            )
            risk_advanced, risk_source = self._risk_for(
                "advanced", uncertainty, drift, disagreement=disagreement
            )
            state = self._state(
                risk_light=light_risks[index],
                uncertainty=uncertainty,
                drift=drift,
                elapsed_ms=states[index].elapsed_ms + per_sample_ms,
                escalation_count=1,
                advanced_available=False,
                risk_advanced=risk_advanced,
                disagreement=disagreement,
                advanced_evaluated=True,
            )
            decision = self.controller.decide(state)
            decisions[index] = decision
            outputs[index] = {
                "prediction": prediction,
                "probabilities": calibrated,
                "uncertainty": uncertainty,
                "temperature": temperature,
                "risk": risk_advanced,
                "risk_source": risk_source,
                "disagreement": disagreement,
                "latency_ms": elapsed_ms / max(1, len(indices)),
                "action": decision.action.value,
                "test_time_augmentation": tta_applied[position],
            }
        return outputs

    def _finalise(
        self,
        *,
        index: int,
        light_prediction: Prediction,
        light_probabilities: np.ndarray,
        light_uncertainty: dict[str, float],
        raw_uncertainty: dict[str, float],
        temperature: float,
        quality: ImageQuality,
        drift: DriftReport,
        risk_light: float,
        decision: RoutingDecision,
        advanced: dict[str, Any] | None,
        light_ms: float,
        started: float,
    ) -> InferenceResult:
        chosen = advanced if advanced is not None and decision.action is Action.ACCEPT else None
        prediction = chosen["prediction"] if chosen else light_prediction
        probabilities = chosen["probabilities"] if chosen else light_probabilities
        uncertainty = chosen["uncertainty"] if chosen else light_uncertainty
        risk_value = chosen["risk"] if chosen else risk_light
        diagnostics = self.diagnosis.diagnose(
            entropy=uncertainty["predictive_entropy"],
            max_probability=uncertainty["max_probability"],
            margin=uncertainty["prediction_margin"],
            predicted_label=prediction.class_label,
            drift_severity=drift.severity,
            drift_score=drift.score,
            disagreement=advanced["disagreement"] if advanced else None,
            quality=quality,
            advanced_used=advanced is not None,
            advanced_failed=False,
            advanced_agrees=(
                prediction.class_index == light_prediction.class_index if advanced else None
            ),
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        decision_payload = decision.to_dict()
        # Whether the escalated call paid for a mirror pass is decided after routing, so it is
        # recorded on the decision record rather than invented before the call.
        decision_payload["test_time_augmentation"] = bool(
            advanced and advanced.get("test_time_augmentation")
        )
        return InferenceResult(
            prediction_id=uuid.uuid4().hex,
            action=decision.action.value,
            predicted_index=int(prediction.class_index),
            predicted_label=prediction.class_label,
            probabilities=[float(value) for value in probabilities],
            model_id=prediction.model_id,
            model_version=prediction.model_version,
            uncertainty=uncertainty,
            calibrated_uncertainty=light_uncertainty,
            temperature=temperature,
            risk={
                "value": round(float(risk_value), 6),
                "source": chosen["risk_source"] if chosen else ("learned" if self._risk_light else "heuristic"),
                "lightweight": round(float(risk_light), 6),
                "advanced": round(float(advanced["risk"]), 6) if advanced else None,
            },
            drift=drift.to_dict(),
            decision=decision_payload,
            diagnostics=[finding.to_dict() for finding in diagnostics],
            quality=quality.to_dict(),
            lightweight={
                "class_index": int(light_prediction.class_index),
                "class_label": light_prediction.class_label,
                "confidence": round(float(light_prediction.confidence), 6),
                "raw_uncertainty": raw_uncertainty,
                "model_id": light_prediction.model_id,
                "version": light_prediction.model_version,
                "latency_ms": round(light_ms, 3),
            },
            advanced=(
                {
                    "class_index": int(advanced["prediction"].class_index),
                    "class_label": advanced["prediction"].class_label,
                    "confidence": round(float(advanced["prediction"].confidence), 6),
                    "model_id": advanced["prediction"].model_id,
                    "version": advanced["prediction"].model_version,
                    "disagreement": round(float(advanced["disagreement"] or 0.0), 6),
                    "latency_ms": round(float(advanced["latency_ms"]), 3),
                    "test_time_augmentation": bool(advanced["test_time_augmentation"]),
                    "uncertainty": advanced["uncertainty"],
                }
                if advanced
                else None
            ),
            timings={
                "lightweight_ms": round(light_ms, 3),
                "advanced_ms": round(float(advanced["latency_ms"]), 3) if advanced else 0.0,
                "total_ms": round(elapsed, 3),
            },
            input_size=self.config.data.image_size,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            latency_ms=elapsed,
        )
