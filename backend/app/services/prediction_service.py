"""Prediction service: run the pipeline and persist the full decision trail."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from aether.config import AetherConfig
from aether.monitoring.events import prediction_event, routing_event
from aether.monitoring.metrics import METRICS
from aether.pipeline import InferenceResult, ReliabilityPipeline
from aether.utils.io import atomic_write_bytes
from aether.utils.logging import get_logger
from backend.app.database.models import (
    DiagnosticRecordModel,
    DriftEventModel,
    PredictionModel,
    RoutingDecisionModel,
    UncertaintyRecordModel,
)
from backend.app.services.review_service import build_review_case_model

logger = get_logger("aether.api.predictions")

REVIEW_ACTIONS = {"HUMAN_REVIEW", "ABSTAIN"}
PERSIST_IMAGE_ACTIONS = REVIEW_ACTIONS | {"MONITOR", "ESCALATE"}


class PredictionService:
    """Orchestrates inference plus durable recording."""

    def __init__(
        self, session: Session, pipeline: ReliabilityPipeline, config: AetherConfig
    ) -> None:
        self.session = session
        self.pipeline = pipeline
        self.config = config

    def run(
        self,
        images: np.ndarray,
        *,
        request_id: str | None = None,
        raw_payloads: list[bytes] | None = None,
        persist_image: bool = True,
    ) -> list[tuple[InferenceResult, str | None]]:
        """Predict a batch and persist predictions, decisions, diagnostics and review cases."""
        results = self.pipeline.predict_batch(images)
        output: list[tuple[InferenceResult, str | None]] = []
        drift_recorded = False
        for index, result in enumerate(results):
            image_ref = None
            if (
                persist_image
                and raw_payloads
                and index < len(raw_payloads)
                and result.action in PERSIST_IMAGE_ACTIONS
            ):
                image_ref = self._store_image(raw_payloads[index])
            review_id = self._persist(result, image_ref=image_ref, request_id=request_id)
            if not drift_recorded and result.drift.get("severity") in ("moderate", "severe"):
                self._persist_drift_event(result)
                drift_recorded = True
            self._record_metrics(result)
            output.append((result, review_id))
        self.session.flush()
        return output

    # ------------------------------------------------------------------ helpers
    def _store_image(self, payload: bytes) -> str:
        """Persist an uploaded image for audit, under a generated name only."""
        directory = self.config.upload_dir
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}.img"
        path = directory / name
        atomic_write_bytes(path, payload)
        return f"uploads/{name}"

    def _persist(
        self, result: InferenceResult, *, image_ref: str | None, request_id: str | None
    ) -> str | None:
        prediction = PredictionModel(
            id=result.prediction_id,
            action=result.action,
            predicted_index=result.predicted_index,
            predicted_label=result.predicted_label,
            probabilities=result.probabilities,
            model_id=result.model_id,
            model_version=result.model_version,
            latency_ms=result.latency_ms,
            input_size=result.input_size,
            drift_severity=result.drift.get("severity", "none"),
            risk=result.risk,
            uncertainty=result.uncertainty,
            decision=result.decision,
            image_ref=image_ref,
            request_id=request_id,
        )
        self.session.add(prediction)
        self.session.add(
            UncertaintyRecordModel(
                prediction_id=prediction.id,
                predictive_entropy=float(result.uncertainty["predictive_entropy"]),
                max_probability=float(result.uncertainty["max_probability"]),
                prediction_margin=float(result.uncertainty["prediction_margin"]),
                temperature=float(result.temperature),
                calibrated=float(result.temperature) != 1.0,
            )
        )
        decision = result.decision
        self.session.add(
            RoutingDecisionModel(
                prediction_id=prediction.id,
                action=result.action,
                policy=str(decision.get("policy", "unknown")),
                risk=float(decision.get("risk", result.risk.get("value", 0.0))),
                explanation=str(decision.get("explanation", ""))[:4000],
                reason_codes=list(decision.get("reason_codes", [])),
                utilities=list(decision.get("utilities", [])),
            )
        )
        for diagnostic in result.diagnostics:
            self.session.add(
                DiagnosticRecordModel(
                    prediction_id=prediction.id,
                    code=str(diagnostic.get("code", "UNKNOWN")),
                    severity=str(diagnostic.get("severity", "info")),
                    evidence=diagnostic.get("evidence", {}),
                    explanation=str(diagnostic.get("explanation", ""))[:4000],
                    recommended_action=str(diagnostic.get("recommended_action", ""))[:2000],
                )
            )
        review_id = None
        if result.action in REVIEW_ACTIONS:
            case = build_review_case_model(result, image_ref=image_ref)
            self.session.add(case)
            review_id = case.review_id
        routing_event(
            result.prediction_id,
            result.action,
            policy=decision.get("policy"),
            risk=result.risk.get("value"),
            drift=result.drift.get("severity"),
        )
        return review_id

    def _persist_drift_event(self, result: InferenceResult) -> None:
        drift = result.drift
        self.session.add(
            DriftEventModel(
                severity=str(drift.get("severity", "none")),
                score=float(drift.get("score", 0.0)),
                window_size=int(drift.get("window_size", 0)),
                status=str(drift.get("status", "ok")),
                details=drift,
                model_id=result.model_id,
            )
        )

    def _record_metrics(self, result: InferenceResult) -> None:
        METRICS.increment("predictions_total")
        METRICS.increment(f"action_{result.action.lower()}")
        METRICS.observe("prediction_latency_ms", result.latency_ms)
        METRICS.set_gauge("last_drift_score", float(result.drift.get("score", 0.0)))
        prediction_event(
            result.prediction_id,
            action=result.action,
            model_id=result.model_id,
            risk=result.risk.get("value"),
            latency_ms=round(result.latency_ms, 2),
        )


def save_raw_payload(config: AetherConfig, payload: bytes) -> str:
    """Persist bytes and return the storage-relative reference."""
    directory = config.upload_dir
    Path(directory).mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.img"
    atomic_write_bytes(Path(directory) / name, payload)
    return f"uploads/{name}"


def prediction_to_dict(prediction: PredictionModel) -> dict[str, Any]:
    return {
        "prediction_id": prediction.id,
        "action": prediction.action,
        "predicted_index": prediction.predicted_index,
        "predicted_label": prediction.predicted_label,
        "model_id": prediction.model_id,
        "model_version": prediction.model_version,
        "latency_ms": prediction.latency_ms,
        "drift_severity": prediction.drift_severity,
        "risk": prediction.risk,
        "uncertainty": prediction.uncertainty,
        "created_at": prediction.created_at.isoformat(),
        "image_ref": prediction.image_ref,
    }
