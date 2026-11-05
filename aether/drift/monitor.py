"""Composite drift monitor.

Two independent detection paths are reported separately and never merged into a single
"accuracy dropped" claim:

1. **Input / feature shift** — image-quality statistics and embedding-space tests.
2. **Prediction behaviour** — entropy, confidence and predicted-class distributions.

Severity is derived from Bonferroni-adjusted evidence across detectors: each test contributes
``-log10(p_adj)``, and the composite score is the maximum contribution, so one decisive signal
is not diluted by many quiet ones. PSI and MMD, which have no analytic null distribution, map
their statistic onto the same scale through configurable reference thresholds.

**Scientific constraint**: unlabelled inference traffic can demonstrate that the input
distribution moved. It cannot demonstrate that accuracy fell. This module therefore never
emits an accuracy-drift statement; that requires verified labels.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aether.config import DriftConfig
from aether.drift.embedding import embedding_drift
from aether.drift.reference import INPUT_FEATURE_NAMES, ReferenceStatistics
from aether.drift.statistical import (
    bonferroni,
    chi_square_class_frequency,
    kolmogorov_smirnov,
    population_stability_index,
)
from aether.utils.logging import get_logger

logger = get_logger("aether.drift.monitor")

SEVERITY_LEVELS = ("none", "low", "moderate", "severe")


@dataclass
class DriftObservation:
    """One inference's drift-relevant signals."""

    input_stats: np.ndarray  # (6,)
    embedding: np.ndarray  # (D,)
    entropy: float
    max_probability: float
    predicted_class: int


@dataclass
class DriftReport:
    """Structured drift assessment for a window of observations."""

    severity: str
    score: float
    window_size: int
    signals: list[dict[str, Any]] = field(default_factory=list)
    input_shift: dict[str, Any] = field(default_factory=dict)
    prediction_shift: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    status: str = "ok"

    @property
    def is_shifted(self) -> bool:
        return self.severity in ("moderate", "severe")

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "score": round(self.score, 4),
            "window_size": self.window_size,
            "status": self.status,
            "signals": self.signals,
            "input_shift": self.input_shift,
            "prediction_shift": self.prediction_shift,
            "warnings": self.warnings,
            "interpretation": (
                "Shift is measured without labels. Accuracy impact cannot be inferred from "
                "unlabelled traffic and is not reported here."
            ),
        }


class RollingWindow:
    """Thread-safe bounded window of recent observations."""

    def __init__(self, maxlen: int) -> None:
        self._items: deque[DriftObservation] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, observation: DriftObservation) -> None:
        with self._lock:
            self._items.append(observation)

    def extend(self, observations: list[DriftObservation]) -> None:
        with self._lock:
            self._items.extend(observations)

    def snapshot(self) -> list[DriftObservation]:
        with self._lock:
            return list(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class DriftMonitor:
    """Evaluates a window of observations against a frozen reference."""

    def __init__(
        self,
        reference: ReferenceStatistics,
        config: DriftConfig,
        num_classes: int = 10,
        seed: int = 0,
    ) -> None:
        self.reference = reference
        self.config = config
        self.num_classes = num_classes
        self.seed = seed

    def evaluate(self, observations: list[DriftObservation]) -> DriftReport:
        """Assess a window of observations against the reference."""
        window_size = len(observations)
        warnings: list[str] = []
        if window_size < self.config.min_window_size:
            return DriftReport(
                severity="none",
                score=0.0,
                window_size=window_size,
                status="insufficient_window",
                warnings=[
                    f"window has {window_size} observations; "
                    f"{self.config.min_window_size} required for a drift assessment"
                ],
            )

        input_stats = np.vstack([obs.input_stats for obs in observations])
        embeddings = np.vstack([obs.embedding for obs in observations])
        entropies = np.array([obs.entropy for obs in observations], dtype=np.float64)
        confidences = np.array([obs.max_probability for obs in observations], dtype=np.float64)
        predicted = np.array([obs.predicted_class for obs in observations], dtype=np.int64)

        raw_tests = []
        signals: list[dict[str, Any]] = []

        # --- Path 1a: input quality statistics (PSI, reference quantile bins) -----------
        input_signals: list[dict[str, Any]] = []
        for index, name in enumerate(INPUT_FEATURE_NAMES):
            psi = population_stability_index(
                self.reference.input_stats[:, index], input_stats[:, index],
                bins=self.config.psi_bins,
            )
            if psi.status == "ok" and psi.statistic is not None:
                # 0.25 PSI is the conventional "major shift" rule of thumb; map onto -log10.
                scaled = max(0.0, 1.0 + psi.statistic / 0.25)
                input_signals.append(
                    {
                        "feature": name,
                        "test": "psi",
                        "statistic": round(float(psi.statistic), 4),
                        "status": psi.status,
                        "score": round(float(scaled), 4),
                    }
                )
            else:
                warnings.append(f"input feature '{name}' not testable: {psi.note or psi.status}")
            raw_tests.append(psi.p_value)

        # --- Path 1b: embedding-space drift (KS per component + MMD) --------------------
        embedding_report: dict[str, Any] | None = None
        mmd_p_value: float | None = None
        try:
            embedding_report = embedding_drift(
                self.reference.projected,
                embeddings,
                self.reference.basis(),
                psi_bins=self.config.psi_bins,
                max_samples=self.config.mmd_max_samples,
                permutations=self.config.mmd_permutations,
                seed=self.seed,
            )
        except ValueError as error:
            # A reference is fitted for one model's embedding space. Comparing another model's
            # embeddings to it is invalid, so the path is skipped and reported rather than
            # silently scored against an unrelated basis.
            warnings.append(f"embedding drift skipped: {error}")
        if embedding_report is not None:
            for test in embedding_report["ks_tests"]:
                if test["status"] == "ok" and test["p_value"] is not None:
                    raw_tests.append(test["p_value"])
            mmd_p_value = embedding_report["mmd"].get("p_value")
            if mmd_p_value is not None and embedding_report["mmd"].get("status") == "ok":
                raw_tests.append(mmd_p_value)

        # --- Path 2: prediction behaviour ----------------------------------------------
        entropy_ks = kolmogorov_smirnov(self.reference.entropy, entropies)
        confidence_ks = kolmogorov_smirnov(self.reference.max_probability, confidences)
        class_counts = np.bincount(predicted, minlength=self.num_classes).astype(np.float64)
        chi2 = chi_square_class_frequency(self.reference.class_histogram, class_counts)
        for test in (entropy_ks, confidence_ks, chi2):
            if test.status == "ok":
                raw_tests.append(test.p_value)
            else:
                warnings.append(f"{test.name} not testable: {test.note or test.status}")

        significant_tests = [
            {
                "test": name,
                "p_value": test.p_value,
                "statistic": test.statistic,
                "status": test.status,
            }
            for name, test in (
                ("entropy_ks", entropy_ks),
                ("confidence_ks", confidence_ks),
                ("class_frequency_chi2", chi2),
            )
            if test.status == "ok"
        ]

        # --- Composite score -----------------------------------------------------------
        adjusted = bonferroni([p for p in raw_tests if p is not None])
        statistical_scores = [-np.log10(max(p, 1e-12)) for p in adjusted]
        composite = float(max(statistical_scores)) if statistical_scores else 0.0
        if embedding_report is not None and mmd_p_value is not None:
            signals.append(
                {
                    "test": "embedding_mmd",
                    "statistic": embedding_report["mmd"].get("statistic"),
                    "p_value": mmd_p_value,
                    "score": round(float(-np.log10(max(mmd_p_value, 1e-12))), 4),
                }
            )
        signals.extend(input_signals)
        signals.extend(significant_tests)
        signals.sort(key=lambda item: -float(item.get("score") or 0.0))

        # MMD and KS tests are correlated (same embedding space): the composite uses the
        # maximum adjusted evidence, and the report keeps every component visible.
        prediction_shift = {
            "entropy_mean": float(entropies.mean()),
            "entropy_reference_mean": float(self.reference.entropy.mean()),
            "confidence_mean": float(confidences.mean()),
            "confidence_reference_mean": float(self.reference.max_probability.mean()),
            "class_histogram": class_counts.astype(int).tolist(),
            "class_histogram_reference": self.reference.class_histogram.astype(int).tolist(),
            "entropy_ks": entropy_ks.to_dict(),
            "confidence_ks": confidence_ks.to_dict(),
            "class_frequency_chi2": chi2.to_dict(),
        }
        # The embedding path can be skipped (reference fitted for another model); report it as
        # absent rather than fabricating a comparison.
        embedding_details: dict[str, Any] = {}
        if embedding_report is not None:
            embedding_details = {
                "mmd": embedding_report["mmd"],
                "significant_components": embedding_report["significant_components"],
                "components_tested": embedding_report["components_tested"],
                "explained_variance": embedding_report["explained_variance"],
            }
        input_shift = {
            "feature_means": {
                name: float(input_stats[:, index].mean())
                for index, name in enumerate(INPUT_FEATURE_NAMES)
            },
            "feature_reference_means": {
                name: float(self.reference.input_stats[:, index].mean())
                for index, name in enumerate(INPUT_FEATURE_NAMES)
            },
            "psi": input_signals,
            "embedding": embedding_details,
        }
        # Evaluate the MMD reference threshold as an additional PSI-like signal.
        mmd_statistic = embedding_details.get("mmd", {}).get("statistic")
        if mmd_statistic is not None and mmd_statistic >= 0.05:
            composite = max(composite, 4.0)
            warnings.append(
                "embedding MMD statistic exceeds the 0.05 practical threshold; "
                "treated as strong evidence of feature shift"
            )

        severity = self.classify(composite)
        return DriftReport(
            severity=severity,
            score=composite,
            window_size=window_size,
            signals=signals,
            input_shift=input_shift,
            prediction_shift=prediction_shift,
            warnings=warnings,
            status="ok",
        )

    def classify(self, score: float) -> str:
        """Map a composite drift score onto the configured severity bands."""
        thresholds = self.config.severity
        if score >= thresholds.severe:
            return "severe"
        if score >= thresholds.moderate:
            return "moderate"
        if score >= thresholds.low:
            return "low"
        return "none"
