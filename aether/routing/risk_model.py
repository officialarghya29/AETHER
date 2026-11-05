"""Learned error-risk model.

A logistic regression maps observable inference-time evidence to ``P(prediction is wrong)``.
It is deliberately a small, interpretable, monotone-ish linear model rather than a black box:
routing decisions must be explainable to an operator, and a linear risk model can be audited,
versioned and refitted in seconds.

Features are available *at inference time* — no labels, no future information:

    entropy, max_probability, margin, shifted_flag, disagreement

``drift_score`` is deliberately excluded. Development data provides only two drift levels
(clean and corrupted), so a coefficient on the raw magnitude is unidentified: the column is
near-collinear with the binary flag and carries no continuous variation to estimate against.
``shifted_flag`` captures the signal the data can actually support.

The model is fitted on the routing-development split only. Test data is never used to fit or
select it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aether.errors import ArtifactNotFoundError
from aether.utils.io import atomic_write_json, read_json
from aether.utils.logging import get_logger

logger = get_logger("aether.routing.risk_model")

FIT_SPLITS: tuple[str, ...] = ("calibration", "routing_dev")

FEATURE_NAMES: tuple[str, ...] = (
    "entropy",
    "max_probability",
    "margin",
    "shifted_flag",
    "disagreement",
)


@dataclass(frozen=True)
class RiskModelArtifact:
    """Fitted coefficients plus the evidence needed to audit the fit."""

    model_id: str
    coefficients: dict[str, float]
    intercept: float
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    split: str
    samples: int
    positive_rate: float
    metrics: dict[str, float] = field(default_factory=dict)
    fitted_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def features_from_evidence(
    *,
    entropy: float,
    max_probability: float,
    margin: float,
    drift_score: float,
    severity: str,
    disagreement: float | None,
) -> dict[str, float]:
    """Build the fixed-order feature dict used by the risk model."""
    return {
        "entropy": float(entropy),
        "max_probability": float(max_probability),
        "margin": float(margin),
        "drift_score": float(drift_score),
        "shifted_flag": 1.0 if severity in ("moderate", "severe") else 0.0,
        "disagreement": float(disagreement) if disagreement is not None else 0.0,
    }


class RiskModel:
    """Logistic risk estimator with explicit standardisation."""

    def __init__(self, artifact: RiskModelArtifact) -> None:
        self.artifact = artifact
        missing = [name for name in FEATURE_NAMES if name not in artifact.coefficients]
        if missing:
            raise ValueError(f"risk model artifact missing coefficients for {missing}")

    @classmethod
    def fit(
        cls,
        features: dict[str, np.ndarray],
        labels: np.ndarray,
        *,
        model_id: str,
        split: str = "calibration",
        l2: float = 1.0,
    ) -> RiskModel:
        """Fit a standardised logistic regression on inference-time features."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, roc_auc_score

        # Refitted models are fitted on the calibration split; routing-dev is permitted so an
        # existing artifact can be reproduced. The test split never participates in fitting, and
        # the routing development split is the only place routing decisions are evaluated.
        if split not in FIT_SPLITS:
            raise ValueError(
                "the risk model must be fitted on the calibration or routing-development split "
                f"(got '{split}'): test data is never used for fitting"
            )
        matrix = np.column_stack([np.asarray(features[name], dtype=np.float64) for name in FEATURE_NAMES])
        labels = np.asarray(labels, dtype=np.int64)
        if matrix.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must have equal length")
        if labels.min() == labels.max():
            raise ValueError("risk model requires both correct and incorrect examples")
        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0)
        stds = np.where(stds < 1e-9, 1.0, stds)
        standardized = (matrix - means) / stds
        estimator = LogisticRegression(C=1.0 / max(l2, 1e-6), max_iter=2000, solver="lbfgs")
        estimator.fit(standardized, labels)
        probabilities = estimator.predict_proba(standardized)[:, 1]
        artifact = RiskModelArtifact(
            model_id=model_id,
            coefficients={
                name: float(value) for name, value in zip(FEATURE_NAMES, estimator.coef_[0], strict=True)
            },
            intercept=float(estimator.intercept_[0]),
            feature_means={name: float(v) for name, v in zip(FEATURE_NAMES, means, strict=True)},
            feature_stds={name: float(v) for name, v in zip(FEATURE_NAMES, stds, strict=True)},
            split=split,
            samples=int(labels.size),
            positive_rate=float(labels.mean()),
            metrics={
                "auc": float(roc_auc_score(labels, probabilities)),
                "brier": float(brier_score_loss(labels, probabilities)),
            },
            fitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        logger.info(
            "risk model fitted",
            extra={
                "model_id": model_id,
                "auc": round(artifact.metrics["auc"], 4),
                "positive_rate": round(artifact.positive_rate, 4),
            },
        )
        return cls(artifact)

    def predict(self, features: dict[str, float]) -> float:
        """Return ``P(error | features)`` for one sample."""
        total = self.artifact.intercept
        for name in FEATURE_NAMES:
            value = float(features.get(name, 0.0))
            mean = self.artifact.feature_means[name]
            std = self.artifact.feature_stds[name] or 1.0
            total += self.artifact.coefficients[name] * ((value - mean) / std)
        return float(1.0 / (1.0 + np.exp(-np.clip(total, -60, 60))))

    def predict_batch(self, rows: list[dict[str, float]]) -> np.ndarray:
        return np.array([self.predict(row) for row in rows], dtype=np.float64)

    def save(self, path: Path) -> None:
        atomic_write_json(path, {"schema_version": 1, "risk_model": self.artifact.to_dict()})
        logger.info("risk model artifact written", extra={"path": str(path)})

    @classmethod
    def load(cls, path: Path) -> RiskModel:
        payload = read_json(path, default=None)
        if not payload:
            raise ArtifactNotFoundError(
                f"risk model artifact not found: {path.name}",
                details={"expected": str(path), "remedy": "run training/calibrate.py"},
            )
        return cls(RiskModelArtifact(**payload["risk_model"]))

    def describe(self) -> dict[str, Any]:
        return self.artifact.to_dict()


def default_risk(evidence: dict[str, float], fallback: float = 0.5) -> float:
    """Heuristic risk when no fitted model is available.

    Used only to keep the service functional; the API reports ``risk_source='heuristic'`` so a
    caller can tell the difference between a fitted estimate and a fallback.
    """
    entropy = evidence.get("entropy", 0.0)
    confidence = evidence.get("max_probability", 1.0)
    drift = evidence.get("drift_score", 0.0)
    risk = 0.15 + 0.55 * min(1.0, entropy) * (1.0 - min(1.0, confidence) * 0.5)
    risk += 0.02 * min(drift, 3.0)
    return float(min(max(risk, 0.01), 0.99) if fallback is None else risk)
