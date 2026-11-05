"""Risk-controlled acceptance thresholds for selective prediction.

Temperature scaling makes a probability *honest*. This module makes the **acceptance threshold**
honest in the only way that is available at serving time: from labelled calibration predictions,
choose the most permissive confidence cut-off whose selective error is still consistent with the
requested budget ``alpha``.

The procedure is a one-sided **Wilson upper confidence bound** on the error rate among accepted
calibration predictions:

1. Rank calibration predictions by predicted-class confidence.
2. For every prefix (a candidate threshold) compute accepted count and accepted errors.
3. Take the largest prefix whose Wilson upper bound at confidence ``1 - alpha`` is at most ``alpha``.

This is a calibration-set risk target, not a distribution-free conformal guarantee. It assumes the
accepted calibration predictions are exchangeable with accepted served predictions; it does not
survive distribution shift, which is why the stability study reports clean and corrupted streams
separately instead of pooling them. Asking for a budget the calibration set cannot support raises
rather than returning a threshold that guarantees nothing.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RiskControlledThreshold:
    """An acceptance threshold with the evidence behind it."""

    alpha: float
    threshold: float
    samples: int
    accepted: int
    calibration_risk: float
    calibration_coverage: float
    upper_bound: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _wilson_upper(errors: int, accepted: int, alpha: float) -> float:
    """One-sided Wilson upper bound on a binomial error rate."""
    if accepted <= 0:
        return 1.0
    if errors <= 0:
        return 1.0 - alpha ** (1.0 / accepted)
    proportion = errors / accepted
    z = NormalDist().inv_cdf(1.0 - alpha)
    denominator = 1.0 + z * z / accepted
    centre = proportion + z * z / (2.0 * accepted)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / accepted + z * z / (4.0 * accepted * accepted)
    )
    return min(1.0, (centre + margin) / denominator)


def acceptance_threshold(
    confidences: np.ndarray,
    correct: np.ndarray,
    alpha: float,
    *,
    minimum_samples: int | None = None,
) -> RiskControlledThreshold:
    """Return the lowest confidence threshold whose selective error is within ``alpha``.

    ``confidences`` are predicted-class probabilities and ``correct`` the matching correctness
    booleans, both from data the model was not trained on. The threshold is the confidence of the
    last accepted calibration prediction, so the router's rule is ``confidence >= threshold``.
    """
    probability = np.asarray(confidences, dtype=np.float64)
    hits = np.asarray(correct, dtype=bool)
    if probability.ndim != 1 or hits.ndim != 1:
        raise ValueError("confidences and correct must be 1-D arrays")
    if probability.shape[0] != hits.shape[0]:
        raise ValueError("confidences and correct must have the same length")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError("confidences must be finite probabilities in [0, 1]")

    samples = int(probability.shape[0])
    required = minimum_samples if minimum_samples is not None else math.ceil(2.0 / alpha)
    if samples < max(1, required):
        raise ValueError(
            f"alpha={alpha} needs at least {required} calibration samples, got {samples}; "
            "the budget cannot be supported by this split"
        )

    order = np.argsort(-probability, kind="stable")
    sorted_confidence = probability[order]
    errors = (~hits[order]).astype(np.int64)
    cumulative_errors = np.cumsum(errors)

    best_index: int | None = None
    best_bound = 1.0
    for index in range(samples):
        accepted = index + 1
        # Ties at the same confidence must be accepted together.
        if index + 1 < samples and sorted_confidence[index + 1] == sorted_confidence[index]:
            continue
        bound = _wilson_upper(int(cumulative_errors[index]), accepted, alpha)
        if bound <= alpha:
            best_index = index
            best_bound = bound

    if best_index is None:
        raise ValueError(
            f"no feasible threshold at alpha={alpha} with {samples} calibration samples"
        )

    accepted = best_index + 1
    risk = float(cumulative_errors[best_index] / accepted)
    return RiskControlledThreshold(
        alpha=float(alpha),
        threshold=float(sorted_confidence[best_index]),
        samples=samples,
        accepted=accepted,
        calibration_risk=risk,
        calibration_coverage=accepted / samples,
        upper_bound=float(min(1.0, best_bound)),
        confidence=1.0 - alpha,
    )


def selective_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Measure coverage and selective error at ``threshold`` on labelled data.

    ``probabilities`` is ``(n, C)``; a sample is accepted when its maximum probability reaches the
    threshold, which is the rule the router applies.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a 2-D (n, classes) array")
    if probs.shape[0] != truth.shape[0]:
        raise ValueError("probabilities and labels must have the same length")

    predicted = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = predicted == truth
    accepted = confidence >= threshold
    accepted_count = int(accepted.sum())
    return {
        "threshold": float(threshold),
        "samples": int(truth.shape[0]),
        "coverage": float(accepted_count / max(1, truth.shape[0])),
        "selective_error": (
            float(1.0 - correct[accepted].mean()) if accepted_count else float("nan")
        ),
        "overall_error": float(1.0 - correct.mean()),
        "accepted": accepted_count,
    }
