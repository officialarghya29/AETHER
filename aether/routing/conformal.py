"""Split conformal risk control for selective prediction.

Temperature scaling makes a probability *honest*; conformal prediction makes a **guarantee**.
Given exchangeable calibration data, choosing the acceptance threshold from calibration scores
controls the error rate among accepted predictions at ``alpha`` — a finite-sample statement, not an
asymptotic hope.

Construction (split conformal, classification):

1. For each calibration sample compute the nonconformity score ``s = 1 - p(true class)``.
2. Take the ``ceil((n + 1)(1 - alpha))``-th smallest score as ``q``.
3. Accept a prediction when its predicted-class probability is at least ``1 - q``.

Under exchangeability this yields ``P(error | accepted) <= alpha``. The guarantee is marginal over
draws of the calibration and test sets; it is not a per-request guarantee, and it says nothing about
inputs whose distribution has shifted away from the calibration split. Both caveats are deliberate:
a reliability layer that overstates its guarantees is worse than one with none.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalThreshold:
    """A calibrated acceptance threshold with the evidence behind it."""

    alpha: float
    threshold: float
    samples: int
    rank: int
    quantile: float
    error_bound: float

    def to_dict(self) -> dict:
        return asdict(self)


def acceptance_threshold(
    true_class_probabilities: np.ndarray,
    alpha: float,
    *,
    minimum_samples: int | None = None,
) -> ConformalThreshold:
    """Return the confidence threshold that bounds selective error at ``alpha``.

    ``true_class_probabilities`` must come from a split that is disjoint from the data the model was
    trained on and from the traffic being evaluated. ``n >= 1/alpha - 1`` samples are required for a
    non-trivial quantile; asking for a tighter guarantee than the calibration set can support raises
    instead of silently returning a threshold that guarantees nothing.
    """
    scores = 1.0 - np.asarray(true_class_probabilities, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("true_class_probabilities must be a 1-D array")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not np.all(np.isfinite(scores)):
        raise ValueError("calibration probabilities must be finite")
    if np.any((scores < -1e-9) | (scores > 1 + 1e-9)):
        raise ValueError("probabilities must lie in [0, 1]")

    samples = int(scores.shape[0])
    required = minimum_samples if minimum_samples is not None else math.ceil(1.0 / alpha) - 1
    if samples < max(1, required):
        raise ValueError(
            f"alpha={alpha} needs at least {required} calibration samples, got {samples}; "
            "the guarantee cannot be supported by this split"
        )

    rank = math.ceil((samples + 1) * (1.0 - alpha))
    if rank > samples:
        # Without an upper order statistic at this alpha the honest answer is "no threshold that
        # keeps the guarantee"; callers should collect more calibration data.
        raise ValueError(
            f"alpha={alpha} is too small for {samples} calibration samples: the required rank "
            f"({rank}) exceeds the sample count"
        )
    quantile = float(np.sort(scores)[rank - 1])
    return ConformalThreshold(
        alpha=float(alpha),
        threshold=float(1.0 - quantile),
        samples=samples,
        rank=rank,
        quantile=quantile,
        error_bound=float(alpha),
    )


def selective_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Measure coverage and selective error at ``threshold`` on labelled data.

    ``probabilities`` is ``(n, C)``; a sample is accepted when its maximum probability reaches the
    threshold, which is the same rule the router applies.
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
