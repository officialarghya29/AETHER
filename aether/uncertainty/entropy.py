"""Confidence and uncertainty scores.

These are deliberately separate quantities and the framework never conflates them:

* **Predictive entropy** measures dispersion of the predictive distribution.
* **Maximum softmax probability (MSP)** is a confidence score, not a probability of correctness.
* **Prediction margin** measures separation between the top two classes.

All routines accept either a 1-D probability vector or a 2-D ``(N, C)`` array, preserve the input
rank in their result (scalar for 1-D, array for 2-D) and use a strictly positive epsilon so
``log(0)`` can never occur.
"""

from __future__ import annotations

import numpy as np

EPSILON = 1e-12


def _as_batch(probabilities: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a 2-D view plus whether the caller passed a single vector."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim == 1:
        return probs[np.newaxis, :], True
    if probs.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D probability array, got shape {probs.shape}")
    return probs, False


def _restore(values: np.ndarray, was_flat: bool):
    values = np.asarray(values)
    return values[0] if was_flat else values


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def predictive_entropy(probabilities: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """``H(p) = -sum_i p_i log(p_i + eps)`` in nats."""
    probs, was_flat = _as_batch(probabilities)
    clipped = np.clip(probs, epsilon, 1.0)
    return _restore(-np.sum(clipped * np.log(clipped), axis=-1), was_flat)


def normalized_entropy(probabilities: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """Entropy divided by ``log C`` so scores are comparable across class counts."""
    probs, was_flat = _as_batch(probabilities)
    num_classes = probs.shape[-1]
    if num_classes < 2:
        return _restore(np.zeros(probs.shape[0], dtype=np.float64), was_flat)
    return _restore(predictive_entropy(probs, epsilon) / np.log(num_classes), was_flat)


def max_probability(probabilities: np.ndarray) -> np.ndarray:
    """Maximum softmax probability (confidence)."""
    probs, was_flat = _as_batch(probabilities)
    return _restore(np.max(probs, axis=-1), was_flat)


def prediction_margin(probabilities: np.ndarray) -> np.ndarray:
    """Difference between the top-1 and top-2 probabilities (0 for a single class)."""
    probs, was_flat = _as_batch(probabilities)
    if probs.shape[-1] < 2:
        return _restore(np.zeros(probs.shape[0], dtype=np.float64), was_flat)
    top2 = np.sort(probs, axis=-1)[:, -2:]
    return _restore(top2[:, 1] - top2[:, 0], was_flat)


def uncertainty_summary(
    probabilities: np.ndarray, epsilon: float = EPSILON
) -> dict[str, float]:
    """Bundle the scalar uncertainty scores for one prediction."""
    return {
        "predictive_entropy": float(predictive_entropy(probabilities, epsilon)),
        "normalized_entropy": float(normalized_entropy(probabilities, epsilon)),
        "max_probability": float(max_probability(probabilities)),
        "prediction_margin": float(prediction_margin(probabilities)),
    }
