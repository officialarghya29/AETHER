"""Classification and calibration evaluation.

Functions here take probabilities and integer labels and return plain dictionaries, which keeps
them reusable by the API, the CLI and the experiment harness without hidden model dependencies.
"""

from __future__ import annotations

import numpy as np

from aether.uncertainty.metrics import (
    accuracy,
    aurc,
    brier_score,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    maximum_calibration_error,
    negative_log_likelihood,
    per_class_accuracy,
    reliability_curve,
)


def classification_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int = 10,
    n_bins: int = 15,
    include_curves: bool = True,
) -> dict:
    """Accuracy, per-class metrics, calibration and confidence summaries."""
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must have equal length")
    predictions = np.argmax(probs, axis=1)
    confidences = probs.max(axis=1)
    correct = (predictions == labels).astype(np.float64)
    matrix = confusion_matrix(predictions, labels, num_classes)
    report = {
        "samples": int(labels.size),
        "accuracy": accuracy(predictions, labels),
        "macro_f1": macro_f1(predictions, labels, num_classes),
        "ece": expected_calibration_error(probs, labels, n_bins),
        "mce": maximum_calibration_error(probs, labels, n_bins),
        "brier": brier_score(probs, labels, num_classes),
        "nll": negative_log_likelihood(probs, labels),
        "mean_confidence": float(confidences.mean()),
        "confidence_gap": float(confidences.mean() - correct.mean()),
        "mean_entropy": float(
            -np.sum(np.clip(probs, 1e-12, 1) * np.log(np.clip(probs, 1e-12, 1)), axis=1).mean()
        ),
        "aurc": aurc(correct, confidences),
        "per_class_accuracy": {
            int(index): value for index, value in per_class_accuracy(predictions, labels, num_classes).items()
        },
        "confusion_matrix": matrix.tolist(),
    }
    if include_curves:
        report["reliability_curve"] = reliability_curve(probs, labels, n_bins)
    return report


def selective_report(
    probabilities: np.ndarray, labels: np.ndarray, coverages: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 1.0)
) -> dict:
    """Risk at fixed coverage levels under confidence-based selection."""
    from aether.uncertainty.metrics import risk_at_coverage

    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    correct = (np.argmax(probs, axis=1) == labels).astype(np.float64)
    confidences = probs.max(axis=1)
    return {
        "aurc": aurc(correct, confidences),
        "risk_at_coverage": {
            f"{coverage:.2f}": risk_at_coverage(correct, confidences, coverage)
            for coverage in coverages
        },
    }
