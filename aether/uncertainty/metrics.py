"""Evaluation metrics for calibration and classification."""

from __future__ import annotations

import numpy as np


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of exact matches."""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if predictions.size == 0:
        return float("nan")
    return float(np.mean(predictions == labels))


def brier_score(probabilities: np.ndarray, labels: np.ndarray, num_classes: int | None = None) -> float:
    """Multiclass Brier score: mean squared error between probabilities and one-hot labels."""
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a 2-D (N, C) array")
    if num_classes is None:
        num_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(probs.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray, epsilon: float = 1e-12) -> float:
    """Mean negative log-likelihood of the true label."""
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), epsilon, 1.0)
    labels = np.asarray(labels, dtype=np.int64)
    return float(-np.mean(np.log(probs[np.arange(probs.shape[0]), labels])))


def _bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def reliability_curve(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> list[dict[str, float | None]]:
    """Equal-width reliability diagram bins.

    Bins with no samples are returned with ``count=0`` and ``None`` accuracy so the frontend can
    break the line instead of plotting a fabricated point.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(np.float64)
    edges = _bin_edges(n_bins)
    rows: list[dict[str, float | None]] = []
    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        # Include the right edge in the final bin so confidence == 1.0 is counted.
        mask = (confidences >= low) & (
            confidences <= high if index == n_bins - 1 else confidences < high
        )
        count = int(mask.sum())
        rows.append(
            {
                "bin_lower": float(low),
                "bin_upper": float(high),
                "count": count,
                "avg_confidence": float(confidences[mask].mean()) if count else None,
                "accuracy": float(correct[mask].mean()) if count else None,
                "gap": (
                    float(abs(correct[mask].mean() - confidences[mask].mean()))
                    if count
                    else None
                ),
            }
        )
    return rows


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """``ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|`` over equal-width bins."""
    probs = np.asarray(probabilities, dtype=np.float64)
    total = probs.shape[0]
    if total == 0:
        return float("nan")
    return float(
        sum(row["count"] / total * row["gap"] for row in reliability_curve(probs, labels, n_bins) if row["count"])
    )


def maximum_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Largest per-bin calibration gap (sensitive to small but badly calibrated bins)."""
    gaps = [
        float(row["gap"])
        for row in reliability_curve(probabilities, labels, n_bins)
        if row["count"] and row["gap"] is not None
    ]
    return max(gaps, default=0.0)


def per_class_accuracy(
    predictions: np.ndarray, labels: np.ndarray, num_classes: int
) -> dict[int, float]:
    """Accuracy restricted to each true class."""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    result: dict[int, float] = {}
    for class_index in range(num_classes):
        mask = labels == class_index
        result[class_index] = float(np.mean(predictions[mask] == class_index)) if mask.any() else float("nan")
    return result


def confusion_matrix(
    predictions: np.ndarray, labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """Row-major confusion matrix with true labels on rows."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, prediction in zip(np.asarray(labels), np.asarray(predictions), strict=True):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def macro_f1(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Macro-averaged F1 over classes present in the labels."""
    matrix = confusion_matrix(predictions, labels, num_classes)
    f1_scores: list[float] = []
    for class_index in range(num_classes):
        true_positive = matrix[class_index, class_index]
        false_positive = matrix[:, class_index].sum() - true_positive
        false_negative = matrix[class_index, :].sum() - true_positive
        if true_positive + false_positive + false_negative == 0:
            continue
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores)) if f1_scores else float("nan")


def risk_coverage_curve(
    correct: np.ndarray, scores: np.ndarray, steps: int = 100
) -> list[dict[str, float]]:
    """Selective-prediction curve: risk (error rate) as a function of coverage.

    Samples are accepted from highest score to lowest, so coverage *k* means the *k* most
    confident fraction of the data is answered automatically.
    """
    correct = np.asarray(correct, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if correct.size == 0:
        return []
    order = np.argsort(-scores)
    sorted_correct = correct[order]
    cumulative_errors = np.cumsum(1.0 - sorted_correct)
    rows: list[dict[str, float]] = []
    total = correct.size
    for step in range(1, steps + 1):
        covered = max(1, int(round(total * step / steps)))
        rows.append(
            {
                "coverage": covered / total,
                "risk": float(cumulative_errors[covered - 1] / covered),
                "threshold": float(scores[order][covered - 1]),
            }
        )
    return rows


def aurc(correct: np.ndarray, scores: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better), trapezoid over coverage."""
    curve = risk_coverage_curve(correct, scores, steps=100)
    if not curve:
        return float("nan")
    coverages = np.array([0.0, *[row["coverage"] for row in curve]])
    risks = np.array([0.0, *[row["risk"] for row in curve]])
    return float(np.trapezoid(risks, coverages))


def risk_at_coverage(correct: np.ndarray, scores: np.ndarray, target: float) -> float:
    """Risk at a fixed coverage target (interpolated along the curve)."""
    curve = risk_coverage_curve(correct, scores, steps=200)
    if not curve:
        return float("nan")
    coverages = np.array([row["coverage"] for row in curve])
    risks = np.array([row["risk"] for row in curve])
    return float(np.interp(target, coverages, risks))
