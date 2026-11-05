"""The acceptance threshold must control selective error, not merely look sensible."""

from __future__ import annotations

import numpy as np
import pytest

from aether.routing.risk_control import acceptance_threshold, selective_report


def _synthetic_split(
    rng: np.random.Generator, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """A two-class model whose confidence is calibrated: P(correct | confidence c) = c.

    Class 0 is always the arg-max, so the predicted-class probability is ``c`` whether or not the
    prediction is correct; the label decides correctness.
    """
    confidence = rng.uniform(0.5, 0.99, size)
    correct = rng.random(size) < confidence
    probabilities = np.stack([confidence, 1.0 - confidence], axis=1)
    labels = np.where(correct, 0, 1)
    return probabilities, labels


def _threshold_inputs(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return probabilities.max(axis=1), probabilities.argmax(axis=1) == labels


def test_threshold_controls_selective_error_within_the_budget() -> None:
    rng = np.random.default_rng(7)
    alpha = 0.10
    observed = []
    for _ in range(40):
        calibration, calibration_labels = _synthetic_split(rng, 2000)
        confidence, correct = _threshold_inputs(calibration, calibration_labels)
        threshold = acceptance_threshold(confidence, correct, alpha)
        evaluation, labels = _synthetic_split(rng, 2000)
        report = selective_report(evaluation, labels, threshold.threshold)
        if report["accepted"]:
            observed.append(report["selective_error"])

    assert observed, "the threshold should accept at least some predictions"
    # The calibration bound is conservative; held-out error should sit at or below the budget.
    assert float(np.mean(observed)) <= alpha + 0.01
    assert float(np.max(observed)) <= alpha + 0.05


def test_tighter_alpha_never_lowers_the_threshold() -> None:
    rng = np.random.default_rng(11)
    probabilities, labels = _synthetic_split(rng, 3000)
    confidence, correct = _threshold_inputs(probabilities, labels)
    loose = acceptance_threshold(confidence, correct, 0.20)
    tight = acceptance_threshold(confidence, correct, 0.05)
    assert tight.threshold >= loose.threshold
    assert tight.calibration_coverage <= loose.calibration_coverage


def test_impossible_alpha_is_rejected_instead_of_faked() -> None:
    confidence = np.linspace(0.5, 0.99, 50)
    correct = np.ones(50, dtype=bool)
    with pytest.raises(ValueError, match="calibration samples"):
        acceptance_threshold(confidence, correct, 0.01)


def test_report_matches_the_acceptance_rule() -> None:
    probabilities = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])
    labels = np.array([0, 1, 0])
    report = selective_report(probabilities, labels, threshold=0.5)
    assert report["accepted"] == 3
    assert report["coverage"] == pytest.approx(1.0)
    assert report["selective_error"] == pytest.approx(1 / 3)
    assert report["overall_error"] == pytest.approx(1 / 3)


def test_all_wrong_confident_predictions_push_the_threshold_up() -> None:
    """A calibration split where confidence is uninformative must not buy coverage."""
    confidence = np.full(1000, 0.95)
    correct = np.zeros(1000, dtype=bool)
    with pytest.raises(ValueError, match="no feasible threshold"):
        acceptance_threshold(confidence, correct, 0.05)
