"""The conformal threshold must actually control selective error, not merely look sensible."""

from __future__ import annotations

import numpy as np
import pytest

from aether.routing.conformal import acceptance_threshold, selective_report


def _synthetic_split(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Confidence-conditional correctness: P(correct | confidence c) = c."""
    confidence = rng.uniform(0.2, 0.99, size)
    correct = rng.random(size) < confidence
    probabilities = np.zeros((size, 2))
    probabilities[:, 0] = confidence
    probabilities[:, 1] = 1.0 - confidence
    labels = np.where(correct, 0, 1)
    return probabilities, labels


def test_threshold_controls_selective_error_within_the_bound() -> None:
    rng = np.random.default_rng(7)
    alpha = 0.10
    errors = []
    for _ in range(40):
        calibration, _ = _synthetic_split(rng, 2000)
        threshold = acceptance_threshold(calibration[:, 0][np.arange(2000) % 2 == 0], alpha)
        evaluation, labels = _synthetic_split(rng, 2000)
        report = selective_report(evaluation, labels, threshold.threshold)
        if report["accepted"]:
            errors.append(report["selective_error"])

    assert errors, "the threshold should accept at least some predictions"
    # Marginal guarantee: the mean over draws should sit at or below alpha (small slack for
    # sampling noise), and no individual draw should be wildly above it.
    assert float(np.mean(errors)) <= alpha + 0.01
    assert float(np.max(errors)) <= alpha + 0.05


def test_tighter_alpha_never_lowers_the_threshold() -> None:
    rng = np.random.default_rng(11)
    probabilities, _ = _synthetic_split(rng, 3000)
    true_class = probabilities.max(axis=1) * 0.9
    loose = acceptance_threshold(true_class, 0.20)
    tight = acceptance_threshold(true_class, 0.05)
    assert tight.threshold >= loose.threshold
    assert tight.rank > loose.rank


def test_impossible_alpha_is_rejected_instead_of_faked() -> None:
    probabilities = np.linspace(0.5, 1.0, 50)
    with pytest.raises(ValueError, match="calibration samples"):
        acceptance_threshold(probabilities, 0.01)


def test_report_matches_the_acceptance_rule() -> None:
    probabilities = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])
    labels = np.array([0, 1, 0])
    report = selective_report(probabilities, labels, threshold=0.5)
    assert report["accepted"] == 2
    assert report["coverage"] == pytest.approx(2 / 3)
    assert report["selective_error"] == pytest.approx(0.0)
    assert report["overall_error"] == pytest.approx(1 / 3)
