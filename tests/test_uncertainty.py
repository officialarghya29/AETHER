"""Uncertainty engine: mathematical correctness and numerical stability."""

from __future__ import annotations

import numpy as np
import pytest

from aether.uncertainty.entropy import (
    log_softmax,
    max_probability,
    normalized_entropy,
    prediction_margin,
    predictive_entropy,
    softmax,
)
from aether.uncertainty.metrics import (
    accuracy,
    aurc,
    brier_score,
    expected_calibration_error,
    macro_f1,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_curve,
    risk_at_coverage,
)


def test_uniform_distribution_has_maximal_entropy():
    num_classes = 10
    uniform = np.full((1, num_classes), 1.0 / num_classes)
    assert predictive_entropy(uniform)[0] == pytest.approx(np.log(num_classes), rel=1e-9)
    assert normalized_entropy(uniform)[0] == pytest.approx(1.0, rel=1e-9)


def test_degenerate_distribution_has_zero_entropy_and_margin_one():
    peaked = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert predictive_entropy(peaked)[0] == pytest.approx(0.0, abs=1e-9)
    assert max_probability(peaked)[0] == pytest.approx(1.0)
    assert prediction_margin(peaked)[0] == pytest.approx(1.0)


def test_entropy_does_not_produce_nan_on_zero_probabilities():
    probabilities = np.array([[0.0, 0.0, 1.0]])
    value = predictive_entropy(probabilities, epsilon=1e-12)
    assert np.isfinite(value).all()
    assert value[0] == pytest.approx(0.0, abs=1e-6)


def test_softmax_is_stable_for_extreme_logits():
    logits = np.array([[1000.0, 1001.0, 999.0]])
    probabilities = softmax(logits)
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.allclose(log_softmax(logits), np.log(probabilities), atol=1e-9)


def test_probabilities_are_shift_invariant_to_logit_offset():
    logits = np.random.default_rng(0).normal(size=(4, 10))
    assert np.allclose(softmax(logits), softmax(logits + 5.0))


def test_margin_matches_top_two_difference():
    probabilities = np.array([[0.4, 0.35, 0.25]])
    assert prediction_margin(probabilities)[0] == pytest.approx(0.05)


def test_entropy_ordering(probabilities):
    entropy = predictive_entropy(probabilities)
    assert entropy[2] < entropy[0] < entropy[1], "peaked must be less uncertain than broad"


def test_brier_score_perfect_and_uniform():
    labels = np.array([0, 1])
    perfect = np.eye(2)[labels]
    assert brier_score(perfect, labels, num_classes=2) == pytest.approx(0.0)
    # With two classes, a uniform prediction scores 0.5 per sample: 0.25 for each component.
    uniform = np.full((2, 2), 0.5)
    assert brier_score(uniform, labels, num_classes=2) == pytest.approx(0.5)


def test_ece_zero_for_perfectly_calibrated_binary_predictor():
    # 100 samples: 80 at confidence 0.8 with 80% accuracy, 20 at confidence 1.0 all correct.
    probabilities = np.concatenate(
        [np.tile([0.8, 0.2], (80, 1)), np.tile([1.0, 0.0], (20, 1))]
    )
    labels = np.concatenate([np.array([0] * 64 + [1] * 16), np.array([0] * 20)])
    assert expected_calibration_error(probabilities, labels, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_overconfidence():
    probabilities = np.tile([0.95, 0.05], (100, 1))
    labels = np.array([0] * 60 + [1] * 40)
    ece = expected_calibration_error(probabilities, labels, n_bins=10)
    assert ece == pytest.approx(0.35, abs=1e-6)
    assert maximum_calibration_error(probabilities, labels, n_bins=10) >= ece


def test_reliability_curve_marks_empty_bins():
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2]])
    rows = reliability_curve(probabilities, np.array([0, 0]), n_bins=5)
    empty = [row for row in rows if row["count"] == 0]
    assert empty and all(row["accuracy"] is None for row in empty)


def test_nll_matches_manual_computation():
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([0, 1])
    expected = -np.mean(np.log([0.9, 0.8]))
    assert negative_log_likelihood(probabilities, labels) == pytest.approx(expected)


def test_accuracy_and_f1_on_known_case():
    predictions = np.array([0, 1, 2, 2])
    labels = np.array([0, 1, 1, 2])
    assert accuracy(predictions, labels) == pytest.approx(0.75)
    assert 0 < macro_f1(predictions, labels, 3) <= 1.0


def test_aurc_prefers_better_ranking():
    correct = np.array([1, 1, 0, 0, 0, 1])
    good_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    bad_scores = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert aurc(correct, good_scores) < aurc(correct, bad_scores)
    assert 0.0 <= risk_at_coverage(correct, good_scores, 0.5) <= 1.0


def test_metrics_handle_empty_input_gracefully():
    empty = np.zeros((0, 3))
    assert np.isnan(expected_calibration_error(empty, np.array([], dtype=int), 5))
    assert np.isnan(accuracy(np.array([]), np.array([])))
