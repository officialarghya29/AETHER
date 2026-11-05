"""Temperature scaling: leakage guards, fitting behaviour and artifact persistence."""

from __future__ import annotations

import numpy as np
import pytest

from aether.uncertainty.calibration import (
    CalibrationStore,
    apply_temperature,
    fit_temperature,
)
from aether.uncertainty.entropy import softmax
from aether.uncertainty.metrics import expected_calibration_error, negative_log_likelihood


def _overconfident_logits(labels: np.ndarray, seed: int = 0, scale: float = 3.0) -> np.ndarray:
    """A ~70%-accurate predictor whose logits are scaled up, the textbook T>1 case.

    Scaling logits does not change the argmax, so accuracy is fixed while confidence becomes
    unjustifiably high — exactly the miscalibration temperature scaling is designed to fix.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=1.0, size=(labels.size, 10))
    correct = rng.random(labels.size) < 0.7
    base[np.flatnonzero(correct), labels[correct]] += 2.5
    return base * scale


def test_temperature_reduces_nll_and_ece_on_miscalibrated_data():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 10, size=2000)
    logits = _overconfident_logits(labels)
    calibration = fit_temperature(logits, labels, model_id="test", split="calibration")
    assert calibration.temperature > 1.0, "overconfident logits require T > 1"
    # The fitted objective is NLL; calibration must improve it, and ECE should drop on data
    # that is miscalibrated purely by logit scale.
    assert calibration.nll_after <= calibration.nll_before + 1e-9
    assert calibration.ece_after < calibration.ece_before
    assert calibration.samples == 2000


def test_fitting_on_test_split_is_refused():
    labels = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="test split"):
        fit_temperature(np.zeros((10, 10)), labels, model_id="m", split="test")


def test_temperature_one_reproduces_softmax():
    logits = np.random.default_rng(0).normal(size=(5, 10))
    assert np.allclose(apply_temperature(logits, 1.0), softmax(logits))


def test_apply_temperature_rejects_non_positive():
    with pytest.raises(ValueError):
        apply_temperature(np.zeros((2, 3)), 0.0)


def test_calibration_is_numerically_stable_with_large_logits():
    labels = np.array([0, 1, 2])
    logits = np.full((3, 10), 1000.0)
    calibration = fit_temperature(logits, labels, model_id="m", split="calibration")
    assert np.isfinite(calibration.temperature)
    assert np.isfinite(calibration.nll_after)


def test_store_round_trips(tmp_path):
    store = CalibrationStore(tmp_path / "temperatures.json")
    labels = np.random.default_rng(1).integers(0, 10, size=300)
    logits = _overconfident_logits(labels)
    calibration = fit_temperature(logits, labels, model_id="role:arch@1.0.0")
    store.save(calibration)
    loaded = store.get("role:arch@1.0.0")
    assert loaded is not None
    assert loaded.temperature == pytest.approx(calibration.temperature)
    assert store.temperature("missing-model", default=1.0) == 1.0


def test_calibration_improves_reported_ece_on_heldout_sample():
    rng = np.random.default_rng(9)
    labels = rng.integers(0, 10, size=1500)
    logits = _overconfident_logits(labels, seed=5, scale=4.0)
    calibration = fit_temperature(logits[:750], labels[:750], model_id="m")
    before = expected_calibration_error(softmax(logits[750:]), labels[750:], 15)
    after = expected_calibration_error(
        apply_temperature(logits[750:], calibration.temperature), labels[750:], 15
    )
    assert after < before
    assert np.isfinite(negative_log_likelihood(apply_temperature(logits[750:], calibration.temperature), labels[750:]))
