"""Drift engine: statistics, severity classification and reference handling."""

from __future__ import annotations

import numpy as np
import pytest

from aether.config import DriftConfig
from aether.drift.embedding import fit_pca
from aether.drift.monitor import DriftMonitor, DriftObservation, RollingWindow
from aether.drift.statistical import (
    bonferroni,
    chi_square_class_frequency,
    kolmogorov_smirnov,
    population_stability_index,
    rbf_mmd,
)


def test_ks_detects_clear_location_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 500)
    current = rng.normal(3, 1, 500)
    result = kolmogorov_smirnov(reference, current)
    assert result.status == "ok"
    assert result.p_value is not None and result.p_value < 1e-6


def test_ks_does_not_flag_identical_distribution():
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 400)
    current = rng.normal(0, 1, 400)
    result = kolmogorov_smirnov(reference, current)
    assert result.p_value is not None and result.p_value > 0.001


def test_ks_requires_minimum_samples():
    result = kolmogorov_smirnov(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert result.status == "insufficient_data"
    assert result.p_value is None


def test_psi_is_zero_for_identical_samples():
    rng = np.random.default_rng(2)
    sample = rng.normal(size=600)
    result = population_stability_index(sample, sample)
    assert result.statistic == pytest.approx(0.0, abs=1e-6)


def test_psi_grows_with_shift():
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 800)
    mild = population_stability_index(reference, rng.normal(0.2, 1, 800))
    severe = population_stability_index(reference, rng.normal(2.0, 1, 800))
    assert mild.statistic < severe.statistic


def test_psi_reports_constant_reference():
    result = population_stability_index(np.ones(100), np.ones(100))
    assert result.status == "constant_feature"


def test_chi_square_flags_changed_class_mix():
    reference = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100], dtype=float)
    current = np.array([400, 0, 0, 100, 100, 100, 100, 100, 100, 0], dtype=float)
    result = chi_square_class_frequency(reference, current)
    assert result.status == "ok"
    assert result.p_value < 0.001


def test_mmd_flags_shift_and_not_null():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(120, 5))
    same = rng.normal(size=(120, 5))
    shifted = rng.normal(loc=2.5, size=(120, 5))
    null = rbf_mmd(reference, same, permutations=40, seed=0)
    alternative = rbf_mmd(reference, shifted, permutations=40, seed=0)
    assert alternative.statistic > null.statistic
    assert alternative.p_value < 0.05 <= 1.0


def test_pca_projection_shape_and_ordering():
    rng = np.random.default_rng(5)
    embeddings = rng.normal(size=(60, 12))
    basis = fit_pca(embeddings, n_components=5)
    projected = basis.transform(embeddings)
    assert projected.shape == (60, 5)
    assert np.all(np.diff(basis.explained_variance_ratio) <= 1e-9)


def test_bonferroni_adjustment():
    assert bonferroni([0.01, 0.02]) == pytest.approx([0.02, 0.04])
    assert bonferroni([0.6, 0.9]) == [1.0, 1.0]
    assert bonferroni([]) == []


def _observations(reference, count: int, shift: float = 0.0, seed: int = 0) -> list[DriftObservation]:
    """Draw a window from the reference distribution, optionally translated by ``shift``.

    Sampling the *reference* spread (not a narrow band around its mean) is what makes this a
    valid no-shift null: a narrower window is itself a distribution shift and is correctly
    flagged by the detectors.
    """
    rng = np.random.default_rng(seed)
    histogram = reference.class_histogram / reference.class_histogram.sum()
    classes = rng.choice(10, size=count, p=histogram)
    stat_std = reference.input_stats.std(axis=0) + 1e-6
    embedding_std = reference.embeddings.std(axis=0) + 1e-6
    rng.shuffle(classes, axis=0)
    rows = []
    for index in range(count):
        rows.append(
            DriftObservation(
                input_stats=reference.input_stats.mean(axis=0)
                + rng.normal(0, 1, 6) * stat_std
                + shift,
                embedding=reference.embeddings.mean(axis=0)
                + rng.normal(0, 1, reference.embeddings.shape[1]) * embedding_std
                + shift,
                entropy=float(rng.normal(reference.entropy.mean(), reference.entropy.std() + 1e-6)),
                max_probability=float(
                    np.clip(
                        rng.normal(
                            reference.max_probability.mean(),
                            reference.max_probability.std() + 1e-6,
                        ),
                        0.0,
                        1.0,
                    )
                ),
                predicted_class=int(classes[index]),
            )
        )
    return rows


def test_monitor_returns_insufficient_window(reference_artifact):
    config = DriftConfig(evaluation_window_size=200, min_window_size=50)
    monitor = DriftMonitor(reference_artifact, config, num_classes=10)
    report = monitor.evaluate(_observations(reference_artifact, 10))
    assert report.status == "insufficient_window"
    assert report.severity == "none"
    assert report.warnings


def test_monitor_detects_severe_shift(reference_artifact):
    config = DriftConfig(evaluation_window_size=120, min_window_size=50)
    monitor = DriftMonitor(reference_artifact, config, num_classes=10)
    report = monitor.evaluate(_observations(reference_artifact, 120, shift=1.5))
    assert report.severity in ("moderate", "severe")
    assert report.score > 0
    payload = report.to_dict()
    assert "cannot be inferred" in payload["interpretation"]


def test_monitor_clean_window_is_not_flagged(reference_artifact):
    config = DriftConfig(evaluation_window_size=120, min_window_size=50)
    monitor = DriftMonitor(reference_artifact, config, num_classes=10)
    report = monitor.evaluate(_observations(reference_artifact, 120, shift=0.0))
    assert report.severity in ("none", "low")


def test_severity_thresholds_are_ordered():
    with pytest.raises(ValueError):
        DriftConfig(severity={"low": 5.0, "moderate": 3.0, "severe": 6.0})


def test_rolling_window_bounds_memory():
    window = RollingWindow(maxlen=5)
    reference = np.zeros(4)
    for index in range(10):
        window.add(
            DriftObservation(
                input_stats=reference,
                embedding=reference,
                entropy=0.1,
                max_probability=0.9,
                predicted_class=index % 10,
            )
        )
    assert len(window) == 5
    assert window.snapshot()[-1].predicted_class == 9
