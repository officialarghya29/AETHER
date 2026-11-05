"""Stability guards: TTA merging, non-finite output rejection, and the config surface."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aether.adapters.base import Prediction
from aether.config import load_config
from aether.errors import InferenceError
from aether.pipeline import ReliabilityPipeline, tta_enabled
from aether.routing.risk_control import threshold_artifact


def _prediction(logits: list[float]) -> Prediction:
    values = np.asarray(logits, dtype=np.float32)
    exp = np.exp(values - values.max())
    return Prediction(
        model_id="advanced:resnet50@1.0.0",
        model_version="1.0.0",
        class_index=int(np.argmax(values)),
        class_label="whatever",
        logits=values,
        probabilities=(exp / exp.sum()).astype(np.float32),
        latency_ms=1.0,
        embedding=None,
    )


def test_tta_averages_logits_and_keeps_probabilities_normalised() -> None:
    merged = ReliabilityPipeline._merge_tta(
        _prediction([3.0, 0.0, 0.0]),
        _prediction([0.0, 0.0, 1.0]),
        ("a", "b", "c"),
    )
    assert np.allclose(merged.logits, [1.5, 0.0, 0.5])
    assert merged.class_index == 0
    assert merged.class_label == "a"
    assert float(merged.probabilities.sum()) == pytest.approx(1.0)
    assert merged.latency_ms == pytest.approx(2.0)


def test_non_finite_model_output_raises_instead_of_being_served() -> None:
    broken = _prediction([0.0, 0.0])
    object.__setattr__(broken, "logits", np.array([np.nan, 0.0], dtype=np.float32))
    with pytest.raises(InferenceError, match="non-finite"):
        ReliabilityPipeline._assert_finite([broken], "advanced")


def test_tta_is_off_by_default_and_validated() -> None:
    config = load_config()
    assert config.routing.tta is False
    assert config.routing.tta_on_drift is True
    updated = config.routing.model_copy(update={"tta": True})
    assert updated.tta is True


def test_tta_is_gated_on_detected_drift() -> None:
    config = load_config()
    enabled = config.model_copy(
        update={"routing": config.routing.model_copy(update={"tta": True})}
    )
    assert tta_enabled(enabled, "none") is False
    assert tta_enabled(enabled, "moderate") is True
    assert tta_enabled(enabled, "severe") is True

    unconditional = enabled.model_copy(
        update={"routing": enabled.routing.model_copy(update={"tta_on_drift": False})}
    )
    assert tta_enabled(unconditional, "none") is True
    assert tta_enabled(config, "severe") is False


def test_api_budget_compares_confidence_with_calibrated_threshold(tmp_path) -> None:
    from backend.app.api.routes import predictions

    threshold_artifact(
        {"lightweight": {0.10: 0.80}, "advanced": {0.10: 0.50}},
        tmp_path / "calibration" / "acceptance_thresholds.json",
    )
    config = SimpleNamespace(artifacts_root=tmp_path)
    result = SimpleNamespace(
        advanced=None, action="ACCEPT", probabilities=[0.02] * 9 + [0.91]
    )
    budget = predictions._risk_budget(result, 0.10, config)
    assert budget["tier"] == "lightweight"
    assert budget["threshold"] == pytest.approx(0.80)
    assert budget["meets_threshold"] is True

    escalated = SimpleNamespace(
        advanced={"prediction": None}, action="ACCEPT", probabilities=[0.55] + [0.05] * 9
    )
    budget = predictions._risk_budget(escalated, 0.10, config)
    assert budget["tier"] == "advanced"
    assert budget["meets_threshold"] is True
