"""Stability guards: TTA merging, non-finite output rejection, and the config surface."""

from __future__ import annotations

import numpy as np
import pytest

from aether.adapters.base import Prediction
from aether.config import load_config
from aether.errors import InferenceError
from aether.pipeline import ReliabilityPipeline


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
    updated = config.routing.model_copy(update={"tta": True})
    assert updated.tta is True
