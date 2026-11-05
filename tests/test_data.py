"""Data pipeline: split integrity, corruption determinism and preprocessing consistency."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aether.config import SplitSizes
from aether.data.cifar import build_stratified_splits
from aether.data.corruptions import (
    CORRUPTION_FAMILIES,
    SEVERITIES,
    apply_corruption,
    corruption_parameters,
)
from aether.data.preprocessing import prepare_model_input, resize_batch


@pytest.fixture
def labels() -> np.ndarray:
    return np.repeat(np.arange(10), 500)


def test_splits_are_disjoint_and_complete(labels):
    sizes = SplitSizes(train=2000, validation=500, calibration=500, routing_dev=1000, holdout=1000)
    splits = build_stratified_splits(labels, sizes, seed=42)
    all_indices = np.concatenate([splits[name] for name in splits])
    assert all_indices.size == sizes.total
    assert np.unique(all_indices).size == sizes.total, "splits must not overlap"
    for name, size in (
        ("train", sizes.train),
        ("validation", sizes.validation),
        ("calibration", sizes.calibration),
        ("routing_dev", sizes.routing_dev),
        ("holdout", sizes.holdout),
    ):
        assert splits[name].size == size


def test_splits_are_stratified(labels):
    sizes = SplitSizes(train=2000, validation=500, calibration=500, routing_dev=1000, holdout=1000)
    splits = build_stratified_splits(labels, sizes, seed=7)
    for name, indices in splits.items():
        counts = np.bincount(labels[indices], minlength=10)
        assert counts.min() == counts.max(), f"{name} is not class-balanced"


def test_split_generation_is_deterministic(labels):
    sizes = SplitSizes(train=2000, validation=500, calibration=500, routing_dev=1000, holdout=1000)
    first = build_stratified_splits(labels, sizes, seed=3)
    second = build_stratified_splits(labels, sizes, seed=3)
    for name in first:
        assert np.array_equal(first[name], second[name])
    third = build_stratified_splits(labels, sizes, seed=4)
    assert not np.array_equal(first["train"], third["train"])


def test_oversized_request_is_rejected(labels):
    sizes = SplitSizes(train=40000, validation=5000, calibration=5000, routing_dev=5000, holdout=5000)
    with pytest.raises(ValueError):
        build_stratified_splits(labels, sizes, seed=1)


def test_every_corruption_preserves_shape_and_dtype():
    rng = np.random.default_rng(0)
    images = (rng.random((3, 32, 32, 3)) * 255).astype(np.uint8)
    for family in CORRUPTION_FAMILIES:
        for severity in SEVERITIES:
            corrupted = apply_corruption(images, family, severity, seed=1)
            assert corrupted.shape == images.shape
            assert corrupted.dtype == np.uint8
            assert np.isfinite(corrupted.astype(np.float64)).all()


def test_corruption_is_deterministic_and_severity_sensitive():
    rng = np.random.default_rng(0)
    images = (rng.random((2, 32, 32, 3)) * 255).astype(np.uint8)
    for family in CORRUPTION_FAMILIES:
        low = apply_corruption(images, family, 1, seed=5)
        again = apply_corruption(images, family, 1, seed=5)
        high = apply_corruption(images, family, 5, seed=5)
        assert np.array_equal(low, again), f"{family} is not deterministic"
        assert not np.array_equal(low, high), f"{family} ignores severity"


def test_severity_parameters_are_monotone_for_blur_and_noise():
    assert corruption_parameters("gaussian_noise", 5) > corruption_parameters("gaussian_noise", 1)
    assert corruption_parameters("gaussian_blur", 5) > corruption_parameters("gaussian_blur", 1)


def test_unknown_corruption_is_rejected():
    with pytest.raises(ValueError):
        corruption_parameters("sunshine", 3)
    with pytest.raises(ValueError):
        corruption_parameters("gaussian_noise", 9)


def test_preprocessing_is_consistent_between_train_and_eval():
    rng = np.random.default_rng(2)
    images = (rng.random((4, 32, 32, 3)) * 255).astype(np.uint8)
    eval_batch = prepare_model_input(images, 64)
    generator = torch.Generator().manual_seed(0)
    train_batch = prepare_model_input(images, 64, train=True, generator=generator)
    assert eval_batch.shape == train_batch.shape == (4, 3, 64, 64)
    # Training augments (crop/flip) but must stay in the same normalised range.
    tolerance = 4.0
    assert train_batch.min() > eval_batch.min() - tolerance
    assert train_batch.max() < eval_batch.max() + tolerance


def test_preprocessing_requires_generator_when_augmenting():
    images = np.zeros((1, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        prepare_model_input(images, 64, train=True)


def test_resize_is_noop_at_matching_size():
    batch = torch.zeros(1, 3, 32, 32)
    assert resize_batch(batch, 32) is batch


def test_preprocessing_rejects_wrong_channel_count():
    with pytest.raises(ValueError):
        prepare_model_input(np.zeros((1, 32, 32, 4), dtype=np.uint8), 64)
