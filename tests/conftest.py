"""Shared pytest fixtures.

Tests never touch the real artifact directories and never require the downloaded dataset or the
trained production checkpoints. Where a model is needed, a small randomly-initialised checkpoint
is created on the fly: that is a plumbing fixture, not evidence of model quality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aether.config import load_config  # noqa: E402


def _update(block, **changes):
    return block.model_copy(update=changes)


@pytest.fixture(scope="session")
def base_config():
    return load_config()


@pytest.fixture
def temp_config(tmp_path: Path, base_config):
    """A config whose every writable path points into ``tmp_path``."""
    artifacts = tmp_path / "artifacts"
    models = base_config.models.model_copy(
        update={
            "lightweight": _update(
                base_config.models.lightweight,
                checkpoint=str(artifacts / "checkpoints" / "light.pt"),
                input_size=32,
            ),
            "advanced": _update(
                base_config.models.advanced,
                architecture="mobilenet_v3_small",
                checkpoint=str(artifacts / "checkpoints" / "adv.pt"),
                input_size=32,
            ),
        }
    )
    return base_config.model_copy(
        update={
            "data": _update(base_config.data, root=str(tmp_path / "data"), image_size=32),
            "models": models,
            "uncertainty": _update(
                base_config.uncertainty,
                temperature_artifact=str(artifacts / "calibration" / "temperatures.json"),
            ),
            "drift": _update(
                base_config.drift,
                reference_artifact=str(artifacts / "reference" / "reference.npz"),
                evaluation_window_size=40,
                min_window_size=20,
                embedding_components=8,
                mmd_permutations=10,
                mmd_max_samples=60,
            ),
            "routing": _update(
                base_config.routing,
                policy="cost_aware",
                learned_policy=_update(
                    base_config.routing.learned_policy,
                    artifact=str(artifacts / "calibration" / "learned_policy.json"),
                ),
            ),
            "storage": _update(
                base_config.storage,
                database_url=f"sqlite:///{tmp_path / 'test.db'}",
                artifacts_root=str(artifacts),
                upload_dir=str(artifacts / "uploads"),
            ),
            "api": _update(base_config.api, rate_limit_per_minute=10000),
        }
    )


def write_tiny_checkpoint(
    path: Path,
    *,
    architecture: str = "mobilenet_v3_small",
    num_classes: int = 10,
    input_size: int = 32,
    version: str = "test",
) -> Path:
    """Create a small checkpoint with a fixed seed for plumbing tests."""
    from aether.adapters.torchvision_adapter import build_architecture

    torch.manual_seed(0)
    model = build_architecture(architecture, num_classes, pretrained=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": {
                "architecture": architecture,
                "num_classes": num_classes,
                "input_size": input_size,
                "version": version,
                "classes": [
                    "airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck",
                ],
                "val_accuracy": None,
            },
        },
        path,
    )
    return path


@pytest.fixture
def tiny_models(temp_config):
    """Populate both configured checkpoints with small random models."""
    from aether.adapters.torchvision_adapter import load_checkpoint_payload

    light = temp_config.resolve(temp_config.models.lightweight.checkpoint)
    advanced = temp_config.resolve(temp_config.models.advanced.checkpoint)
    if not light.exists():
        write_tiny_checkpoint(light)
    if not advanced.exists():
        write_tiny_checkpoint(advanced)
    assert load_checkpoint_payload(light)["metadata"]["architecture"] == "mobilenet_v3_small"
    return temp_config


@pytest.fixture
def images() -> np.ndarray:
    """Deterministic random images for plumbing tests only."""
    rng = np.random.default_rng(7)
    return (rng.random((6, 32, 32, 3)) * 255).astype(np.uint8)


@pytest.fixture
def probabilities() -> np.ndarray:
    return np.array(
        [
            [0.70, 0.10, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01],
            [0.25, 0.25, 0.10, 0.10, 0.08, 0.07, 0.05, 0.05, 0.03, 0.02],
            [0.99, 0.005, 0.001, 0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0005, 0.0005],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def reference_artifact(temp_config, tiny_models):
    """Build a small reference artifact from deterministic synthetic embeddings."""
    from aether.drift.reference import build_reference, save_reference

    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(120, 16))
    probabilities = rng.dirichlet(np.ones(10), size=120)
    input_stats = rng.normal(loc=0.4, scale=0.1, size=(120, 6))
    reference = build_reference(
        model_id="test:model",
        input_stats=input_stats,
        embeddings=embeddings,
        probabilities=probabilities,
        predicted_labels=probabilities.argmax(axis=1),
        num_classes=10,
        n_components=6,
        source="synthetic fixture",
    )
    path = temp_config.resolve(temp_config.drift.reference_artifact)
    save_reference(reference, path)
    return reference
