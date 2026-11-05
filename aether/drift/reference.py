"""Reference distribution storage.

A reference artifact is the frozen definition of "normal". It contains the raw reference
embeddings (for MMD), a PCA basis fitted only on reference data, the projected reference
sample, input-quality statistics, and the prediction-behaviour distributions (entropy,
confidence, class histogram) produced by the lightweight model on that reference window.

Artifacts are stored as ``.npz`` plus a sidecar JSON manifest so the numbers are auditable
without loading the arrays.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aether.drift.embedding import PCABasis, fit_pca
from aether.errors import ReferenceNotFoundError
from aether.utils.io import atomic_write_json, read_json
from aether.utils.logging import get_logger

logger = get_logger("aether.drift.reference")

#: Names of the image-quality statistics in the order stored in ``input_stats``.
INPUT_FEATURE_NAMES: tuple[str, ...] = (
    "brightness",
    "contrast",
    "sharpness",
    "mean_r",
    "mean_g",
    "mean_b",
)


@dataclass(frozen=True)
class ReferenceStatistics:
    """Frozen reference window and its derived statistics."""

    model_id: str
    input_stats: np.ndarray  # (N, 6)
    embeddings: np.ndarray  # (N, D)
    projected: np.ndarray  # (N, k)
    entropy: np.ndarray  # (N,)
    max_probability: np.ndarray  # (N,)
    class_histogram: np.ndarray  # (C,)
    pca_mean: np.ndarray
    pca_components: np.ndarray
    explained_variance_ratio: np.ndarray
    manifest: dict[str, Any]

    def basis(self) -> PCABasis:
        return PCABasis(
            mean=self.pca_mean,
            components=self.pca_components,
            explained_variance_ratio=self.explained_variance_ratio,
        )


def build_reference(
    *,
    model_id: str,
    input_stats: np.ndarray,
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    predicted_labels: np.ndarray,
    num_classes: int,
    n_components: int = 32,
    source: str = "",
    dataset_fingerprint: str = "",
    seed: int = 42,
) -> ReferenceStatistics:
    """Fit PCA and derive prediction-behaviour distributions from a reference window."""
    from aether.uncertainty.entropy import max_probability, predictive_entropy

    embeddings = np.asarray(embeddings, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if embeddings.shape[0] != probabilities.shape[0]:
        raise ValueError("embeddings and probabilities must describe the same samples")
    if embeddings.shape[0] < 50:
        raise ValueError("reference window must contain at least 50 samples")

    basis = fit_pca(embeddings, n_components=n_components)
    projected = basis.transform(embeddings)
    class_histogram = np.bincount(
        np.asarray(predicted_labels, dtype=np.int64), minlength=num_classes
    ).astype(np.float64)
    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "projected_dim": int(projected.shape[1]),
        "explained_variance": float(basis.explained_variance_ratio.sum()),
        "input_features": list(INPUT_FEATURE_NAMES),
        "source": source,
        "dataset_fingerprint": dataset_fingerprint,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "class_histogram": class_histogram.astype(int).tolist(),
        "note": (
            "Reference statistics define the no-shift condition. Drift at inference time is "
            "measured against this frozen window; it is never recomputed from live traffic."
        ),
    }
    return ReferenceStatistics(
        model_id=model_id,
        input_stats=np.asarray(input_stats, dtype=np.float64),
        embeddings=embeddings,
        projected=projected,
        entropy=predictive_entropy(probabilities),
        max_probability=max_probability(probabilities),
        class_histogram=class_histogram,
        pca_mean=basis.mean,
        pca_components=basis.components,
        explained_variance_ratio=basis.explained_variance_ratio,
        manifest=manifest,
    )


def save_reference(reference: ReferenceStatistics, path: Path) -> None:
    """Persist a reference artifact and its JSON sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        input_stats=reference.input_stats,
        embeddings=reference.embeddings.astype(np.float32),
        projected=reference.projected.astype(np.float32),
        entropy=reference.entropy.astype(np.float32),
        max_probability=reference.max_probability.astype(np.float32),
        class_histogram=reference.class_histogram,
        pca_mean=reference.pca_mean.astype(np.float32),
        pca_components=reference.pca_components.astype(np.float32),
        explained_variance_ratio=reference.explained_variance_ratio.astype(np.float32),
    )
    atomic_write_json(path.with_suffix(".json"), reference.manifest)
    logger.info(
        "reference artifact written",
        extra={"path": str(path), "samples": reference.manifest["samples"]},
    )


def load_reference(path: Path) -> ReferenceStatistics:
    """Load a reference artifact, failing explicitly when it is absent or stale."""
    if not path.exists():
        raise ReferenceNotFoundError(
            f"reference statistics not found: {path.name}",
            details={"expected": str(path), "remedy": "run training/build_reference.py"},
        )
    manifest = read_json(path.with_suffix(".json"), default={}) or {}
    try:
        with np.load(path) as data:
            return ReferenceStatistics(
                model_id=str(manifest.get("model_id", "unknown")),
                input_stats=np.asarray(data["input_stats"], dtype=np.float64),
                embeddings=np.asarray(data["embeddings"], dtype=np.float64),
                projected=np.asarray(data["projected"], dtype=np.float64),
                entropy=np.asarray(data["entropy"], dtype=np.float64),
                max_probability=np.asarray(data["max_probability"], dtype=np.float64),
                class_histogram=np.asarray(data["class_histogram"], dtype=np.float64),
                pca_mean=np.asarray(data["pca_mean"], dtype=np.float64),
                pca_components=np.asarray(data["pca_components"], dtype=np.float64),
                explained_variance_ratio=np.asarray(
                    data["explained_variance_ratio"], dtype=np.float64
                ),
                manifest=manifest,
            )
    except KeyError as exc:
        raise ReferenceNotFoundError(f"reference artifact is missing key {exc}") from exc
