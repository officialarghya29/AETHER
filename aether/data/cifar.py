"""CIFAR-10 loading, provenance and stratified split construction.

The dataset is downloaded once into ``data/raw`` and never committed. Every split decision is
recorded in ``data/processed/splits.json`` together with a SHA-256 fingerprint of the pixel
arrays, so any reported number can be traced back to an exact byte-level input set.

Split discipline (the core scientific guarantee of the project):

=================  ========  ==================================================
split              size      permitted use
=================  ========  ==================================================
train              32000     gradient updates only
validation         4000      model selection / early stopping
calibration        4000      temperature scaling and risk-model fitting
routing_dev        5000      routing thresholds, drift thresholds, ablations
holdout            5000      reference windows and final internal checks
official test      10000     reported results only, never tuning
=================  ========  ==================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aether.config import SplitSizes
from aether.utils.io import atomic_write_json, sha256_bytes
from aether.utils.logging import get_logger

logger = get_logger("aether.data.cifar")

CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Channel statistics of the CIFAR-10 training partition (Hendrycks & Gimpel scaling).
CIFAR10_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "calibration", "routing_dev", "holdout")


@dataclass(frozen=True)
class CifarData:
    """In-memory CIFAR-10 arrays (uint8 NHWC) with provenance."""

    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    fingerprint: dict[str, Any]
    source: str

    def summary(self) -> dict[str, Any]:
        return {
            "train_images": int(self.train_images.shape[0]),
            "test_images": int(self.test_images.shape[0]),
            "shape": list(self.train_images.shape[1:]),
            "fingerprint": self.fingerprint,
            "source": self.source,
        }


def load_cifar10(root: Path, download: bool = True) -> CifarData:
    """Load CIFAR-10 from the local artefact, falling back to torchvision's mirror.

    ``data/raw/cifar10.npz`` is produced by ``scripts/download_cifar.py`` and carries the exact
    provenance (source URL, array digests). Preferences order: local artefact, then the
    official release through torchvision. Raises ``FileNotFoundError`` when neither is
    available and downloading is disabled: the pipeline never fabricates data.
    """
    local_npz = root / "raw" / "cifar10.npz"
    if local_npz.exists():
        return _load_npz(local_npz, root / "raw" / "cifar10.json")

    from torchvision.datasets import CIFAR10

    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        train_ds = CIFAR10(root=str(raw_dir), train=True, download=download)
        test_ds = CIFAR10(root=str(raw_dir), train=False, download=download)
    except RuntimeError as exc:  # torchvision raises RuntimeError for download failures
        raise FileNotFoundError(
            f"CIFAR-10 is not available under {raw_dir} and could not be downloaded: {exc}"
        ) from exc

    train_images = np.asarray(train_ds.data, dtype=np.uint8)
    train_labels = np.asarray(train_ds.targets, dtype=np.int64)
    test_images = np.asarray(test_ds.data, dtype=np.uint8)
    test_labels = np.asarray(test_ds.targets, dtype=np.int64)

    fingerprint = {
        "train_sha256": sha256_bytes(train_images.tobytes()),
        "test_sha256": sha256_bytes(test_images.tobytes()),
        "train_sha256_labels": sha256_bytes(train_labels.tobytes()),
        "test_sha256_labels": sha256_bytes(test_labels.tobytes()),
        "dtype": str(train_images.dtype),
        "shape": list(train_images.shape),
    }
    logger.info(
        "loaded CIFAR-10",
        extra={
            "train_images": train_images.shape[0],
            "test_images": test_images.shape[0],
            "train_fingerprint": fingerprint["train_sha256"][:12],
        },
    )
    return CifarData(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        fingerprint=fingerprint,
        source="https://www.cs.toronto.edu/~kriz/cifar.html (via torchvision CIFAR10)",
    )


def _load_npz(npz_path: Path, manifest_path: Path) -> CifarData:
    """Load the local CIFAR-10 artefact and verify it against its manifest."""
    import json

    with np.load(npz_path) as payload:
        train_images = np.asarray(payload["train_images"], dtype=np.uint8)
        train_labels = np.asarray(payload["train_labels"], dtype=np.int64)
        test_images = np.asarray(payload["test_images"], dtype=np.uint8)
        test_labels = np.asarray(payload["test_labels"], dtype=np.int64)
    if train_images.shape[0] != 50000 or test_images.shape[0] != 10000:
        raise ValueError(
            f"unexpected CIFAR-10 split sizes: {train_images.shape[0]}/{test_images.shape[0]}"
        )
    if train_images.shape[1:] != (32, 32, 3):
        raise ValueError(f"unexpected CIFAR-10 image shape: {train_images.shape[1:]}")

    fingerprint = {
        "train_sha256": sha256_bytes(train_images.tobytes()),
        "test_sha256": sha256_bytes(test_images.tobytes()),
        "train_sha256_labels": sha256_bytes(train_labels.tobytes()),
        "test_sha256_labels": sha256_bytes(test_labels.tobytes()),
        "dtype": str(train_images.dtype),
        "shape": list(train_images.shape),
    }
    source = "local cifar10.npz"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = str(manifest.get("source", source))
        recorded = manifest.get("train", {}).get("images_sha256")
        if recorded and recorded != fingerprint["train_sha256"]:
            raise ValueError(
                "CIFAR-10 artefact does not match its manifest digest; re-run "
                "scripts/download_cifar.py instead of trusting a modified file"
            )
    logger.info(
        "loaded CIFAR-10 artefact",
        extra={"path": npz_path.name, "train_fingerprint": fingerprint["train_sha256"][:12]},
    )
    return CifarData(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        fingerprint=fingerprint,
        source=source,
    )


def build_stratified_splits(
    labels: np.ndarray,
    sizes: SplitSizes,
    seed: int,
    num_classes: int = 10,
) -> dict[str, np.ndarray]:
    """Allocate class-balanced index arrays for each non-test split.

    Allocation is deterministic for a given ``(labels, sizes, seed)``: per class, indices are
    permuted with a seeded RNG and sliced in fixed proportion, so reruns are byte-identical.
    """
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    total = sizes.total
    if total > labels.shape[0]:
        raise ValueError(f"requested {total} indices from a pool of {labels.shape[0]}")

    rng = np.random.default_rng(seed)
    per_class_pool: list[np.ndarray] = []
    for class_index in range(num_classes):
        class_indices = np.flatnonzero(labels == class_index)
        if class_indices.size == 0:
            raise ValueError(f"label {class_index} has no samples in the split pool")
        per_class_pool.append(class_indices[rng.permutation(class_indices.size)])

    # Proportional per-class quotas, with a largest-remainder correction to hit exact totals.
    quotas = np.zeros((num_classes, len(SPLIT_NAMES)), dtype=np.int64)
    requested = np.array([getattr(sizes, name) for name in SPLIT_NAMES], dtype=np.int64)
    class_sizes = np.array(
        [per_class_pool[c].size for c in range(num_classes)], dtype=np.float64
    )
    for split_index, split_total in enumerate(requested):
        if split_total == 0:
            continue
        ideal = class_sizes / class_sizes.sum() * split_total
        floors = np.floor(ideal).astype(np.int64)
        remainder = split_total - int(floors.sum())
        order = np.argsort(-(ideal - floors), kind="stable")
        floors[order[:remainder]] += 1
        quotas[:, split_index] = floors
    if (quotas.sum() == 0).any():
        raise ValueError("empty split allocation produced by the requested sizes")

    splits: dict[str, np.ndarray] = {}
    for split_index, name in enumerate(SPLIT_NAMES):
        selected: list[np.ndarray] = []
        for class_index in range(num_classes):
            take = int(quotas[class_index, split_index])
            if take == 0:
                continue
            if take > per_class_pool[class_index].size:
                raise ValueError(f"split '{name}' requests more class samples than available")
            selected.append(per_class_pool[class_index][:take])
            per_class_pool[class_index] = per_class_pool[class_index][take:]
        splits[name] = np.sort(np.concatenate(selected))
    return splits


def split_manifest(
    splits: dict[str, np.ndarray],
    labels: np.ndarray,
    seed: int,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    """Describe a split assignment: sizes, class balance and provenance."""
    manifest: dict[str, Any] = {
        "seed": seed,
        "fingerprint": fingerprint,
        "classes": list(CIFAR10_CLASSES),
        "splits": {},
    }
    for name in SPLIT_NAMES:
        indices = splits[name]
        counts = np.bincount(labels[indices], minlength=len(CIFAR10_CLASSES))
        manifest["splits"][name] = {
            "size": int(indices.size),
            "index_sha256": sha256_bytes(indices.astype(np.int64).tobytes()),
            "class_counts": {CIFAR10_CLASSES[i]: int(c) for i, c in enumerate(counts)},
        }
    return manifest


def build_and_persist_splits(
    root: Path,
    labels: np.ndarray,
    sizes: SplitSizes,
    seed: int,
    fingerprint: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Build splits and persist the manifest; reuse an identical existing assignment."""
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    splits_path = processed / "splits.npz"
    manifest_path = processed / "splits.json"

    splits = build_stratified_splits(labels, sizes, seed)
    if splits_path.exists() and manifest_path.exists():
        with np.load(splits_path) as cached:
            cached_seed = int(cached["seed"]) if "seed" in cached else -1
            # The archive must stay open while its members are compared.
            same = cached_seed == seed and all(
                name in cached and np.array_equal(cached[name], splits[name])
                for name in SPLIT_NAMES
            )
        if same:
            logger.info("reusing existing split assignment", extra={"seed": seed})
            return splits
    # numpy's stubs type ``**kwds`` loosely, so the saver is bound through an untyped alias.
    savez: Any = np.savez_compressed
    savez(splits_path, seed=np.int64(seed), **splits)
    atomic_write_json(manifest_path, split_manifest(splits, labels, seed, fingerprint))
    logger.info(
        "wrote split manifest",
        extra={"path": str(manifest_path), "sizes": {k: int(v.size) for k, v in splits.items()}},
    )
    return splits
