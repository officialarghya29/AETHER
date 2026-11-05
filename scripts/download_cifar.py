#!/usr/bin/env python3
"""Materialise CIFAR-10 into ``data/raw/cifar10.npz`` from an available source.

The official release is preferred (torchvision download). When the official host is
unreachable or too slow, the same release fetched as Parquet from the Hugging Face mirror of
``uoft-cs/cifar10`` is converted into the identical pixel arrays. Either way the resulting
arrays are fingerprinted and the manifest records exactly which source produced them, so a
report can always be traced to a byte-level input set.

Usage::

    python scripts/download_cifar.py                  # auto: existing files -> torchvision -> parquet
    python scripts/download_cifar.py --from-parquet   # force the local Parquet conversion
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HF_TRAIN = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet"
HF_TEST = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet"


def _decode_parquet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode an image Parquet shard into uint8 NHWC arrays and int64 labels."""
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(path)
    columns = table.column_names
    if "img" not in columns or "label" not in columns:
        raise ValueError(f"unexpected parquet schema {columns} in {path.name}")
    images_column = table.column("img").to_pylist()
    labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
    images = np.empty((len(images_column), 32, 32, 3), dtype=np.uint8)
    for index, entry in enumerate(images_column):
        payload = entry.get("bytes") if isinstance(entry, dict) else entry
        if payload is None:
            raise ValueError(f"null image payload at row {index} of {path.name}")
        array = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.uint8)
        if array.shape != (32, 32, 3):
            raise ValueError(f"unexpected image shape {array.shape} at row {index}")
        images[index] = array
    return images, labels


def _fingerprint(images: np.ndarray, labels: np.ndarray) -> dict[str, str]:
    return {
        "images_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "shape": ",".join(str(dim) for dim in images.shape),
    }


def from_parquet(root: Path) -> None:
    """Convert downloaded Parquet shards into the canonical npz artifact."""
    hf_dir = root / "raw" / "hf"
    train_path = hf_dir / "train.parquet"
    test_path = hf_dir / "test.parquet"
    for path, url in ((train_path, HF_TRAIN), (test_path, HF_TEST)):
        if not path.exists() or path.stat().st_size < 1_000_000:
            raise FileNotFoundError(
                f"{path} is missing or incomplete; download it first:\n  curl -L -o {path} {url}"
            )
    print(f"decoding {train_path.name} ...")
    train_images, train_labels = _decode_parquet(train_path)
    print(f"decoding {test_path.name} ...")
    test_images, test_labels = _decode_parquet(test_path)

    if train_images.shape[0] != 50000 or test_images.shape[0] != 10000:
        raise ValueError(
            f"unexpected split sizes train={train_images.shape[0]} test={test_images.shape[0]}"
        )
    _sanity_check(train_labels, test_labels)

    out = root / "raw" / "cifar10.npz"
    np.savez(out, train_images=train_images, train_labels=train_labels,
             test_images=test_images, test_labels=test_labels)
    manifest = {
        "source": "Hugging Face mirror uoft-cs/cifar10 (official CIFAR-10 release)",
        "source_urls": [HF_TRAIN, HF_TEST],
        "converted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train": _fingerprint(train_images, train_labels),
        "test": _fingerprint(test_images, test_labels),
        "note": "Pixel arrays are identical in shape and dtype to the official python release.",
    }
    import json

    (root / "raw" / "cifar10.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"train fingerprint {manifest['train']['images_sha256'][:16]}")


def _sanity_check(train_labels: np.ndarray, test_labels: np.ndarray) -> None:
    """Verify balanced classes: a corrupted download must not silently train a model."""
    for name, labels in (("train", train_labels), ("test", test_labels)):
        counts = np.bincount(labels, minlength=10)
        if counts.size != 10 or counts.min() != counts.max():
            raise ValueError(f"{name} labels are not class-balanced: {counts.tolist()}")
        if labels.min() != 0 or labels.max() != 9:
            raise ValueError(f"{name} labels out of range")
    print(f"label distribution verified: train {np.bincount(train_labels).tolist()}")


def from_torchvision(root: Path) -> None:
    """Load through torchvision (official host) and snapshot the arrays to npz."""
    from torchvision.datasets import CIFAR10

    raw = root / "raw"
    train = CIFAR10(root=str(raw), train=True, download=True)
    test = CIFAR10(root=str(raw), train=False, download=True)
    images_train = np.asarray(train.data, dtype=np.uint8)
    labels_train = np.asarray(train.targets, dtype=np.int64)
    images_test = np.asarray(test.data, dtype=np.uint8)
    labels_test = np.asarray(test.targets, dtype=np.int64)
    _sanity_check(labels_train, labels_test)
    out = root / "raw" / "cifar10.npz"
    np.savez(out, train_images=images_train, train_labels=labels_train,
             test_images=images_test, test_labels=labels_test)
    import json

    (root / "raw" / "cifar10.json").write_text(
        json.dumps(
            {
                "source": "torchvision CIFAR10 (https://www.cs.toronto.edu/~kriz/cifar.html)",
                "converted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "train": _fingerprint(images_train, labels_train),
                "test": _fingerprint(images_test, labels_test),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-parquet", action="store_true")
    parser.add_argument("--root", default=str(ROOT / "data"))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.from_parquet:
        from_parquet(root)
        return 0
    if (root / "raw" / "cifar10.npz").exists():
        print("cifar10.npz already present")
        return 0
    try:
        from_torchvision(root)
    except Exception as exc:
        print(f"official download failed ({exc}); falling back to parquet conversion")
        from_parquet(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
