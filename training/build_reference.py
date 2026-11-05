#!/usr/bin/env python3
"""Build the frozen reference distribution used for drift detection.

Source window: the **holdout** split, which is disjoint from training, calibration, routing
development and the official test set. The reference therefore cannot encode information from
any split used to tune or report results.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aether.config import load_config  # noqa: E402
from aether.data.cifar import build_and_persist_splits, load_cifar10  # noqa: E402
from aether.drift.reference import build_reference, save_reference  # noqa: E402
from aether.models.registry import ModelRegistry  # noqa: E402
from aether.runtime import configure_runtime, resolve_device  # noqa: E402
from aether.utils.io import seed_everything  # noqa: E402
from aether.utils.logging import configure_logging, get_logger  # noqa: E402
from experiments.common import (  # noqa: E402
    collect_outputs,
    environment_info,
    image_quality_matrix,
    write_report,
)

logger = get_logger("aether.reference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--window", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config.monitoring.log_level, config.monitoring.json_logs)
    configure_runtime(config.runtime)
    seed_everything(config.runtime.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)

    data = load_cifar10(config.data_root)
    splits = build_and_persist_splits(
        config.data_root, data.train_labels, config.data.splits, config.data.split_seed, data.fingerprint
    )
    window = args.window or config.data.reference_window_size
    holdout_indices = splits["holdout"][:window]
    images = data.train_images[holdout_indices]

    registry = ModelRegistry(config, device)
    adapter = registry.get("lightweight")
    outputs = collect_outputs(adapter, images, image_size=config.data.image_size, collect_embeddings=True)
    quality = image_quality_matrix(images)

    reference = build_reference(
        model_id=adapter.metadata.model_id,
        input_stats=quality,
        embeddings=outputs["embeddings"],
        probabilities=outputs["probabilities"],
        predicted_labels=outputs["probabilities"].argmax(axis=1),
        num_classes=config.models.lightweight.num_classes,
        n_components=config.drift.embedding_components,
        source=f"cifar10 holdout split, first {window} samples",
        dataset_fingerprint=data.fingerprint["train_sha256"][:16],
        seed=config.runtime.seed,
    )
    path = config.resolve(config.drift.reference_artifact)
    save_reference(reference, path)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment_info(config),
        "artifact": str(path),
        "manifest": reference.manifest,
        "input_statistics": {
            name: {
                "mean": float(reference.input_stats[:, index].mean()),
                "std": float(reference.input_stats[:, index].std()),
                "p05": float(np.quantile(reference.input_stats[:, index], 0.05)),
                "p95": float(np.quantile(reference.input_stats[:, index], 0.95)),
            }
            for index, name in enumerate(
                ("brightness", "contrast", "sharpness", "mean_r", "mean_g", "mean_b")
            )
        },
        "entropy": {
            "mean": float(reference.entropy.mean()),
            "p95": float(np.quantile(reference.entropy, 0.95)),
        },
        "confidence": {"mean": float(reference.max_probability.mean())},
        "class_histogram": reference.class_histogram.astype(int).tolist(),
        "numpy_version": np.__version__,
    }
    write_report(config.resolve("artifacts/reports/reference.json"), report)
    print(f"reference written: {path} ({reference.manifest['samples']} samples, "
          f"EV={reference.manifest['explained_variance']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
