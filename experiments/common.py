"""Shared helpers for experiment scripts.

Model inference for experiments goes through the same adapter and preprocessing used by the
service, so a benchmark number describes the deployed code path.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aether.config import AetherConfig  # noqa: E402
from aether.data.preprocessing import prepare_model_input  # noqa: E402


def collect_outputs(
    adapter,
    images: np.ndarray,
    *,
    image_size: int,
    batch_size: int = 256,
    collect_embeddings: bool = False,
) -> dict:
    """Run an adapter over uint8 NHWC images and return arrays plus timing."""
    logits_chunks = []
    probability_chunks = []
    embedding_chunks = []
    latencies = []
    for start in range(0, images.shape[0], batch_size):
        chunk = images[start : start + batch_size]
        batch = prepare_model_input(chunk, image_size)
        started = time.perf_counter()
        predictions = adapter.predict(batch, return_embeddings=collect_embeddings)
        latencies.append((time.perf_counter() - started) * 1000.0)
        logits_chunks.append(np.stack([p.logits for p in predictions]))
        probability_chunks.append(np.stack([p.probabilities for p in predictions]))
        if collect_embeddings:
            embedding_chunks.append(np.stack([p.embedding for p in predictions]))
    result = {
        "logits": np.concatenate(logits_chunks).astype(np.float32),
        "probabilities": np.concatenate(probability_chunks).astype(np.float64),
        "latency_ms_per_batch": latencies,
        "samples": int(images.shape[0]),
    }
    if collect_embeddings:
        result["embeddings"] = np.concatenate(embedding_chunks).astype(np.float32)
    return result


def image_quality_matrix(images: np.ndarray) -> np.ndarray:
    """Per-image input statistics in the same order the drift monitor expects."""
    from aether.diagnosis.image_quality import analyze_image

    rows = []
    for image in images:
        quality = analyze_image(image)
        channel_means = image.reshape(-1, 3).mean(axis=0) / 255.0
        rows.append(
            [
                quality.brightness,
                quality.contrast,
                quality.sharpness,
                float(channel_means[0]),
                float(channel_means[1]),
                float(channel_means[2]),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def write_report(path: Path, payload: dict) -> None:
    from aether.utils.io import atomic_write_json

    atomic_write_json(path, payload)
    print(f"wrote {path}")


def environment_info(config: AetherConfig) -> dict:
    import platform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "cpu_count": torch.get_num_threads(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "image_size": config.data.image_size,
        "seed": config.runtime.seed,
    }
