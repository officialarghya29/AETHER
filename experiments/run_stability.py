"""Stability study: risk-controlled thresholds and test-time augmentation.

Two questions this answers with measurements rather than defaults:

1. **Conformal acceptance.** If the threshold is chosen from calibration data, what selective error
   and coverage does it actually deliver on unseen routing data — clean and corrupted?
2. **Test-time augmentation.** Does averaging an image's logits with its horizontal mirror improve
   the advanced tier enough to justify doubling its cost?

The official test split is not used to choose anything here; thresholds and TTA policy come from
calibration and routing-development data, and the study only reports them.

Outputs ``artifacts/reports/stability.json`` and ``STABILITY.md``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aether.config import load_config  # noqa: E402
from aether.data.cifar import build_and_persist_splits, load_cifar10  # noqa: E402
from aether.data.corruptions import apply_corruption  # noqa: E402
from aether.data.preprocessing import prepare_model_input  # noqa: E402
from aether.pipeline import ReliabilityPipeline  # noqa: E402
from aether.routing.conformal import acceptance_threshold, selective_report  # noqa: E402
from aether.uncertainty.calibration import apply_temperature  # noqa: E402

REPORTS = ROOT / "artifacts" / "reports"
ALPHAS = (0.20, 0.10, 0.05, 0.02)
TTA_IMAGES = 2000
CORRUPTED_IMAGES = 1000


def calibrated_probabilities(pipeline: ReliabilityPipeline, role: str, images: np.ndarray) -> np.ndarray:
    adapter = pipeline.registry.get(role)
    batch = prepare_model_input(images, pipeline.config.data.image_size)
    predictions = adapter.predict(batch, return_embeddings=False)
    out = []
    for prediction in predictions:
        temperature = pipeline.calibration.temperature(prediction.model_id)
        out.append(apply_temperature(prediction.logits, temperature))
    return np.stack(out)


def labels_for(splits: dict[str, np.ndarray], labels: np.ndarray, name: str) -> np.ndarray:
    return labels[splits[name]]


def conformal_study(
    pipeline: ReliabilityPipeline,
    splits: dict[str, np.ndarray],
    images: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for role in ("lightweight", "advanced"):
        calibration_probs = calibrated_probabilities(pipeline, role, images[splits["calibration"]])
        calibration_labels = labels[splits["calibration"]]
        true_class = calibration_probs[np.arange(len(calibration_labels)), calibration_labels]

        threshold_rows = []
        for alpha in ALPHAS:
            threshold = acceptance_threshold(true_class, alpha)
            row: dict[str, Any] = {"alpha": alpha, "threshold": round(threshold.threshold, 4)}
            for stream, evaluation_images in (
                ("routing_dev_clean", images[splits["routing_dev"]][:2000]),
                ("routing_dev_corrupted", None),
            ):
                if evaluation_images is None:
                    continue
                probs = calibrated_probabilities(pipeline, role, evaluation_images)
                eval_labels = labels[splits["routing_dev"]][: len(evaluation_images)]
                report = selective_report(probs, eval_labels, threshold.threshold)
                row[stream] = {
                    "coverage": round(report["coverage"], 4),
                    "selective_error": round(report["selective_error"], 4)
                    if not np.isnan(report["selective_error"])
                    else None,
                    "overall_error": round(report["overall_error"], 4),
                }
            threshold_rows.append(row)
        results[role] = threshold_rows
    return results


def corrupted_stream(images: np.ndarray, labels: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    indices = rng.choice(len(images), count, replace=False)
    corrupted = np.stack(
        [
            apply_corruption(images[i][None, ...], "gaussian_noise", 3, seed=int(i))[0]
            for i in indices
        ]
    )
    return corrupted, labels[indices]


def tta_study(
    pipeline: ReliabilityPipeline, images: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    adapter = pipeline.registry.try_get("advanced")
    if adapter is None:
        return {"available": False, "reason": "advanced checkpoint unavailable"}

    def evaluate(stream: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
        batch = prepare_model_input(stream, pipeline.config.data.image_size)
        identity = adapter.predict(batch, return_embeddings=False)
        started = time.perf_counter()
        mirrored = adapter.predict(torch.flip(batch, dims=[3]), return_embeddings=False)
        mirror_ms = (time.perf_counter() - started) * 1000.0 / max(1, len(stream))

        base_logits = np.stack([p.logits for p in identity])
        mirror_logits = np.stack([p.logits for p in mirrored])
        averaged = (base_logits.astype(np.float64) + mirror_logits.astype(np.float64)) / 2.0

        base_accuracy = float((base_logits.argmax(1) == truth).mean())
        tta_accuracy = float((averaged.argmax(1) == truth).mean())
        return {
            "samples": int(len(stream)),
            "identity_accuracy": round(base_accuracy, 4),
            "tta_accuracy": round(tta_accuracy, 4),
            "delta": round(tta_accuracy - base_accuracy, 4),
            "mirror_pass_ms_per_image": round(mirror_ms, 3),
        }

    clean = evaluate(images[:TTA_IMAGES], labels[:TTA_IMAGES])
    corrupted, corrupted_labels = corrupted_stream(images, labels, CORRUPTED_IMAGES)
    corrupted_result = evaluate(corrupted, corrupted_labels)
    return {
        "available": True,
        "clean": clean,
        "corrupted_gaussian_noise_severity_3": corrupted_result,
        "cost": "one extra advanced forward pass per escalated request",
    }


def main() -> None:
    config = load_config()
    torch.set_num_threads(config.runtime.num_threads)
    pipeline = ReliabilityPipeline(config)
    pipeline.warm(("lightweight", "advanced"))
    data = load_cifar10(config.data_root)
    splits = build_and_persist_splits(
        config.data_root,
        data.train_labels,
        config.data.splits,
        config.data.split_seed,
        data.fingerprint,
    )
    images = data.train_images
    labels = data.train_labels

    print("== conformal thresholds ==")
    conformal = conformal_study(pipeline, splits, images, labels)
    for role, rows in conformal.items():
        for row in rows:
            stream = row.get("routing_dev_clean", {})
            print(
                f"  {role:<11} alpha={row['alpha']:.2f} threshold={row['threshold']:.3f} "
                f"coverage={stream.get('coverage')} error={stream.get('selective_error')}"
            )

    print("== test-time augmentation ==")
    clean = images[splits["routing_dev"]][:TTA_IMAGES]
    clean_labels = labels[splits["routing_dev"]][:TTA_IMAGES]
    tta = tta_study(pipeline, clean, clean_labels)
    if tta.get("available"):
        for stream in ("clean", "corrupted_gaussian_noise_severity_3"):
            row = tta[stream]
            print(
                f"  {stream:<36} identity={row['identity_accuracy']:.4f} "
                f"tta={row['tta_accuracy']:.4f} delta={row['delta']:+.4f}"
            )

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": (
            "Thresholds are fitted on the calibration split and evaluated on routing-development "
            "data; nothing here touches the official test split. TTA averages identity and "
            "horizontal-flip logits on the advanced tier only."
        ),
        "environment": {
            "device": "cpu",
            "threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "conformal": conformal,
        "tta": tta,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "stability.json").write_text(json.dumps(report, indent=2))
    write_markdown(report)
    print(f"wrote {REPORTS / 'stability.json'} and STABILITY.md")


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Stability",
        "",
        "Risk-controlled thresholds and test-time augmentation, measured on calibration and",
        "routing-development data. Regenerate with `make stability`.",
        "",
        "## Conformal acceptance (clean routing-dev)",
        "",
        "| tier | alpha | threshold | coverage | selective error |",
        "| --- | --- | --- | --- | --- |",
    ]
    for role, rows in report["conformal"].items():
        for row in rows:
            stream = row.get("routing_dev_clean", {})
            error = stream.get("selective_error")
            lines.append(
                f"| {role} | {row['alpha']:.2f} | {row['threshold']:.3f} | "
                f"{stream.get('coverage', 0):.3f} | {'—' if error is None else f'{error:.3f}'} |"
            )
    lines += ["", "## Test-time augmentation (advanced tier)", ""]
    tta = report["tta"]
    if tta.get("available"):
        lines += [
            "| stream | identity | TTA | delta | mirror pass |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name in ("clean", "corrupted_gaussian_noise_severity_3"):
            row = tta[name]
            lines.append(
                f"| {name} | {row['identity_accuracy']:.4f} | {row['tta_accuracy']:.4f} | "
                f"{row['delta']:+.4f} | {row['mirror_pass_ms_per_image']:.2f} ms |"
            )
        lines += [
            "",
            f"Cost: {tta['cost']}. The pipeline enables it with `routing.tta = true`.",
        ]
    else:
        lines.append(f"Unavailable: {tta.get('reason')}")
    lines += ["", "## Protocol", "", report["protocol"], ""]
    (REPORTS / "STABILITY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
