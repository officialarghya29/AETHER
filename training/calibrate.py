#!/usr/bin/env python3
"""Fit temperature scaling and the learned risk models.

Split discipline enforced here:

* temperatures are fitted on the **calibration** split only;
* risk models are also fitted on the **calibration** split, augmented with corrupted calibration
  samples so the drift features have observed variation;
* the routing-development split is untouched here and is used only to evaluate policies;
* the official test split is never loaded.

Outputs
-------
``artifacts/calibration/temperatures.json``   per-model temperature + before/after metrics
``artifacts/calibration/learned_policy.json`` fitted risk models (first call, pre-call advanced,
                                              post-escalation advanced) + expected benefit
``artifacts/reports/calibration.json``        full audit report
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aether.adapters.base import ModelAdapter  # noqa: E402
from aether.config import load_config  # noqa: E402
from aether.data.cifar import build_and_persist_splits, load_cifar10  # noqa: E402
from aether.data.corruptions import apply_corruption  # noqa: E402
from aether.drift.monitor import DriftMonitor, DriftObservation  # noqa: E402
from aether.drift.reference import load_reference  # noqa: E402
from aether.models.registry import ModelRegistry  # noqa: E402
from aether.routing.learned_policy import build_policy_artifact, save_policy_artifact  # noqa: E402
from aether.routing.risk_model import RiskModel  # noqa: E402
from aether.runtime import configure_runtime, resolve_device  # noqa: E402
from aether.uncertainty.calibration import CalibrationStore, fit_temperature  # noqa: E402
from aether.uncertainty.entropy import (  # noqa: E402
    max_probability,
    prediction_margin,
    predictive_entropy,
)
from aether.utils.io import seed_everything  # noqa: E402
from aether.utils.logging import configure_logging, get_logger  # noqa: E402
from experiments.common import (  # noqa: E402
    collect_outputs,
    environment_info,
    image_quality_matrix,
    write_report,
)

logger = get_logger("aether.calibrate")

DEV_CORRUPTION_FAMILIES = (
    "gaussian_noise",
    "gaussian_blur",
    "brightness",
    "contrast",
    "jpeg_compression",
    "pixelate",
)
DEV_CORRUPTION_SEVERITY = 4
DEV_CORRUPTION_SAMPLES = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def _evidence(probabilities: np.ndarray, drift_score: float, severity: str, disagreement: float = 0.0) -> dict[str, np.ndarray]:
    entropy = predictive_entropy(probabilities)
    msp = max_probability(probabilities)
    margin = prediction_margin(probabilities)
    return {
        "entropy": entropy,
        "max_probability": msp,
        "margin": margin,
        "drift_score": np.full(entropy.shape, drift_score),
        "shifted_flag": np.full(entropy.shape, 1.0 if severity in ("moderate", "severe") else 0.0),
        "disagreement": np.full(entropy.shape, disagreement),
    }


def _concat(blocks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}


def build_dev_evidence(
    config,
    adapter: ModelAdapter,
    monitor: DriftMonitor | None,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    tag: str,
) -> dict:
    """Collect evidence for one split, including per-window drift scores."""
    outputs = collect_outputs(
        adapter, images, image_size=config.data.image_size, collect_embeddings=True
    )
    quality = image_quality_matrix(images)
    window = config.drift.evaluation_window_size
    entropy = predictive_entropy(outputs["probabilities"])
    confidences = max_probability(outputs["probabilities"])
    predicted = outputs["probabilities"].argmax(axis=1)
    drift_scores = np.zeros(images.shape[0])
    severities: list[str] = ["none"] * images.shape[0]
    for start in range(0, images.shape[0], window) if monitor is not None else []:
        stop = min(images.shape[0], start + window)
        observations = [
            DriftObservation(
                input_stats=quality[index],
                embedding=outputs["embeddings"][index],
                entropy=float(entropy[index]),
                max_probability=float(confidences[index]),
                predicted_class=int(predicted[index]),
            )
            for index in range(start, stop)
        ]
        report = monitor.evaluate(observations)
        drift_scores[start:stop] = report.score
        severities.extend([report.severity] * (stop - start))
    logger.info(
        "dev evidence collected", extra={"split": tag, "samples": int(images.shape[0])}
    )
    return {
        "probabilities": outputs["probabilities"],
        "logits": outputs["logits"],
        "labels": labels,
        "entropy": entropy,
        "confidence": confidences,
        "drift_score": drift_scores,
        "drift_severity": severities,
        "correct": (predicted == labels).astype(np.int64),
        "latency_ms": float(np.mean(outputs["latency_ms_per_batch"]) / min(config.data.eval_batch_size, images.shape[0])),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config.monitoring.log_level, config.monitoring.json_logs)
    configure_runtime(config.runtime)
    seed_everything(config.runtime.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    registry = ModelRegistry(config, device)
    light = registry.get("lightweight")
    advanced = registry.get("advanced")

    data = load_cifar10(config.data_root)
    splits = build_and_persist_splits(
        config.data_root, data.train_labels, config.data.splits, config.data.split_seed, data.fingerprint
    )
    calibration_images = data.train_images[splits["calibration"]]
    calibration_labels = data.train_labels[splits["calibration"]]

    # ---- temperature scaling on the calibration split -------------------------------
    store = CalibrationStore(config.resolve(config.uncertainty.temperature_artifact))
    temperature_records = {}
    for adapter in (light, advanced):
        outputs = collect_outputs(adapter, calibration_images, image_size=config.data.image_size)
        calibration = fit_temperature(
            outputs["logits"],
            calibration_labels,
            model_id=adapter.metadata.model_id,
            split="calibration",
            dataset_fingerprint=data.fingerprint["train_sha256"][:16],
            n_bins=config.uncertainty.calibration_bins,
        )
        store.save(calibration)
        temperature_records[adapter.metadata.role] = calibration.to_dict()

    # ---- risk models on the calibration split (routing_dev stays evaluation-only) ----
    reference = load_reference(config.resolve(config.drift.reference_artifact))
    monitor = DriftMonitor(reference, config.drift, num_classes=light.num_classes, seed=config.runtime.seed)
    fit_images = calibration_images
    fit_labels = calibration_labels
    clean = build_dev_evidence(config, light, monitor, fit_images, fit_labels, tag="cal_clean")

    rng = np.random.default_rng(config.runtime.seed)
    corrupted_blocks = []
    corruption_manifest = {}
    for family in DEV_CORRUPTION_FAMILIES:
        take = rng.choice(fit_images.shape[0], DEV_CORRUPTION_SAMPLES, replace=False)
        take.sort()
        corrupted_images = apply_corruption(
            fit_images[take], family, DEV_CORRUPTION_SEVERITY, seed=config.runtime.seed
        )
        block = build_dev_evidence(
            config, light, monitor, corrupted_images, fit_labels[take], tag=f"cal_{family}"
        )
        corrupted_blocks.append(block)
        corruption_manifest[family] = DEV_CORRUPTION_SEVERITY

    light_features = _concat(
        [_evidence(clean["probabilities"], 0.0, "none")]
        + [
            _evidence(block["probabilities"], float(block["drift_score"].mean()), "severe")
            for block in corrupted_blocks
        ]
    )
    # The model estimates P(error), so the label must be 1 for incorrect predictions. Fitting on
    # correctness inverted every coefficient and made the router prefer uncertain samples.
    light_labels = np.concatenate(
        [1 - clean["correct"]] + [1 - block["correct"] for block in corrupted_blocks]
    )
    # disagreement is unavailable before escalation, so it is excluded from this model
    light_features["disagreement"] = np.zeros_like(light_labels, dtype=np.float64)
    light_model = RiskModel.fit(
        light_features, light_labels, model_id=f"risk:{light.metadata.model_id}", split="calibration"
    )

    # Drift is measured in the served (lightweight) embedding space only: a reference artifact is
    # model-specific, so the advanced model reuses those scores instead of being compared to a
    # basis fitted on different features.
    advanced_blocks = [
        build_dev_evidence(config, advanced, None, fit_images, fit_labels, tag="cal_clean_advanced")
    ]
    for family in DEV_CORRUPTION_FAMILIES:
        take = rng.choice(fit_images.shape[0], DEV_CORRUPTION_SAMPLES, replace=False)
        take.sort()
        corrupted_images = apply_corruption(
            fit_images[take], family, DEV_CORRUPTION_SEVERITY, seed=config.runtime.seed
        )
        advanced_blocks.append(
            build_dev_evidence(
                config, advanced, None, corrupted_images, fit_labels[take], tag=f"cal_{family}_advanced"
            )
        )
    advanced_features = []
    advanced_labels = []
    for clean_block, adv_block in zip([clean] + corrupted_blocks, advanced_blocks, strict=True):
        disagreement = 0.5 * np.abs(clean_block["probabilities"] - adv_block["probabilities"]).sum(axis=1)
        advanced_features.append(
            _evidence(adv_block["probabilities"], float(clean_block["drift_score"].mean()),
                      "none" if clean_block is clean else "severe", 0.0)
        )
        advanced_features[-1]["disagreement"] = disagreement
        advanced_labels.append(1 - adv_block["correct"])
    advanced_model = RiskModel.fit(
        _concat(advanced_features),
        np.concatenate(advanced_labels),
        model_id=f"risk:{advanced.metadata.model_id}",
        split="calibration",
    )
    # Pre-call head: the same lightweight evidence, labelled with whether the *advanced* model
    # would be wrong. Without it, escalation could only be priced with information that does not
    # exist until after the second call is paid for.
    precall_model = RiskModel.fit(
        light_features,
        np.concatenate(advanced_labels),
        model_id=f"risk:{advanced.metadata.model_id}:precall",
        split="calibration",
    )

    light_risk = light_model.predict_batch(
        [{key: float(values[i]) for key, values in light_features.items()} for i in range(len(light_labels))]
    )
    precall_risk = precall_model.predict_batch(
        [{key: float(values[i]) for key, values in light_features.items()} for i in range(len(light_labels))]
    )
    clean_rows = len(clean["correct"])
    expected_benefit = float(np.mean(light_risk[:clean_rows] - precall_risk[:clean_rows]))

    artifact = build_policy_artifact(
        lightweight=light_model,
        advanced=advanced_model,
        advanced_precall=precall_model,
        samples=int(light_labels.size),
        split="calibration",
        expected_benefit=expected_benefit,
        notes=[
            "Risk models are fitted on offline logged outcomes from the calibration split.",
            "Corrupted calibration samples provide drift-signal variation; they are not test data.",
            "Disagreement is excluded from the first-call and pre-call models because it is unknown "
            "before a second model call.",
            "The pre-call advanced head is fitted on lightweight evidence only, so escalation is "
            "priced without consulting the second model's output.",
        ],
    )
    save_policy_artifact(artifact, config.resolve(config.routing.learned_policy.artifact))

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment_info(config),
        "splits": {"calibration": int(calibration_labels.size)},
        "temperatures": temperature_records,
        "risk_models": {
            "lightweight": light_model.describe(),
            "advanced": advanced_model.describe(),
            "advanced_precall": precall_model.describe(),
            "expected_risk_reduction": expected_benefit,
        },
        "calibration_corruption_augmentation": {
            "families": corruption_manifest,
            "severity": DEV_CORRUPTION_SEVERITY,
            "samples_per_family": DEV_CORRUPTION_SAMPLES,
        },
        "calibration_clean_drift": {
            "mean_drift_score": float(clean["drift_score"].mean()),
            "max_drift_score": float(clean["drift_score"].max()),
            "severities": {level: clean["drift_severity"].count(level) for level in set(clean["drift_severity"])},
        },
        "measured_latency_ms_per_sample": {
            "lightweight": clean["latency_ms"],
            "advanced": advanced_blocks[0]["latency_ms"],
        },
    }
    write_report(config.resolve("artifacts/reports/calibration.json"), report)
    print(f"temperature lightweight: {temperature_records['lightweight']['temperature']:.4f}")
    print(f"temperature advanced:    {temperature_records['advanced']['temperature']:.4f}")
    print(f"expected risk reduction: {expected_benefit:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
