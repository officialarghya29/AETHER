#!/usr/bin/env python3
"""Reproducible reliability study.

Produces the numbers reported in the README:

1. clean test-set classification and calibration for both models
2. corruption robustness across 10 families x 5 severities on a fixed stratified subsample
3. drift detection behaviour: false-positive rate on clean windows, detection rate on corrupted
4. routing policy comparison including ablations

Every number comes from the frozen official test split (reported only) or the routing-
development split (policy development). Nothing is tuned here.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aether.config import load_config  # noqa: E402
from aether.data.cifar import build_and_persist_splits, load_cifar10  # noqa: E402
from aether.data.corruptions import CORRUPTION_FAMILIES, SEVERITIES, apply_corruption  # noqa: E402
from aether.drift.monitor import DriftMonitor, DriftObservation  # noqa: E402
from aether.drift.reference import load_reference  # noqa: E402
from aether.evaluation.classification import classification_report, selective_report  # noqa: E402
from aether.evaluation.robustness import robustness_suite, summarise_robustness  # noqa: E402
from aether.evaluation.routing import PolicyRecord, simulate_policy  # noqa: E402
from aether.models.registry import ModelRegistry  # noqa: E402
from aether.routing.learned_policy import load_precall_risk_model, load_risk_models  # noqa: E402
from aether.runtime import configure_runtime, resolve_device  # noqa: E402
from aether.uncertainty.calibration import CalibrationStore  # noqa: E402
from aether.uncertainty.entropy import (  # noqa: E402
    max_probability,
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

logger = get_logger("aether.benchmarks")
ROBUSTNESS_SAMPLES = 2500
DRIFT_FAMILIES = ("gaussian_noise", "gaussian_blur", "brightness", "contrast", "pixelate", "jpeg_compression")
DRIFT_SEVERITIES = (1, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--robustness-samples", type=int, default=ROBUSTNESS_SAMPLES)
    parser.add_argument("--skip-robustness", action="store_true")
    return parser.parse_args()


def stratified_indices(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per_class = count // len(np.unique(labels))
    chosen = [rng.choice(np.flatnonzero(labels == c), per_class, replace=False) for c in np.unique(labels)]
    return np.sort(np.concatenate(chosen))


def calibrate_probabilities(store: CalibrationStore, logits: np.ndarray, model_id: str) -> np.ndarray:
    from aether.uncertainty.calibration import apply_temperature

    return apply_temperature(logits, store.temperature(model_id))


def drift_study(config, monitor, adapter, images: np.ndarray, labels: np.ndarray, tag: str, family: str | None, severity: int) -> dict:
    """Measure drift severity distribution for one (family, severity) condition."""
    from experiments.common import collect_outputs as collect

    if family is not None:
        images = apply_corruption(images, family, severity, seed=config.runtime.seed)
    outputs = collect(adapter, images, image_size=config.data.image_size, collect_embeddings=True)
    quality = image_quality_matrix(images)
    entropy = predictive_entropy(outputs["probabilities"])
    confidence = max_probability(outputs["probabilities"])
    predicted = outputs["probabilities"].argmax(axis=1)
    window = config.drift.evaluation_window_size
    severities: list[str] = []
    scores: list[float] = []
    for start in range(0, images.shape[0], window):
        stop = min(images.shape[0], start + window)
        if stop - start < config.drift.min_window_size:
            continue
        observations = [
            DriftObservation(
                input_stats=quality[i],
                embedding=outputs["embeddings"][i],
                entropy=float(entropy[i]),
                max_probability=float(confidence[i]),
                predicted_class=int(predicted[i]),
            )
            for i in range(start, stop)
        ]
        report = monitor.evaluate(observations)
        severities.append(report.severity)
        scores.append(report.score)
    flagged = sum(1 for level in severities if level in ("moderate", "severe"))
    return {
        "condition": tag,
        "family": family or "clean",
        "severity": severity,
        "windows": len(severities),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "max_score": float(np.max(scores)) if scores else 0.0,
        "severity_counts": {level: severities.count(level) for level in ("none", "low", "moderate", "severe")},
        "detection_rate": flagged / len(severities) if severities else None,
    }


def build_routing_records(config, light, advanced, monitor, images, labels, family, severity):
    outputs_light = collect_outputs(light, images, image_size=config.data.image_size, collect_embeddings=True)
    outputs_advanced = collect_outputs(advanced, images, image_size=config.data.image_size)
    quality = image_quality_matrix(images)
    entropy = predictive_entropy(outputs_light["probabilities"])
    confidence = max_probability(outputs_light["probabilities"])
    predicted = outputs_light["probabilities"].argmax(axis=1)
    window = config.drift.evaluation_window_size
    drift_scores = np.zeros(images.shape[0])
    drift_severities: list[str] = []
    for start in range(0, images.shape[0], window):
        stop = min(images.shape[0], start + window)
        observations = [
            DriftObservation(
                input_stats=quality[i],
                embedding=outputs_light["embeddings"][i],
                entropy=float(entropy[i]),
                max_probability=float(confidence[i]),
                predicted_class=int(predicted[i]),
            )
            for i in range(start, stop)
        ]
        report = monitor.evaluate(observations)
        drift_scores[start:stop] = report.score
        drift_severities.extend([report.severity] * (stop - start))
    light_ms = float(np.mean(outputs_light["latency_ms_per_batch"]) / min(images.shape[0], config.data.eval_batch_size))
    advanced_ms = float(np.mean(outputs_advanced["latency_ms_per_batch"]) / min(images.shape[0], config.data.eval_batch_size))
    return [
        PolicyRecord(
            light_probabilities=outputs_light["probabilities"][i],
            advanced_probabilities=outputs_advanced["probabilities"][i],
            label=int(labels[i]),
            drift_severity=drift_severities[i],
            drift_score=float(drift_scores[i]),
            light_latency_ms=float(light_ms),
            advanced_latency_ms=float(advanced_ms),
        )
        for i in range(images.shape[0])
    ]


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
    store = CalibrationStore(config.resolve(config.uncertainty.temperature_artifact))

    data = load_cifar10(config.data_root)
    splits = build_and_persist_splits(
        config.data_root, data.train_labels, config.data.splits, config.data.split_seed, data.fingerprint
    )
    test_images, test_labels = data.test_images, data.test_labels
    report: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment_info(config),
        "dataset": {
            "name": "cifar10",
            "test_samples": int(test_labels.size),
            "fingerprint": data.fingerprint["test_sha256"][:16],
            "source": data.source,
        },
        "calibration": {
            role: {
                "temperature": store.temperature(adapter.metadata.model_id),
                "model_id": adapter.metadata.model_id,
            }
            for role, adapter in (("lightweight", light), ("advanced", advanced))
        },
    }
    print("== clean test evaluation ==")
    for role, adapter in (("lightweight", light), ("advanced", advanced)):
        outputs = collect_outputs(adapter, test_images, image_size=config.data.image_size)
        probabilities = calibrate_probabilities(store, outputs["logits"], adapter.metadata.model_id)
        classification = classification_report(probabilities, test_labels, num_classes=adapter.num_classes)
        selective = selective_report(probabilities, test_labels)
        per_sample_ms = float(
            np.mean(outputs["latency_ms_per_batch"])
            / min(config.data.eval_batch_size, test_labels.size)
        )
        report[role] = {
            "model_id": adapter.metadata.model_id,
            "parameters": adapter.metadata.parameters,
            "clean_test": classification,
            "selective": selective,
            "latency_ms_per_sample": per_sample_ms,
            "raw_uncalibrated": classification_report(
                outputs["probabilities"], test_labels, num_classes=adapter.num_classes, include_curves=False
            ),
        }
        print(f"{role}: acc={classification['accuracy']:.4f} ece={classification['ece']:.4f} "
              f"brier={classification['brier']:.4f} aurc={classification['aurc']:.4f} "
              f"{per_sample_ms:.2f} ms/img")

    reference = load_reference(config.resolve(config.drift.reference_artifact))
    monitor = DriftMonitor(reference, config.drift, num_classes=light.num_classes, seed=config.runtime.seed)

    print("== drift study ==")
    dev_images = data.train_images[splits["routing_dev"]]
    dev_labels = data.train_labels[splits["routing_dev"]]
    drift_rows = [drift_study(config, monitor, light, dev_images, dev_labels, "clean_dev", None, 0)]
    for family in DRIFT_FAMILIES:
        for severity in DRIFT_SEVERITIES:
            drift_rows.append(
                drift_study(config, monitor, light, dev_images, dev_labels, f"{family}_s{severity}", family, severity)
            )
    clean_window = [row for row in drift_rows if row["family"] == "clean"]
    corrupted = [row for row in drift_rows if row["family"] != "clean"]
    report["drift"] = {
        "windows": drift_rows,
        "clean_false_positive_rate": clean_window[0]["detection_rate"] if clean_window else None,
        "mean_detection_rate_by_severity": {
            str(severity): float(
                np.mean([row["detection_rate"] for row in corrupted if row["severity"] == severity])
            )
            for severity in DRIFT_SEVERITIES
        },
        "note": (
            "Detection means the composite score reached 'moderate' or 'severe'. Shift detection "
            "is not accuracy verification: labels are used here only to report the study."
        ),
    }
    for row in drift_rows:
        print(f"  {row['condition']:22s} score={row['mean_score']:6.2f} "
              f"detected={row['detection_rate'] if row['detection_rate'] is None else round(row['detection_rate'], 3)}")

    print("== routing study ==")
    records = build_routing_records(config, light, advanced, monitor, dev_images, dev_labels, None, 0)
    for family in ("gaussian_noise", "brightness"):
        take = stratified_indices(dev_labels, 1500, config.runtime.seed + len(family))
        corrupted_images = apply_corruption(dev_images[take], family, 4, seed=config.runtime.seed)
        records.extend(
            build_routing_records(
                config, light, advanced, monitor, corrupted_images, dev_labels[take], family, 4
            )
        )
    policy_path = config.resolve(config.routing.learned_policy.artifact)
    light_risk, advanced_risk = load_risk_models(policy_path)
    precall_risk = load_precall_risk_model(policy_path)
    if precall_risk is None:
        raise SystemExit(
            "learned policy artifact has no pre-call advanced head; rerun training/calibrate.py"
        )
    baseline_policies = ["always_light", "always_advanced", "fixed_threshold", "uncertainty_only", "shift_aware", "aether"]
    routing_rows = [
        simulate_policy(records, config, policy, risk_light_model=light_risk, risk_advanced_model=advanced_risk, risk_precall_model=precall_risk)
        for policy in baseline_policies
    ]
    clean_records = [record for record in records if record.drift_severity == "none"]
    no_drift_records = [dataclasses.replace(record, drift_severity="none", drift_score=0.0) for record in records]
    no_escalation = config.model_copy(
        update={"routing": config.routing.model_copy(update={"max_escalations": 0})}
    )
    ablations = {
        "no_uncertainty": simulate_policy(records, config, "aether", risk_light_model=light_risk, risk_advanced_model=advanced_risk, risk_precall_model=precall_risk, constant_risk=0.5),
        "no_drift_signal": simulate_policy(no_drift_records, config, "aether", risk_light_model=light_risk, risk_advanced_model=advanced_risk, risk_precall_model=precall_risk),
        "no_cost_awareness": simulate_policy(records, config, "shift_aware", risk_light_model=light_risk, risk_advanced_model=advanced_risk, risk_precall_model=precall_risk),
        "no_disagreement": simulate_policy(records, config, "aether", risk_light_model=light_risk, risk_advanced_model=advanced_risk, risk_precall_model=precall_risk, ignore_disagreement=True),
        "no_escalation": simulate_policy(records, no_escalation, "aether", risk_light_model=light_risk, risk_advanced_model=advanced_risk, risk_precall_model=precall_risk),
        "clean_only_subset": simulate_policy(clean_records, config, "aether", risk_light_model=light_risk, risk_advanced_model=advanced_risk, risk_precall_model=precall_risk),
    }
    report["routing"] = {
        "records": len(records),
        "stream": {
            "clean_samples": int(dev_labels.size),
            "corrupted_samples": int(2 * 1500),
            "corrupted_conditions": "gaussian_noise and brightness at severity 4",
            "note": "Policy accuracies below are measured on this mixed stream, not on clean data.",
        },
        "policies": routing_rows,
        "ablations": ablations,
        "interpretation": (
            "Reviews are modelled as correct with probability 1 - reviewer_error_rate and are "
            "charged the configured review cost; escalation is charged its cost and latency."
        ),
    }
    for row in routing_rows:
        print(f"  {row['policy']:18s} cov={row['coverage']:.3f} acc={row['accuracy_on_answered']:.4f} "
              f"lat={row['mean_latency_ms']:7.2f}ms cost={row['expected_cost_per_sample']:.3f}")
    for name, row in ablations.items():
        print(f"  ablate {name:20s} cov={row['coverage']:.3f} acc={row['accuracy_on_answered']:.4f} cost={row['expected_cost_per_sample']:.3f}")

    if not args.skip_robustness:
        print(f"== robustness (n={args.robustness_samples}) ==")
        subset = stratified_indices(test_labels, args.robustness_samples, config.runtime.seed)
        subset_images, subset_labels = test_images[subset], test_labels[subset]
        report["robustness"] = {}
        for role, adapter in (("lightweight", light), ("advanced", advanced)):
            def predict(images: np.ndarray, adapter=adapter) -> np.ndarray:
                outputs = collect_outputs(adapter, images, image_size=config.data.image_size)
                return calibrate_probabilities(store, outputs["logits"], adapter.metadata.model_id)

            rows = robustness_suite(
                predict, subset_images, subset_labels, num_classes=adapter.num_classes,
                n_bins=config.uncertainty.calibration_bins, seed=config.runtime.seed,
            )
            report["robustness"][role] = {
                "rows": rows,
                "summary": summarise_robustness(rows),
            }
            summary = report["robustness"][role]["summary"]
            print(f"  {role}: clean={summary['clean']['accuracy']:.4f} "
                  f"mean_corrupt={summary['mean_corrupted_accuracy']:.4f} "
                  f"worst={summary['worst_corrupted_accuracy']:.4f} "
                  f"mean_ece={summary['mean_corrupted_ece']:.4f}")

    report["protocol"] = {
        "test_split_used_for": "reporting only",
        "routing_dev_split_used_for": "policy development and drift threshold evaluation",
        "robustness_subsample": args.robustness_samples,
        "robustness_severities": list(SEVERITIES),
        "robustness_families": list(CORRUPTION_FAMILIES),
        "seeds": config.runtime.seed,
    }
    json_path = config.resolve("artifacts/reports/benchmarks.json")
    write_report(json_path, report)
    write_markdown(config, report, config.resolve("artifacts/reports/BENCHMARKS.md"))
    return 0


def write_markdown(config, report: dict, path: Path) -> None:
    lines = ["# AETHER-ML benchmark report", ""]
    env = report["environment"]
    lines += [
        f"Generated: {report['generated_at']}  ",
        f"Device: `{env['device']}` ({env['device_name']}), threads: {env['cpu_count']}  ",
        f"PyTorch {env['torch']}, NumPy {env['numpy']}, image size {env['image_size']}, seed {env['seed']}  ",
        f"Test set: {report['dataset']['test_samples']} CIFAR-10 images "
        f"(fingerprint `{report['dataset']['fingerprint']}`)",
        "",
        "## Clean test performance",
        "",
        "| model | accuracy | macro F1 | ECE | Brier | NLL | AURC | ms/sample |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for role in ("lightweight", "advanced"):
        entry = report[role]
        clean = entry["clean_test"]
        lines.append(
            f"| {role} ({entry['parameters']:,} params) | {clean['accuracy']:.4f} | {clean['macro_f1']:.4f} | "
            f"{clean['ece']:.4f} | {clean['brier']:.4f} | {clean['nll']:.4f} | {clean['aurc']:.4f} | "
            f"{entry['latency_ms_per_sample']:.2f} |"
        )
    lines += ["", "Selective prediction (risk at coverage):", ""]
    for role in ("lightweight", "advanced"):
        entry = report[role]["selective"]
        parts = ", ".join(f"{key}->{value:.4f}" for key, value in entry["risk_at_coverage"].items())
        lines.append(f"- **{role}**: AURC {entry['aurc']:.4f}; {parts}")
    lines += ["", "## Robustness", ""]
    if "robustness" in report:
        lines += [
            "| model | clean acc | mean corrupted acc | worst corrupted acc | mean acc drop | mean corrupted ECE |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for role in ("lightweight", "advanced"):
            summary = report["robustness"][role]["summary"]
            lines.append(
                f"| {role} | {summary['clean']['accuracy']:.4f} | {summary['mean_corrupted_accuracy']:.4f} | "
                f"{summary['worst_corrupted_accuracy']:.4f} | {summary['mean_accuracy_drop']:.4f} | "
                f"{summary['mean_corrupted_ece']:.4f} |"
            )
        lines += [
            "",
            f"Worst condition: `{report['robustness']['lightweight']['summary']['worst_condition']['family']}` "
            f"severity {report['robustness']['lightweight']['summary']['worst_condition']['severity']}.",
        ]
    lines += [
        "",
        "## Drift detection",
        "",
        f"- False-positive rate on clean development windows: "
        f"`{report['drift']['clean_false_positive_rate']}`",
    ]
    for severity, rate in report["drift"]["mean_detection_rate_by_severity"].items():
        lines.append(f"- Mean detection rate at severity {severity}: `{rate:.3f}`")
    lines += [
        "",
        "## Routing policies",
        "",
        "Stream: 5,000 clean routing-dev samples plus 3,000 corrupted samples "
        "(gaussian noise and brightness, severity 4). Accuracies below are on this mixed stream.",
        "",
        "| policy | coverage | acc on answered | escalated | unresolved escalation | review | abstain | mean latency (ms) | cost/sample |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["routing"]["policies"]:
        lines.append(
            f"| {row['policy']} | {row['coverage']:.3f} | {row['accuracy_on_answered']:.4f} | "
            f"{row['escalation_rate']:.3f} | {row['unresolved_escalation_rate']:.3f} | "
            f"{row['review_rate']:.3f} | {row['abstention_rate']:.3f} | "
            f"{row['mean_latency_ms']:.2f} | {row['expected_cost_per_sample']:.3f} |"
        )
    lines += ["", "### Ablations", "", "| variant | coverage | acc on answered | cost/sample |", "| --- | --- | --- | --- |"]
    for name, row in report["routing"]["ablations"].items():
        lines.append(
            f"| {name} | {row['coverage']:.3f} | {row['accuracy_on_answered']:.4f} | "
            f"{row['expected_cost_per_sample']:.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Costs are relative units: error 10, escalation 1, review 4, abstention 2. An abstention is "
        "charged the abstention cost and counted as uncovered (never as a wrong answer); a review "
        "carries the review cost plus the expected residual error.",
        "- Shift detection is measured without labels; accuracy impact on unlabelled traffic is not claimed.",
        "- Corruption families and severity tables are documented in `aether/data/corruptions.py`.",
        "- The official test split is used for reporting only and never for tuning thresholds.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
