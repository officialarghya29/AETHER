"""Stability study: risk-controlled acceptance, drift-gated TTA and a severity sweep.

Three questions answered with measurements rather than defaults:

1. **Risk control.** If the acceptance threshold is fitted on calibration data, what selective error
   and coverage does it actually deliver on unseen routing data — clean and corrupted?
2. **Severity sweep.** How does that control degrade as corruption gets worse?
3. **Test-time augmentation.** Does averaging an image's logits with its horizontal mirror improve
   the advanced tier enough to justify the cost, and how often does drift gating let us skip it?

The official test split is not used to choose anything here; thresholds and TTA policy come from
calibration and routing-development data, and the study only reports them.

Outputs ``artifacts/reports/stability.json``, ``STABILITY.md`` and the acceptance-threshold
artifact the API reads (``artifacts/calibration/acceptance_thresholds.json``).
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
from aether.routing.risk_control import (  # noqa: E402
    acceptance_threshold,
    selective_report,
    threshold_artifact,
)
from aether.uncertainty.calibration import apply_temperature  # noqa: E402

REPORTS = ROOT / "artifacts" / "reports"
SNAPSHOT = ROOT / "docs" / "data" / "stability-snapshot.json"
ARTIFACT = ROOT / "artifacts" / "calibration" / "acceptance_thresholds.json"
BENCHMARKS = REPORTS / "benchmarks.json"
ALPHAS = (0.20, 0.10, 0.05, 0.02)
SWEEP_ALPHAS = (0.10, 0.05)
SWEEP_SEVERITIES = (1, 3, 5)
EVAL_IMAGES = 2000
CORRUPTED_IMAGES = 1000
TTA_IMAGES = 2000


def calibrated_probabilities(
    pipeline: ReliabilityPipeline, role: str, images: np.ndarray
) -> np.ndarray:
    adapter = pipeline.registry.get(role)
    batch = prepare_model_input(images, pipeline.config.data.image_size)
    predictions = adapter.predict(batch, return_embeddings=False)
    out = []
    for prediction in predictions:
        temperature = pipeline.calibration.temperature(prediction.model_id)
        out.append(apply_temperature(prediction.logits, temperature))
    return np.stack(out)


def corrupt_stream(
    images: np.ndarray, labels: np.ndarray, severity: int, count: int = CORRUPTED_IMAGES
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    indices = rng.choice(len(images), count, replace=False)
    corrupted = np.stack(
        [
            apply_corruption(images[i][None, ...], "gaussian_noise", severity, seed=int(i))[0]
            for i in indices
        ]
    )
    return corrupted, labels[indices]


def risk_control_study(
    pipeline: ReliabilityPipeline,
    splits: dict[str, np.ndarray],
    images: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[float, float]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit thresholds on calibration data and test them on clean and corrupted routing data."""
    routing = images[splits["routing_dev"]][:EVAL_IMAGES]
    routing_labels = labels[splits["routing_dev"]][: len(routing)]
    corrupted, corrupted_labels = corrupt_stream(routing, routing_labels, 3)

    results: dict[str, Any] = {}
    thresholds: dict[str, dict[float, float]] = {}
    calibration_cache: dict[str, np.ndarray] = {}
    for role in ("lightweight", "advanced"):
        calibration_probs = calibrated_probabilities(pipeline, role, images[splits["calibration"]])
        calibration_labels = labels[splits["calibration"]]
        calibration_cache[role] = calibration_probs
        confidence = calibration_probs.max(axis=1)
        correct = calibration_probs.argmax(axis=1) == calibration_labels

        clean_probs = calibrated_probabilities(pipeline, role, routing)
        corrupted_probs = calibrated_probabilities(pipeline, role, corrupted)

        thresholds[role] = {}
        threshold_rows = []
        for alpha in ALPHAS:
            threshold = acceptance_threshold(confidence, correct, alpha)
            thresholds[role][alpha] = threshold.threshold
            row: dict[str, Any] = {
                "alpha": alpha,
                "threshold": round(threshold.threshold, 4),
                "calibration_coverage": round(threshold.calibration_coverage, 4),
                "calibration_risk": round(threshold.calibration_risk, 4),
                "calibration_bound": round(threshold.upper_bound, 4),
            }
            for stream, probs, stream_labels in (
                ("routing_dev_clean", clean_probs, routing_labels),
                ("routing_dev_corrupted", corrupted_probs, corrupted_labels),
            ):
                report = selective_report(probs, stream_labels, threshold.threshold)
                row[stream] = {
                    "coverage": round(report["coverage"], 4),
                    "selective_error": round(report["selective_error"], 4)
                    if not np.isnan(report["selective_error"])
                    else None,
                    "overall_error": round(report["overall_error"], 4),
                }
            threshold_rows.append(row)
        results[role] = threshold_rows
    return results, thresholds, calibration_cache, {"clean": routing, "corrupted": corrupted}


def severity_sweep(
    pipeline: ReliabilityPipeline,
    splits: dict[str, np.ndarray],
    images: np.ndarray,
    labels: np.ndarray,
    thresholds: dict[str, dict[float, float]],
) -> list[dict[str, Any]]:
    """Evaluate the calibration thresholds at increasing corruption severity."""
    routing = images[splits["routing_dev"]][:CORRUPTED_IMAGES]
    routing_labels = labels[splits["routing_dev"]][: len(routing)]
    rows: list[dict[str, Any]] = []
    for role in ("lightweight", "advanced"):
        stream_probabilities: dict[int, np.ndarray] = {}
        for severity in SWEEP_SEVERITIES:
            corrupted, _ = corrupt_stream(routing, routing_labels, severity, CORRUPTED_IMAGES)
            stream_probabilities[severity] = calibrated_probabilities(pipeline, role, corrupted)
        for alpha in SWEEP_ALPHAS:
            entry: dict[str, Any] = {
                "tier": role,
                "alpha": alpha,
                "threshold": round(thresholds[role][alpha], 4),
                "streams": [],
            }
            for severity in SWEEP_SEVERITIES:
                report = selective_report(
                    stream_probabilities[severity], routing_labels, thresholds[role][alpha]
                )
                entry["streams"].append(
                    {
                        "severity": severity,
                        "coverage": round(report["coverage"], 4),
                        "selective_error": round(report["selective_error"], 4)
                        if not np.isnan(report["selective_error"])
                        else None,
                        "overall_error": round(report["overall_error"], 4),
                    }
                )
            rows.append(entry)
    return rows


def drift_gating(path: Path) -> dict[str, Any]:
    """Read the measured detection rates that decide when the mirror pass is paid."""
    if not path.exists():
        return {"available": False}
    bench = json.loads(path.read_text())
    rows = bench["drift"]["windows"]
    severity_three = [
        row["detection_rate"]
        for row in rows
        if row.get("family") == "gaussian_noise" and row.get("severity") == 3
    ]
    return {
        "available": True,
        "clean_extra_pass_rate": bench["drift"]["clean_false_positive_rate"],
        "corrupted_extra_pass_rate": severity_three[0] if severity_three else None,
    }


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
    corrupted, corrupted_labels = corrupt_stream(images, labels, 3, CORRUPTED_IMAGES)
    corrupted_result = evaluate(corrupted, corrupted_labels)
    gating = drift_gating(BENCHMARKS)
    return {
        "available": True,
        "clean": clean,
        "corrupted_gaussian_noise_severity_3": corrupted_result,
        "drift_gating": gating,
        "cost": "one extra advanced forward pass per escalated request, skipped on clean windows",
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

    print("== risk-controlled thresholds ==")
    risk_control, thresholds, _, _ = risk_control_study(pipeline, splits, images, labels)
    for role, rows in risk_control.items():
        for row in rows:
            clean = row["routing_dev_clean"]
            print(
                f"  {role:<11} alpha={row['alpha']:.2f} threshold={row['threshold']:.3f} "
                f"coverage={clean['coverage']} error={clean['selective_error']}"
            )

    print("== severity sweep ==")
    sweep = severity_sweep(pipeline, splits, images, labels, thresholds)
    for entry in sweep:
        errors = ", ".join(
            f"s{stream['severity']}={stream['selective_error']}" for stream in entry["streams"]
        )
        print(f"  {entry['tier']:<11} alpha={entry['alpha']:.2f} {errors}")

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
            "horizontal-flip logits on the advanced tier only, and drift gating means the mirror "
            "pass is paid only when the monitor flags the window."
        ),
        "environment": {
            "device": "cpu",
            "threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "risk_control": risk_control,
        "severity_sweep": sweep,
        "tta": tta,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "stability.json").write_text(json.dumps(report, indent=2))
    payload = threshold_artifact(thresholds, ARTIFACT)
    write_markdown(report)
    write_snapshot(report)
    print(f"wrote {REPORTS / 'stability.json'}, STABILITY.md and {ARTIFACT}")
    print(f"calibrated budgets: {sorted(payload['tiers'])}")


def _snapshot_tta(tta: dict[str, Any]) -> dict[str, Any]:
    """Keep the published accuracy, cost and gating figures in one recorded place."""
    if not tta.get("available"):
        return {}
    snapshot: dict[str, Any] = {
        name: {
            "identity": tta[name]["identity_accuracy"],
            "tta": tta[name]["tta_accuracy"],
            "delta": tta[name]["delta"],
            "mirror_pass_ms": tta[name]["mirror_pass_ms_per_image"],
        }
        for name in ("clean", "corrupted_gaussian_noise_severity_3")
    }
    if tta.get("drift_gating", {}).get("available"):
        snapshot["drift_gating"] = tta["drift_gating"]
    return snapshot


def write_snapshot(report: dict[str, Any]) -> None:
    """Record the headline numbers so CI can fail when the README drifts from them."""
    payload: dict[str, Any] = {
        "generated_at": report["generated_at"],
        "risk_control": {
            tier: [
                {
                    "alpha": row["alpha"],
                    "threshold": row["threshold"],
                    "clean_coverage": row["routing_dev_clean"]["coverage"],
                    "clean_error": row["routing_dev_clean"]["selective_error"],
                    "corrupted_error": row["routing_dev_corrupted"]["selective_error"],
                }
                for row in rows
            ]
            for tier, rows in report["risk_control"].items()
        },
        "severity_sweep": [
            {
                "tier": entry["tier"],
                "alpha": entry["alpha"],
                "streams": [
                    {
                        "severity": stream["severity"],
                        "coverage": stream["coverage"],
                        "error": stream["selective_error"],
                    }
                    for stream in entry["streams"]
                ],
            }
            for entry in report["severity_sweep"]
        ],
        "tta": _snapshot_tta(report["tta"]),
    }
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(payload, indent=2))


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Stability",
        "",
        "Risk-controlled thresholds, severity sweep and test-time augmentation, measured on",
        "calibration and routing-development data. Regenerate with `make stability`.",
        "",
        "## Risk-controlled acceptance",
        "",
        "| tier | alpha | threshold | coverage (clean) | selective error (clean) | "
        "selective error (corrupted) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for role, rows in report["risk_control"].items():
        for row in rows:
            clean = row["routing_dev_clean"]
            corrupted = row["routing_dev_corrupted"]
            clean_error = clean.get("selective_error")
            corrupted_error = corrupted.get("selective_error")
            lines.append(
                f"| {role} | {row['alpha']:.2f} | {row['threshold']:.3f} | "
                f"{clean['coverage']:.3f} | "
                f"{'—' if clean_error is None else f'{clean_error:.3f}'} | "
                f"{'—' if corrupted_error is None else f'{corrupted_error:.3f}'} |"
            )

    lines += [
        "",
        "## Severity sweep (gaussian noise)",
        "",
        "| tier | alpha | severity | coverage | selective error | overall error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report["severity_sweep"]:
        for stream in entry["streams"]:
            error = stream["selective_error"]
            rendered = "—" if error is None else f"{error:.3f}"
            lines.append(
                f"| {entry['tier']} | {entry['alpha']:.2f} | {stream['severity']} | "
                f"{stream['coverage']:.3f} | {rendered} | {stream['overall_error']:.3f} |"
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
        gating = tta["drift_gating"]
        if gating.get("available"):
            lines += [
                "",
                "Drift gating: the mirror pass is paid on "
                f"{gating['clean_extra_pass_rate'] * 100:.0f}% of clean windows and "
                f"{gating['corrupted_extra_pass_rate'] * 100:.0f}% of severity-3 noisy windows, "
                "using the detector's measured rates.",
            ]
        lines += ["", f"Cost: {tta['cost']}."]
    else:
        lines.append(f"Unavailable: {tta.get('reason')}")
    lines += ["", "## Protocol", "", report["protocol"], ""]
    (REPORTS / "STABILITY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
