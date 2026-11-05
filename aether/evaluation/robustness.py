"""Corruption robustness evaluation.

The harness applies the documented corruption suite to a labelled sample, asks a predict
function for probabilities, and records accuracy and calibration degradation per corruption and
severity. It never tunes anything: results are reports, not feedback into thresholds.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aether.data.corruptions import (
    CORRUPTION_FAMILIES,
    SEVERITIES,
    apply_corruption,
    corruption_parameters,
)

PredictFn = Callable[[np.ndarray], np.ndarray]


def stratified_subsample(
    labels: np.ndarray, max_samples: int, seed: int = 0
) -> np.ndarray:
    """Deterministic class-balanced index subsample."""
    if max_samples <= 0 or max_samples >= labels.size:
        return np.arange(labels.size)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, max_samples // classes.size)
    chosen: list[np.ndarray] = []
    for class_index in classes:
        indices = np.flatnonzero(labels == class_index)
        take = min(per_class, indices.size)
        chosen.append(rng.choice(indices, take, replace=False))
    return np.sort(np.concatenate(chosen))


def evaluate_condition(
    predict_fn: PredictFn,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    family: str,
    severity: int,
    num_classes: int = 10,
    n_bins: int = 15,
    seed: int = 0,
) -> dict:
    """Predict one corrupted condition and summarise it."""
    from aether.evaluation.classification import classification_report

    if family == "clean":
        corrupted = images
        parameter = None
    else:
        corrupted = apply_corruption(images, family, severity, seed=seed)
        parameter = corruption_parameters(family, severity)
    probabilities = np.asarray(predict_fn(corrupted), dtype=np.float64)
    if probabilities.shape[0] != labels.shape[0]:
        raise RuntimeError("predict_fn returned a different number of rows than inputs")
    report = classification_report(
        probabilities, labels, num_classes=num_classes, n_bins=n_bins, include_curves=False
    )
    return {
        "family": family,
        "severity": int(severity) if family != "clean" else 0,
        "parameter": parameter,
        "samples": report["samples"],
        "accuracy": report["accuracy"],
        "ece": report["ece"],
        "brier": report["brier"],
        "mean_confidence": report["mean_confidence"],
        "mean_entropy": report["mean_entropy"],
    }


def robustness_suite(
    predict_fn: PredictFn,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    families: tuple[str, ...] = CORRUPTION_FAMILIES,
    severities: tuple[int, ...] = SEVERITIES,
    num_classes: int = 10,
    n_bins: int = 15,
    seed: int = 0,
) -> list[dict]:
    """Run clean plus every (family, severity) condition and return ordered rows."""
    rows = [
        evaluate_condition(
            predict_fn,
            images,
            labels,
            family="clean",
            severity=0,
            num_classes=num_classes,
            n_bins=n_bins,
            seed=seed,
        )
    ]
    for family in families:
        for severity in severities:
            rows.append(
                evaluate_condition(
                    predict_fn,
                    images,
                    labels,
                    family=family,
                    severity=severity,
                    num_classes=num_classes,
                    n_bins=n_bins,
                    seed=seed,
                )
            )
    return rows


def summarise_robustness(rows: list[dict]) -> dict:
    """Aggregate accuracy drop and calibration degradation relative to the clean condition."""
    clean = next((row for row in rows if row["family"] == "clean"), None)
    corrupted = [row for row in rows if row["family"] != "clean"]
    if clean is None or not corrupted:
        return {"clean": clean, "conditions": len(rows), "available": False}
    accuracies = np.array([row["accuracy"] for row in corrupted])
    eces = np.array([row["ece"] for row in corrupted])
    return {
        "available": True,
        "clean": clean,
        "conditions": len(rows),
        "mean_corrupted_accuracy": float(accuracies.mean()),
        "worst_corrupted_accuracy": float(accuracies.min()),
        "worst_condition": min(corrupted, key=lambda row: row["accuracy"]),
        "mean_accuracy_drop": float(clean["accuracy"] - accuracies.mean()),
        "max_accuracy_drop": float(clean["accuracy"] - accuracies.min()),
        "mean_corrupted_ece": float(eces.mean()),
        "mean_ece_increase": float(eces.mean() - clean["ece"]),
        "by_family": {
            family: {
                "mean_accuracy": float(
                    np.mean([row["accuracy"] for row in corrupted if row["family"] == family])
                ),
                "mean_ece": float(
                    np.mean([row["ece"] for row in corrupted if row["family"] == family])
                ),
            }
            for family in sorted({row["family"] for row in corrupted})
        },
    }
