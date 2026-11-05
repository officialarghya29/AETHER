"""Efficiency study: where the time goes, and how it scales with scope.

Three questions, each measured rather than assumed:

1. **Component scope** — preprocessing, inference, calibration/uncertainty, drift bookkeeping,
   risk scoring, routing and persistence.
2. **Batch scope** — throughput versus batch size, with the periodic drift evaluation excluded
   from the per-image steady-state numbers and timed separately.
3. **Feature scope** — the marginal cost of drift bookkeeping, the learned heads and escalation,
   measured by disabling exactly one capability at a time, interleaved across rounds.

The composite drift score runs once per window, not per request, so charging it to every image
would misrepresent steady-state latency (and made a previous version of this harness report a
*negative* drift overhead when evaluation happened to land on a different configuration).

Outputs ``artifacts/reports/efficiency.json`` and ``EFFICIENCY.md``. Regenerate with
``make efficiency``.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aether.config import load_config  # noqa: E402
from aether.data.preprocessing import prepare_model_input  # noqa: E402
from aether.diagnosis.image_quality import analyze_image  # noqa: E402
from aether.drift.monitor import DriftObservation, DriftReport, RollingWindow  # noqa: E402
from aether.pipeline import ReliabilityPipeline, _uncertainty  # noqa: E402
from aether.routing.risk_model import features_from_evidence  # noqa: E402
from aether.uncertainty.calibration import apply_temperature  # noqa: E402

REPORTS = ROOT / "artifacts" / "reports"
ROUNDS = 9


def measure(fn: Callable[[], Any], *, repeat: int = 5, warmup: int = 1) -> dict[str, float]:
    """Run ``fn`` and report mean/p50/p95 milliseconds."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(samples)
    return {
        "mean_ms": round(statistics.fmean(samples), 4),
        "p50_ms": round(statistics.median(samples), 4),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))], 4),
        "min_ms": round(min(samples), 4),
        "samples": len(samples),
    }


@contextmanager
def patched(target: Any, attribute: str, value: Any):
    """Temporarily replace an attribute (used to isolate one capability at a time)."""
    original = getattr(target, attribute)
    setattr(target, attribute, value)
    try:
        yield
    finally:
        setattr(target, attribute, original)


def images_from_test_split(count: int, seed: int = 42) -> np.ndarray:
    data = np.load(ROOT / "data" / "raw" / "cifar10.npz")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data["test_images"]), count, replace=False)
    return data["test_images"][indices].astype(np.uint8)


def rss_mb() -> float:
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 1e6, 1)
    except OSError:
        return 0.0


def stub_evaluation(report: DriftReport) -> Callable[..., DriftReport]:
    """A cheap replacement for the composite evaluation, so timing isolates bookkeeping."""

    def evaluate(_observations: list[DriftObservation]) -> DriftReport:
        return report

    return evaluate


# ------------------------------------------------------------------------ components


def component_scope(pipeline: ReliabilityPipeline, images: np.ndarray) -> dict[str, Any]:
    adapter = pipeline.registry.get("lightweight")
    batch, _qualities = pipeline._prepare(images)
    predictions = adapter.predict(batch)
    calibrated = [
        apply_temperature(p.logits, pipeline.calibration.temperature(p.model_id))
        for p in predictions
    ]
    uncertainties = [_uncertainty(c, pipeline.config.uncertainty.epsilon) for c in calibrated]
    features = [
        features_from_evidence(
            entropy=u["predictive_entropy"],
            max_probability=u["max_probability"],
            margin=u["prediction_margin"],
            drift_score=0.0,
            severity="none",
            disagreement=None,
        )
        for u in uncertainties
    ]
    n = len(images)
    window_report = pipeline._last_drift or DriftReport(severity="none", score=0.0, window_size=0)

    def per_image(fn: Callable[[int], Any]) -> Callable[[], Any]:
        def run() -> Any:
            for index in range(n):
                fn(index)

        return run

    # Drift bookkeeping uses a private window so the pipeline's live state is untouched.
    scratch = RollingWindow(pipeline.config.drift.evaluation_window_size)
    probe_embedding = predictions[0].embedding
    probe_stats = np.zeros(6, dtype=np.float64)

    def append_observation() -> None:
        scratch.add(
            DriftObservation(
                input_stats=probe_stats,
                embedding=probe_embedding.astype(np.float64) if probe_embedding is not None else probe_stats,
                entropy=0.5,
                max_probability=0.8,
                predicted_class=0,
            )
        )

    costs: dict[str, dict[str, Any]] = {
        "quality_analysis": measure(per_image(lambda i: analyze_image(images[i])), repeat=5),
        "preprocess_batch": measure(lambda: prepare_model_input(images, 64), repeat=7),
        "light_inference_batch": measure(lambda: adapter.predict(batch), repeat=7),
        "temperature_lookup": measure(
            per_image(lambda i: pipeline.calibration.temperature(predictions[i].model_id)), repeat=5
        ),
        "calibrate_and_uncertainty": measure(
            per_image(
                lambda i: _uncertainty(
                    apply_temperature(predictions[i].logits, 0.75),
                    pipeline.config.uncertainty.epsilon,
                )
            ),
            repeat=5,
        ),
        "risk_features": measure(
            lambda: [
                features_from_evidence(
                    entropy=u["predictive_entropy"],
                    max_probability=u["max_probability"],
                    margin=u["prediction_margin"],
                    drift_score=0.0,
                    severity="none",
                    disagreement=None,
                )
                for u in uncertainties
            ],
            repeat=7,
        ),
        "risk_heads_two": measure(
            lambda: [pipeline._risk_from_features("lightweight", f) for f in features],
            repeat=7,
        ),
        "drift_bookkeeping": measure(append_observation, repeat=7),
        "routing_decide": measure(
            lambda: [
                pipeline.controller.decide(
                    pipeline._state(
                        risk_light=0.2,
                        uncertainty=u,
                        drift=window_report,
                        elapsed_ms=1.0,
                        escalation_count=0,
                        advanced_available=True,
                        risk_advanced=0.1,
                    )
                )
                for u in uncertainties
            ],
            repeat=7,
        ),
    }
    for cost in costs.values():
        cost["unit"] = "per_image"
        cost["per_image_ms"] = round(cost["mean_ms"] / max(1, n), 4)

    # The composite drift score is evaluated once per window. Time it directly and amortise.
    try:
        monitor = pipeline.drift_monitor()
        snapshot = pipeline.window.snapshot()
        if len(snapshot) >= pipeline.config.drift.min_window_size:
            evaluation = measure(lambda: monitor.evaluate(snapshot), repeat=3)
            evaluation["unit"] = "per_window"
            evaluation["window_size"] = len(snapshot)
            evaluation["per_image_ms"] = round(evaluation["mean_ms"] / len(snapshot), 4)
            costs["drift_evaluate_window"] = evaluation
    except Exception as error:  # pragma: no cover - reference dependent
        costs["drift_evaluate_window"] = {"unit": "per_window", "error": str(error)}
    return costs


# ------------------------------------------------------------------------------ batch


def batch_scaling(pipeline: ReliabilityPipeline, pool: np.ndarray) -> list[dict[str, Any]]:
    """Steady-state throughput, excluding the periodic composite drift evaluation."""
    sizes = (1, 2, 4, 8, 16, 32, 64)
    samples_by_size: dict[int, list[float]] = {size: [] for size in sizes}
    report = DriftReport(severity="none", score=0.0, window_size=len(pipeline.window))
    monitor = pipeline.drift_monitor()
    with patched(monitor, "evaluate", stub_evaluation(report)):
        for size in sizes:
            pipeline.predict_batch(pool[:size])  # warm this shape
        for _ in range(ROUNDS):
            for size in sizes:
                started = time.perf_counter()
                pipeline.predict_batch(pool[:size])
                samples_by_size[size].append((time.perf_counter() - started) * 1000.0)

    rows = []
    for size in sizes:
        samples = samples_by_size[size]
        median_ms = statistics.median(samples)
        rows.append(
            {
                "batch_size": size,
                "mean_batch_ms": round(median_ms, 3),
                "min_batch_ms": round(min(samples), 3),
                "ms_per_image": round(median_ms / size, 3),
                "images_per_second": round(1000.0 * size / median_ms, 1),
                "p50_batch_ms": round(median_ms, 3),
            }
        )
        print(f"  batch {size:>2}: {median_ms / size:7.3f} ms/img  {1000.0 * size / median_ms:7.1f} img/s")
    return rows


# ----------------------------------------------------------------------------- scopes


def scope_costs(pipeline: ReliabilityPipeline, images: np.ndarray) -> dict[str, Any]:
    """Marginal cost of each capability, isolated by disabling one thing at a time.

    Every configuration runs with the composite evaluation stubbed, so the comparison measures
    steady-state work only; the evaluation cost is reported separately in ``components``.
    Configurations are interleaved across rounds and the median is reported, because sequential
    A-then-B timing on a shared CPU is dominated by frequency and contention drift.
    """
    monitor = pipeline.drift_monitor()
    report = DriftReport(severity="none", score=0.0, window_size=len(pipeline.window))
    stub = stub_evaluation(report)

    def no_drift(*args: Any, **kwargs: Any) -> list[DriftReport]:
        return [report for _ in args[0]]

    # Escalation is disabled in every variant: letting a variant change the routing mix would
    # measure ResNet50 calls rather than the feature being isolated. Escalation is timed directly
    # below instead.
    light = (pipeline, "_advanced_present", False)
    variants: dict[str, list[tuple[Any, str, Any]]] = {
        "baseline": [(monitor, "evaluate", stub), light],
        "without_drift": [(monitor, "evaluate", stub), light, (pipeline, "_evaluate_drift", no_drift)],
        "heuristic_risk": [
            (monitor, "evaluate", stub),
            light,
            (pipeline, "_risk_light", None),
            (pipeline, "_risk_advanced", None),
            (pipeline, "_risk_precall", None),
        ],
    }
    order = list(variants)
    samples: dict[str, list[float]] = {name: [] for name in order}

    for name in order:  # warm every configuration once
        with ExitStack() as stack:
            for target, attribute, value in variants[name]:
                stack.enter_context(patched(target, attribute, value))
            pipeline.predict_batch(images)

    for _ in range(ROUNDS):
        for name in order:
            with ExitStack() as stack:
                for target, attribute, value in variants[name]:
                    stack.enter_context(patched(target, attribute, value))
                started = time.perf_counter()
                pipeline.predict_batch(images)
                samples[name].append((time.perf_counter() - started) * 1000.0 / len(images))

    medians = {name: round(statistics.median(values), 4) for name, values in samples.items()}
    evaluation = None
    snapshot = pipeline.window.snapshot()
    if len(snapshot) >= pipeline.config.drift.min_window_size:
        evaluation = measure(lambda: monitor.evaluate(snapshot), repeat=3)

    advanced = pipeline.registry.try_get("advanced")
    escalation_ms: float | None = None
    if advanced is not None:
        prepared, _ = pipeline._prepare(images)
        advanced.predict(prepared)
        calls = []
        for _ in range(5):
            started = time.perf_counter()
            advanced.predict(prepared)
            calls.append((time.perf_counter() - started) * 1000.0 / len(images))
        escalation_ms = round(statistics.median(calls), 3)

    return {
        "baseline_ms_per_image": medians["baseline"],
        "without_drift_ms_per_image": medians["without_drift"],
        "heuristic_risk_ms_per_image": medians["heuristic_risk"],
        "drift_bookkeeping_ms_per_image": round(
            medians["baseline"] - medians["without_drift"], 4
        ),
        "learned_head_ms_per_image": round(medians["baseline"] - medians["heuristic_risk"], 4),
        "escalation_ms_per_escalated_image": escalation_ms,
        "drift_evaluation_ms": (
            round(evaluation["mean_ms"], 3) if evaluation is not None else None
        ),
        "drift_evaluation_amortised_ms_per_image": (
            round(evaluation["mean_ms"] / len(snapshot), 4) if evaluation is not None else None
        ),
        "rounds": ROUNDS,
        "batch_size": len(images),
        "note": (
            "Median of interleaved rounds with escalation disabled (so every variant answers from "
            "the light tier) and the composite evaluation stubbed out; that evaluation runs once "
            "per window and is reported separately. Escalation is timed directly on the advanced "
            "model. BENCHMARKS.md reports the measured escalation mix."
        ),
    }


# ------------------------------------------------------------------------ persistence


def persistence_scope(pipeline: ReliabilityPipeline, images: np.ndarray) -> dict[str, Any]:
    """Cost of a durable write path (SQLite in-memory, same ORM models as the API)."""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from backend.app.services.prediction_service import PredictionService

        try:
            from backend.app.database.base import Base
        except ImportError:  # pragma: no cover - layout guard
            from backend.app.database.models import Base  # type: ignore[attr-defined]

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        raw = [b"" for _ in range(len(images))]
        with Session(engine) as session:
            service = PredictionService(session, pipeline, pipeline.config)

            def persist_once() -> None:
                service.run(images, raw_payloads=raw, persist_image=False)

            persist_once()
            samples = []
            for _ in range(5):
                started = time.perf_counter()
                persist_once()
                samples.append((time.perf_counter() - started) * 1000.0)
            mean_ms = statistics.fmean(samples)
        return {
            "available": True,
            "mean_ms": round(mean_ms, 3),
            "ms_per_prediction": round(mean_ms / len(images), 3),
            "records_per_prediction": 3,
            "backend": "sqlite in-memory via SQLAlchemy ORM",
            "note": "Prediction, routing decision and uncertainty rows; no image bytes.",
        }
    except Exception as error:  # pragma: no cover - environment dependent
        return {"available": False, "error": str(error)}


# ---------------------------------------------------------------------------- memory


def memory_scope(pipeline: ReliabilityPipeline, images: np.ndarray) -> dict[str, Any]:
    before = rss_mb()
    tracemalloc.start()
    for _ in range(40):
        pipeline.predict_batch(images)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after = rss_mb()
    return {
        "rss_before_mb": before,
        "rss_after_mb": after,
        "rss_growth_mb": round(after - before, 1),
        "python_peak_mb": round(peak / 1e6, 2),
        "window_observations": len(pipeline.window),
        "window_capacity": pipeline.config.drift.evaluation_window_size,
        "predictions_in_test": 40 * len(images),
    }


def main() -> None:
    config = load_config()
    torch.set_num_threads(config.runtime.num_threads)
    pipeline = ReliabilityPipeline(config)
    pipeline.warm(("lightweight", "advanced"))
    pool = images_from_test_split(64)
    batch = pool[:16]

    print("== component scope ==")
    components = component_scope(pipeline, batch)
    for name, cost in components.items():
        value = cost.get("per_image_ms")
        print(f"  {name:<26} {value:8.4f} ms/img" if value is not None else f"  {name}: {cost}")

    print("== batch scaling ==")
    batches = batch_scaling(pipeline, pool)

    print("== feature scope ==")
    scopes = scope_costs(pipeline, pool[:8])
    for key, value in scopes.items():
        print(f"  {key}: {value}")

    print("== persistence ==")
    persist = persistence_scope(pipeline, pool[:8])
    print(f"  {persist}")

    print("== memory ==")
    memory = memory_scope(pipeline, pool[:8])
    print(f"  {memory}")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "device": "cpu",
            "cpu_count": os.cpu_count(),
            "threads": torch.get_num_threads(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "sample": {
            "image_size": int(pool.shape[1]),
            "batch_for_components": len(batch),
            "batch_for_scopes": 8,
            "batch_for_memory": 8,
        },
        "components": components,
        "batch_scaling": batches,
        "scopes": scopes,
        "persistence": persist,
        "memory": memory,
        "protocol": (
            "Warmed before timing; repeated runs on CPU; medians across interleaved rounds. "
            "Per-image steady-state numbers exclude the composite drift evaluation, which runs "
            "once per window and is timed separately and amortised. The scope study disables "
            "exactly one capability at a time on the same pipeline instance."
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "efficiency.json").write_text(json.dumps(report, indent=2))
    write_markdown(report)
    print(f"wrote {REPORTS / 'efficiency.json'} and EFFICIENCY.md")


def write_markdown(report: dict[str, Any]) -> None:
    env = report["environment"]
    lines = [
        "# Efficiency",
        "",
        "Measured, not asserted. Regenerate with `make efficiency`.",
        "",
        f"CPU only · {env['cpu_count']} cores · torch {env['torch']} · numpy {env['numpy']} · "
        f"Python {env['python']} · {env['threads']} torch threads",
        "",
        "## Component scope",
        "",
        "| stage | per image | batch p50 | batch p95 |",
        "| --- | --- | --- | --- |",
    ]
    for name, cost in report["components"].items():
        if "p50_ms" not in cost:
            lines.append(f"| {name} | not measured: {cost.get('error', 'n/a')} | — | — |")
            continue
        if cost.get("unit") == "per_window":
            lines.append(
                f"| {name} (per window of {cost.get('window_size', '?')}) | "
                f"{cost['per_image_ms']:.4f} ms amortised | {cost['p50_ms']:.3f} ms | "
                f"{cost['p95_ms']:.3f} ms |"
            )
        else:
            lines.append(
                f"| {name} | {cost['per_image_ms']:.4f} ms | {cost['p50_ms']:.3f} ms | "
                f"{cost['p95_ms']:.3f} ms |"
            )
    lines += [
        "",
        "## Batch scope",
        "",
        "| batch | batch latency | ms / image | images / s |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["batch_scaling"]:
        lines.append(
            f"| {row['batch_size']} | {row['mean_batch_ms']:.3f} ms | {row['ms_per_image']:.3f} | "
            f"{row['images_per_second']:.1f} |"
        )
    scopes = report["scopes"]
    lines += [
        "",
        "## Feature scope (per image, end to end)",
        "",
        "All configurations answer from the light tier (escalation disabled) so the comparison "
        "isolates the capability, not the routing mix.",
        "",
        "| configuration | ms / image | Δ vs baseline |",
        "| --- | --- | --- |",
        f"| baseline | {scopes['baseline_ms_per_image']:.3f} | — |",
        f"| without drift bookkeeping | {scopes['without_drift_ms_per_image']:.3f} | "
        f"{scopes['drift_bookkeeping_ms_per_image']:+.3f} |",
        f"| heuristic heads | {scopes['heuristic_risk_ms_per_image']:.3f} | "
        f"{scopes['learned_head_ms_per_image']:+.3f} |",
        f"| forced escalation (advanced call) | "
        f"{scopes.get('escalation_ms_per_escalated_image') or float('nan'):.3f} | "
        f"only escalated images pay this |",
        "",
        (
            f"Composite drift evaluation: {scopes['drift_evaluation_ms']:.1f} ms per window, "
            f"{scopes['drift_evaluation_amortised_ms_per_image']:.4f} ms/image amortised over the "
            "window."
            if scopes.get("drift_evaluation_ms") is not None
            else "Composite drift evaluation: not measured (reference unavailable)."
        ),
        "",
        "## Persistence",
        "",
    ]
    persist = report["persistence"]
    if persist.get("available"):
        lines.append(
            f"{persist['ms_per_prediction']:.3f} ms per prediction to write the prediction, routing "
            f"decision and uncertainty records ({persist['backend']})."
        )
    else:
        lines.append(f"Persistence could not be measured here: {persist.get('error')}")
    memory = report["memory"]
    lines += [
        "",
        "## Memory",
        "",
        f"RSS grew {memory['rss_growth_mb']:+.1f} MB across {memory['predictions_in_test']} "
        f"predictions; the drift window holds {memory['window_observations']}/"
        f"{memory['window_capacity']} observations.",
        "",
        "## Protocol",
        "",
        report["protocol"],
        "",
    ]
    (REPORTS / "EFFICIENCY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
