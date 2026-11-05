# Handoff — AETHER-ML v1.0.0

Written at the end of the initial build. This file describes what exists, where it lives, what is
verified, and what a successor should do next.

## 1. Current state

| Area | State |
| --- | --- |
| Core library (`aether/`) | Complete and tested |
| Training | Complete: MobileNetV3-Small 10 epochs, ResNet50 6 epochs, checkpoints present |
| Calibration | Complete: temperatures, both risk models, learned policy artifact |
| Drift reference | Complete: 2,000 holdout samples, 32 PCA components, EV 0.597 |
| Benchmarks | Complete: `artifacts/reports/benchmarks.json` + `BENCHMARKS.md` |
| Backend | Complete; migrations applied; live smoke test performed |
| Frontend | Complete; production build succeeds |
| Tests | 96 passing; ruff clean |
| Containers / CI | Files written; images not built in this environment |
| Git | Single commit on `main`, pushed to `origin` |

## 2. Artifacts and how to regenerate them

```bash
make data          # data/raw/cifar10.npz + cifar10.json (fingerprinted)
make train         # artifacts/checkpoints/{mobilenet_v3_small,resnet50}.pt
make reference     # artifacts/reference/reference_stats.npz
make calibrate     # artifacts/calibration/{temperatures,learned_policy}.json
make benchmarks    # artifacts/reports/{benchmarks.json,BENCHMARKS.md}
```

Approximate wall-clock on the reference CPU (8 threads, no GPU):

| Stage | Time |
| --- | --- |
| MobileNet training (10 epochs) | ~15 min |
| ResNet50 training (6 epochs) | ~47 min |
| Reference build | <1 min |
| Calibration + risk models | ~5 min |
| Benchmarks (full) | ~12 min |

## 3. Exact commands

```bash
# service
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
python -m alembic upgrade head
python -m aether.cli status
python -m aether.cli predict path/to/image.png

# frontend
cd frontend && npm ci && npm run build && npm run dev

# verification
python -m pytest
python -m ruff check aether backend training experiments tests scripts
python -m mypy aether

# containers (verified to build: backend 2.25 GB, frontend 74.5 MB)
docker compose up --build -d
```

## 4. Known limitations (do not paper over)

1. **Corruption coverage** — 10 of 15 CIFAR-10-C families; fog, frost, snow, spatter, elastic are
   absent. Do not describe results as full CIFAR-10-C coverage.
2. **One drift reference per deployment** — the reference is fitted to the lightweight model's
   embedding space. A second model's embeddings are rejected (`dimension_mismatch` warning) rather
   than compared to a foreign basis. Supporting per-model references is the next architectural step.
3. **Per-process drift windows** — with multiple API replicas each holds a partial window. Evaluate
   drift in a dedicated replica or move windows to shared storage.
4. **Reviewer error is assumed** (`routing.reviewer_error_rate = 0.02`), not measured.
5. **CPU-only measurements** — latency numbers are specific to the recorded environment.
6. **Process-local rate limiting** — not a distributed quota.
7. **No dependency-vulnerability scanning in CI** (`pip-audit` / `npm audit` not wired).
8. **MLflow is optional and not exercised** — reports are local JSON by default.
9. **The learned router is a logistic risk model over five features** — a defensible baseline, not a
   novel method.

## 5. Next actions, in order

1. Run the compose stack against PostgreSQL (`docker compose up -d`, then hit `/health` and
   `/api/v1/predict` through the container) — both images build here, but orchestration has not been
   exercised.
2. Add per-model drift references (keyed by `model_id`) so escalation evidence can be compared in
   the advanced model's own embedding space.
3. Move drift windows to shared storage (or a dedicated drift worker) before scaling the API.
4. Measure reviewer accuracy on a labelled sample and replace the assumed error rate.
5. Add `pip-audit` and `npm audit` to CI.
6. Extend corruption coverage to the remaining five families with faithful implementations.
7. Add multiple training seeds and report variance in the benchmark table.
8. Write the research write-up only after (2)-(7), using the existing report as evidence.

## 6. Traps discovered during the build

These were real failures; keep their tests.

- Prediction ids must be canonical 32-character hex, or every follow-up lookup 404s.
- Uncertainty helpers must preserve input rank (scalar for 1-D, array for 2-D).
- `torch.set_num_interop_threads` is process-global and single-shot; configure it once.
- `RoutingController` takes `policy_name`, not `policy`.
- Risk models must be fitted on **error** labels, not correctness labels.
- `drift_score` must not be a risk-model feature: it has two levels in development data.
- A predicted advanced risk must never be treated as a completed evaluation (`advanced_evaluated`).
- Simulation must set `advanced_evaluated=True` after escalation, or the rule policy never returns
  the advanced answer.
- The reports endpoint must resolve under the configured artifact root.
