# Implementation status — AETHER-ML v1.0.0

Evidence for every phase is recorded here. Numbers come from the artifacts named in each row; none
are estimated. Where something was not executed, it says so.

## Phase 0 — Repository intelligence

The workspace started empty apart from a new git repository. Audit findings: no existing source, no
dataset, no weights, Python 3.12 with PEP 668 (system pip blocked), no GPU, 8 usable CPU threads.
Decision: create a virtual environment with system site packages for tooling, and run training with
the system interpreter that already had PyTorch.

## Phase 1 — Requirements and architecture

Written: `docs/PRD.md`, `docs/SRS.md`, `docs/TDD.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`
(18 decisions).

## Phase 2 — Environment and foundation

- `pyproject.toml` with pinned ranges, ruff, pytest and mypy configuration.
- `aether/config.py` (strict Pydantic models with cross-field validation), `aether/utils/`,
  `aether/errors.py`, `aether/runtime.py`.
- `configs/aether.yaml`, `.env.example`, `.gitignore`.
- Evidence: `python -c "import backend.app.main"` succeeds; `python -m aether.cli status` runs.

## Phase 3 — Data and models

- CIFAR-10 materialised from the Hugging Face Parquet mirror after the Toronto host proved unusable
  (~16 KB/s). Fingerprints: `data/raw/cifar10.json`, train `c5b8e1d062f7…`.
- Stratified splits 32k/4k/4k/5k/5k, seed 42, persisted to `data/processed/splits.json`.
- Training (`training/train.py`): MobileNetV3-Small 10 epochs (best val 0.7885), ResNet50 6 epochs
  (best val 0.9520, epoch 5). Reports: `artifacts/reports/training_{lightweight,advanced}.json`.
- Clean test results (from `artifacts/reports/benchmarks.json`):

| model | accuracy | ECE | Brier | AURC | ms/sample |
| --- | --- | --- | --- | --- | --- |
| lightweight | 0.7874 | 0.0218 | 0.3018 | 0.0677 | 0.17 |
| advanced | 0.9442 | 0.0131 | 0.0893 | 0.0119 | 1.93 |

## Phase 4 — Uncertainty engine

Implemented entropy, normalised entropy, MSP, margin, temperature scaling, ECE/MCE, Brier, NLL,
reliability curves, risk–coverage/AURC and MC dropout.
Fitted on the calibration split only: lightweight T=0.7527 (ECE 0.0759→0.0207), advanced T=0.7055
(ECE 0.0760→0.0125). Artifact: `artifacts/calibration/temperatures.json`.

## Phase 5 — Drift engine

PSI, KS, permutation MMD, PCA embedding projection, prediction-behaviour tests, Bonferroni
composite severity, rolling windows, frozen reference.
Reference: 2,000 holdout samples, 32 components, explained variance 0.597
(`artifacts/reports/reference.json`).
Measured behaviour: clean-window false-positive rate 0.0; mean detection rate 0.793 at severity 1,
1.000 at severities 3 and 5.

## Phase 6 — Adaptive router

Rule-based, cost-aware and learned policies; escalation budgets; latency affordability; explicit
estimate-versus-evaluated distinction; deterministic decisions.
Risk models fitted on the calibration split: lightweight fit AUC 0.8079, post-escalation advanced
0.9017, pre-call advanced 0.7232; expected pre-call risk reduction 0.1525
(`artifacts/calibration/learned_policy.json`). Policies are evaluated out-of-sample on the
routing-development split, never on their fitting data.

## Phase 7 — Diagnosis and feedback

Ten diagnostic codes with severity, evidence, explanation and recommended action; image-quality
indicators; review queue; verified-label validation; candidate dataset with manifest and digest;
offline promotion gate; model registry and rollback.

## Phase 8 — Backend

FastAPI app with health, system, models, prediction, batch, uncertainty, diagnostics, drift, routing,
metrics, reviews, feedback, experiments and reports endpoints. Pydantic v2 schemas, request ids,
rate limiting, optional bearer auth, sanitised error contract.
Evidence: `tests/test_pipeline_api.py` exercises 11 HTTP scenarios including duplicate review (409),
blank reviewer (400), undecodable upload (422) and rate limiting (429).

## Phase 9 — Database

Eleven tables with foreign keys, indexes and timestamps. Alembic migration
`3eeafdd0b290_initial_aether_schema.py` generated and applied against SQLite in development;
the compose stack applies it to PostgreSQL on start.

## Phase 10 — Frontend

Eight pages (Overview, Live Inference, Reliability, Drift Monitoring, Routing Intelligence, Failure
Diagnosis, Human Review, Experiment Centre) reading only live API data, with explicit empty and
error states. Evidence: `npm run build` succeeds (640 modules; 640 kB JS / 186 kB gzip).

## Phase 11 — Monitoring and experiment tracking

Structured events (`routing.decision`, `prediction.completed`), in-process metrics, and file-based
reports. MLflow is available as an optional extra but was not exercised in this build.

## Phase 12 — Security

Upload limits, safe decoding, generated filenames, explicit CORS, rate limiting, constant-time token
comparison, non-root container, no secrets in the repository. Documented in `docs/SECURITY.md`.

## Phase 13 — Testing

- `python -m pytest` → **96 passed**.
- `python -m ruff check aether backend training experiments tests scripts` → clean.
- Coverage of unit, integration, failure and API contract paths. No test was skipped or weakened to
  make the suite pass; three defects were found and fixed through failing tests instead.

## Phase 14 — Scientific evaluation

Full study in `artifacts/reports/benchmarks.json` and `BENCHMARKS.md`: clean performance, selective
prediction, robustness over 10 families × 5 severities, drift detection rates, six routing policies,
and six ablations. Protocol: test split for reporting only; routing data from the development split.

## Phase 15 — Documentation

`README.md` plus `docs/` PRD, SRS, TDD, ARCHITECTURE, ALGORITHMS, EXPERIMENTS, API, SECURITY,
DEPLOYMENT, DECISIONS, CONTRIBUTING, IMPLEMENTATION_STATUS, HANDOFF.

## Phase 16 — Deployment

`Dockerfile` (non-root, health check, CPU wheels), `frontend/Dockerfile` + `nginx.conf`,
`docker-compose.yml` (PostgreSQL, migrations on start, dashboard).

**Verified:** both images build locally with Docker 29.9.3 / Compose v5.7.0
(`docker compose build`: backend 2.25 GB, frontend 74.5 MB). The stack has not been run against
PostgreSQL in this environment, so compose orchestration remains *implemented and built, not
orchestrated here*.

## Phase 17 — Release engineering

`.github/workflows/ci.yml` (lint, typecheck, tests, frontend build, both container builds),
`CHANGELOG.md`, `LICENSE`, conventional commits, single `main` branch.

**Verified locally:** `ruff check` clean, `mypy aether` clean over 52 modules (pydantic plugin
enabled), `pytest` 99 passing. **Not verified:** the workflow itself has not executed on GitHub
Actions from this machine.

## Honest summary

| Item | State |
| --- | --- |
| Core reliability pipeline | Implemented and verified |
| Models trained and benchmarked | Implemented and verified |
| Calibration without leakage | Implemented and verified |
| Drift detection | Implemented and verified |
| Routing policies | Implemented and verified offline |
| Feedback and recovery | Implemented and verified by tests |
| API + database | Implemented and verified by tests and a live smoke test |
| Dashboard | Implemented; production build verified; not visually audited in a browser |
| Containers | Images build locally; compose stack not orchestrated here |
| CI workflow | Written, not executed here |
| mypy | Clean over `aether/` (52 modules) |
| Full CIFAR-10-C coverage | Not implemented (10 of 15 families) |
