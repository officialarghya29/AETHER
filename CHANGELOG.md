# Changelog

All notable changes to AETHER-ML are recorded here. The format follows Keep a Changelog and the
project adheres to semantic versioning.

## [1.0.0] — 2026-11-05

### Changed

- **Licence is now proprietary.** All rights reserved by Arghya Bose; the MIT grant was removed and
  the README, package metadata and dashboard package now state the same terms.
- **README is now evidence-first.** It carries generated figures (`experiments/make_figures.py`),
  dashboard screenshots, side-by-side policy tables, the routing mathematics and an explicit
  real-versus-unverified section.

### Fixed

- **README math render.** GitHub's KaTeX sandbox rejects `\operatorname`, which turned the
  temperature-scaling display block into an error box. The formula now uses `\mathrm{softmax}`
  (and `\mathbf{1}` for the indicator), and `tests/test_readme_math.py` fails the suite if a
  GitHub-blocked macro ever reappears.

- **Efficiency pass.** The calibration artifact was re-read and re-parsed on every prediction;
  it is now cached by modification time and size. The risk feature vector is built once and shared
  by both first-call heads, drift bookkeeping reuses the calibrated uncertainty, channel means are
  vectorised over the batch, and checkpoint presence is cached instead of stat-ed per batch.

- **Escalation was priced with stale risk.** After a completed second call, the cost-aware policy
  still scored ACCEPT with the first-call risk, so it paid for escalation and then abstained. ACCEPT
  and HUMAN_REVIEW are now priced with the measured advanced risk.
- **Cold-start latency blocked escalation.** The first (weight-loading) latency sample was recorded
  as steady state, which made every later escalation look unaffordable; warm-up samples are now
  excluded.
- **Reliability chart rendered empty in the dashboard** because recharts line animation never
  completed before paint; the chart is now drawn without animation.

### Added

- **Error-budget panel and TTA audit trail**: the live-inference page can request a budget and
  shows the calibrated threshold verdict, and the routing timeline marks decisions whose
  escalated call paid for a drift-gated mirror pass.

- **Drift-gated test-time augmentation**: `routing.tta_on_drift` (default on) means the
  mirror pass is paid only for escalated requests whose window the drift monitor flagged,
  measured at 0% of clean windows and 100% of severity-3 windows.
- **Per-severity stability sweep** in `make stability`: coverage and selective error at
  gaussian-noise severities 1, 3 and 5 at two calibrated budgets, plotted as
  `docs/assets/stability-sweep.png`.
- **Error budgets over the API**: `POST /predict` and `/predict/batch` accept an optional
  `error_budget` and return the calibrated acceptance threshold and whether the served
  confidence clears it.

- **Risk-controlled acceptance thresholds** (`aether/routing/risk_control.py`): the
  cut-off is the most permissive calibration prefix whose selective error stays within
  `alpha` under a one-sided Wilson bound, evaluated on clean and corrupted routing data by
  `make stability`. An earlier quantile version did not bound selective error and was
  replaced after the study caught it.
- **Test-time augmentation** for the escalated tier (`routing.tta`), averaging identity and
  horizontal-flip logits; measured as opt-in and documented with its latency cost.
- **Non-finite output guard**: a model that returns NaN/Inf logits now raises `InferenceError`
  instead of serving a confidently meaningless decision.

- **Efficiency study** (`make efficiency`): component, batch, feature, persistence and memory
  scopes, reported in `artifacts/reports/EFFICIENCY.md` and summarised in the README.

- **Reliability pipeline.** One orchestration path (`aether.pipeline.ReliabilityPipeline`) shared by
  the HTTP API, the CLI and the benchmark suite, so served and measured behaviour are identical.
- **Model adapters.** MobileNetV3-Small and ResNet50 fine-tuned from ImageNet weights on CIFAR-10 at
  64 px, with a common adapter contract, embedding extraction, latency accounting and version tags.
- **Uncertainty engine.** Predictive entropy, normalised entropy, MSP, prediction margin, temperature
  scaling, ECE/MCE, Brier, NLL, reliability curves, risk-coverage curves and MC dropout.
- **Drift engine.** PSI, KS and permutation MMD on input statistics and embeddings, plus
  prediction-behaviour tests, Bonferroni-corrected composite severity, rolling windows and frozen
  reference artifacts.
- **Adaptive router.** Rule-based, cost-aware and learned policies over one evidence structure, with
  escalation budgets, latency affordability checks and an explicit estimate-versus-evaluated
  distinction for advanced-model risk.
- **Diagnosis engine.** Evidence-backed diagnostic codes with severity, explanation and recommended
  action, plus image-quality indicators (blur, brightness, contrast, channel statistics).
- **Feedback and recovery.** Review queue, verified-label validation, candidate dataset builder with
  manifest and digest, offline promotion gate, model registry and rollback.
- **FastAPI backend.** Prediction, batch, uncertainty, diagnostics, drift, routing, reviews, feedback,
  experiments, metrics and reports endpoints; Alembic migrations; request ids; rate limiting;
  optional bearer auth; sanitised error contract.
- **React dashboard.** Eight pages driven by live API data: overview, live inference, reliability,
  drift monitoring, routing intelligence, failure diagnosis, human review and experiment centre.
- **Experiment suite.** Clean, robustness, drift, routing and ablation studies writing
  `artifacts/reports/benchmarks.json` and `BENCHMARKS.md`.
- **Deployment.** Non-root backend image, nginx-served dashboard, PostgreSQL compose stack, CI
  workflow for lint, typecheck, tests, frontend build and container build.

### Notes

- Corruption coverage is 10 of the 15 CIFAR-10-C families; fog, frost, snow, spatter and elastic
  transform are intentionally out of scope for v1.0.0 and documented as such.
- Training ran on CPU; timings in the benchmark report describe that hardware only.
