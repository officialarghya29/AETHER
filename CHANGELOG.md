# Changelog

All notable changes to AETHER-ML are recorded here. The format follows Keep a Changelog and the
project adheres to semantic versioning.

## [1.0.0] — 2026-11-05

### Added

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
