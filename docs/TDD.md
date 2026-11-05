# Technical Design Document — AETHER-ML

## 1. Module boundaries

| Package | Public surface | Depends on |
| --- | --- | --- |
| `aether.config` | `AetherConfig`, `load_config` | pydantic |
| `aether.adapters` | `ModelAdapter`, `TorchvisionAdapter` | torch, torchvision, numpy |
| `aether.models` | `ModelRegistry` | adapters, config |
| `aether.uncertainty` | entropy, `CalibrationStore`, metrics, MC dropout | numpy, scipy |
| `aether.drift` | `DriftMonitor`, `build_reference`, statistics | numpy, scipy, sklearn |
| `aether.routing` | `RoutingController`, policies, `RiskModel` | config, drift, numpy |
| `aether.diagnosis` | `DiagnosisEngine`, `analyze_image` | numpy, scipy |
| `aether.feedback` | review queue, dataset builder | numpy |
| `aether.recovery` | evaluation gate, registry, promotion, rollback | feedback, numpy |
| `aether.evaluation` | classification, robustness, routing simulation | uncertainty, routing |
| `aether.pipeline` | `ReliabilityPipeline`, `InferenceResult` | all of the above |
| `backend.app` | FastAPI app, routes, services, database | aether |

Only `aether.pipeline` may touch every subsystem. Services depend on the pipeline, never on
individual detectors, which keeps the HTTP layer free of ML logic.

## 2. Key interfaces

```python
class ModelAdapter(Protocol):
    metadata: ModelMetadata
    role: str
    num_classes: int
    def predict(self, images: np.ndarray) -> list[Prediction]: ...
    def predict_batch_tensor(self, batch: torch.Tensor) -> list[Prediction]: ...

class DecisionPolicy(Protocol):
    name: str
    def decide(self, state: RoutingState, config: RoutingConfig) -> RoutingDecision: ...
```

`RoutingState` is frozen and contains only inference-time evidence:

```python
risk_light, entropy, max_probability, margin, drift_score, drift_severity,
risk_advanced, advanced_evaluated, disagreement, escalation_count,
advanced_available, escalation_affordable, elapsed_ms, remaining_budget_ms
```

`advanced_evaluated` is deliberately separate from `risk_advanced`: a predicted advanced risk may
price an escalation, but only a completed second call may justify returning the advanced answer.

## 3. Data model

| Table | Purpose | Notable constraints |
| --- | --- | --- |
| `predictions` | One row per answered request | 32-char hex id, action, model id/version, risk JSON, latency |
| `uncertainty_records` | Raw and calibrated uncertainty | FK to prediction, temperature, calibrated flag |
| `routing_decisions` | Decision evidence | action, policy, risk, reason codes, utilities |
| `diagnostic_records` | Findings | code, severity, evidence JSON |
| `drift_events` | Composite drift assessments | severity, score, window size, status |
| `review_cases` | Human queue | unique per prediction, status machine |
| `human_feedback` | Verified labels | reviewer, verified flag, timestamps |
| `model_versions` | Registry | unique (role, version), status |
| `experiments` | Run metadata | config JSON, metrics JSON |
| `audit_events` | Append-only trail | actor, action, payload |

All tables carry created/updated timestamps and indexed foreign keys. Images are never stored as
blobs; `predictions.image_ref` points at a generated file under the artifact root.

## 4. Configuration

`configs/aether.yaml` is parsed into strict Pydantic models. Cross-field invariants are validated
after construction:

- split sizes must sum to the available training partition;
- routing risks must satisfy `accept_risk <= escalate_risk <= high_risk`;
- drift severities must be ordered;
- device selection rejects `cuda` when CUDA is unavailable at runtime.

Environment overrides use `AETHER__SECTION__KEY` and `DATABASE_URL`; secrets are read from the
environment only.

## 5. Algorithm summary

| Concern | Method | Artifact |
| --- | --- | --- |
| Calibration | single-parameter temperature scaling, NLL objective, L-BFGS-B bounded in [0.05, 20] | `temperatures.json` |
| Error risk | L2-regularised logistic regression over 6 evidence features | `learned_policy.json` |
| Advanced risk | second logistic model including disagreement | `learned_policy.json` |
| Input drift | PSI over decile bins; two-sample KS | computed per window |
| Embedding drift | RBF MMD with permutation p-values on PCA components | computed per window |
| Behaviour drift | entropy/confidence KS, class-frequency chi-square | computed per window |
| Composite | max Bonferroni-adjusted `-log10 p`, mapped to bands | per window |
| Routing | expected utility with explicit costs and feasibility | per request |
| Promotion | accuracy floor, accuracy gain, ECE ceiling, sample floor | registry history |

## 6. Concurrency and resource use

- The pipeline is created once per process; the lightweight adapter is warmed at startup, the
  advanced adapter lazily on first escalation.
- Torch inference runs under `inference_mode` with worker threads configured once per process.
- SQLAlchemy sessions are per request; commits happen in the route after the service returns.
- Uploads are size- and pixel-limited before decoding; temporary data stays in memory.
- Drift windows are bounded ring buffers; evaluation runs every `evaluation_window_size` samples.

## 7. Testing strategy

| Level | Scope |
| --- | --- |
| Unit | entropy/calibration/metrics, PSI/KS/MMD, PCA, routing policies and risk models, diagnosis rules, image quality, split construction, corruption determinism, evidence validation |
| Integration | pipeline with real (small) checkpoints, database persistence, API contract, error shapes, rate limiting, review lifecycle, candidate dataset, promotion and rollback |
| Failure | malformed upload, oversized batch, missing artifact, duplicate review submission, unknown report, invalid configuration |
| Performance | per-sample latency recorded in the benchmark report; no hardcoded thresholds |

Model weights used in tests are randomly initialised fixtures. They verify plumbing, not accuracy;
accuracy claims come only from the trained checkpoints and the benchmark report.

## 8. Dependency decisions

| Dependency | Why |
| --- | --- |
| torch + torchvision | Required for the reference implementation; CPU wheels keep CI light |
| numpy/scipy/scikit-learn | Statistics, PCA, logistic regression |
| FastAPI + Pydantic v2 | Typed HTTP contract and validation |
| SQLAlchemy 2 + Alembic | Explicit schema and reversible migrations |
| Pillow | Safe image decoding with limits |
| MLflow (optional extra) | Experiment tracking; local JSON reports are the default |
| React/Vite/Tailwind/Recharts | Typed dashboard with small runtime surface |

Rejected: Redis, Celery, Kafka, Kubernetes — no requirement in v1.0.0 justifies distributed state
or a task queue.
