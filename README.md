# AETHER-ML

**Adaptive Evidence-based Trust, Evaluation & Recovery for ML systems.**

A reliability layer for computer-vision inference: every prediction carries calibrated uncertainty,
drift evidence and a cost-aware routing decision — answer with the fast model, escalate to the
strong model, send to a human, or abstain.

```
evidence → calibrated risk → decision → audit trail
```

No fabricated metrics. Every number below was produced on this machine, on a held-out CIFAR-10
split that was never used for tuning.

---

## Why

A single model serving production traffic has two failure modes that accuracy cannot describe:
it is confidently wrong, and it silently degrades when the world shifts. AETHER-ML treats each
prediction as an evidence-gathering problem — uncertainty, input quality, distribution shift and
model disagreement are measured, fused into a calibrated error probability, and priced against
explicit costs before an answer is emitted.

The system is a two-tier fleet: a **lightweight** MobileNetV3-Small for the common case and an
**advanced** ResNet50 as an escalation target. The routing policy decides per request whether the
second call is worth its latency and cost.

## Measured results

CPU only (8 threads, torch 2.13.0+cpu), 10,000-image official CIFAR-10 test split, fingerprint
`f9b1e2130bab680d`, seed 42. Full report: [`artifacts/reports/BENCHMARKS.md`](artifacts/reports/BENCHMARKS.md)
(regenerate with `make benchmarks`).

| model | params | accuracy | macro F1 | ECE | Brier | AURC | latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lightweight · MobileNetV3-Small | 1.53 M | 0.7874 | 0.7879 | 0.0218 | 0.3018 | 0.0677 | 0.18 ms |
| advanced · ResNet50 | 23.53 M | 0.9442 | 0.9440 | 0.0131 | 0.0893 | 0.0119 | 2.11 ms |

Calibration (temperature scaling fitted on a dedicated 4,000-sample split):

| model | temperature | ECE before | ECE after |
| --- | --- | --- | --- |
| lightweight | 0.7527 | 0.0759 | 0.0207 |
| advanced | 0.7055 | 0.0760 | 0.0125 |

Selective prediction is the point: at 50 % coverage the lightweight model's error rate falls from
21.3 % to 4.5 %, and the advanced model's from 5.6 % to 0.8 %.

**Robustness** — 10 corruption families × 5 severities, 2,500 stratified images per condition:

| model | clean | mean corrupted | worst corrupted | mean ECE (corrupted) |
| --- | --- | --- | --- | --- |
| lightweight | 0.7916 | 0.5318 | 0.2108 | 0.1622 |
| advanced | 0.9468 | 0.6961 | 0.1904 | 0.1312 |

**Drift detection** — PSI on input statistics, per-component KS and permutation MMD on embeddings,
plus KS/χ² on prediction behaviour; Bonferroni-combined into one score with bands at 1.5 / 3.0 / 6.0.
Measured over 200-sample windows, 6 corruption families × 3 severities:

| condition | detection rate |
| --- | --- |
| clean development windows (false positives) | 0.000 |
| severity 1 | 0.793 |
| severity 3 | 1.000 |
| severity 5 | 1.000 |

**Learned risk models** — logistic regression over calibrated uncertainty and the drift band, fitted
on the 4,000-sample calibration split (the routing-dev split is reserved for evaluating policies):

| head | evidence | ROC AUC | Brier |
| --- | --- | --- | --- |
| lightweight error risk | first call | 0.8079 | 0.1761 |
| advanced error risk | after the second call | 0.9017 | 0.1049 |
| advanced pre-call risk | first call, prices escalation | 0.7232 | 0.1575 |

The pre-call head is the honest one: it estimates what the advanced model *would* get wrong from
evidence available **before** paying for the call. Mean predicted risk reduction 0.1525.

**Routing** — replay over 5,000 clean routing-dev samples plus 3,000 corrupted ones (gaussian
noise, brightness; severity 4), which was never used to fit any head. Costs: error 10, escalate 1,
review 4, abstain 2.

| policy | coverage | acc on answered | escalated | review | abstain | ms/img | cost/img |
| --- | --- | --- | --- | --- | --- | --- | --- |
| always_light | 1.000 | 0.6550 | 0.000 | 0.000 | 0.000 | 0.18 | 3.450 |
| always_advanced | 1.000 | 0.8456 | 0.000 | 0.000 | 0.000 | 2.05 | 2.544 |
| fixed_threshold | 0.279 | 0.9588 | 0.741 | 0.000 | 0.721 | 0.26 | 1.594 |
| uncertainty_only | 0.251 | 0.9602 | 0.000 | 0.000 | 0.749 | 0.18 | 1.598 |
| shift_aware | 1.000 | 0.8413 | 0.741 | 0.000 | 0.000 | 1.70 | 2.328 |
| **aether** | **0.653** | **0.9382** | 0.316 | 0.000 | 0.347 | 0.78 | **1.390** |

AETHER answers 65.3 % of the stream at 93.8 % accuracy — 28 points above the fast model it runs
on — for 40 % of the cost of always using the strong model, at 0.78 ms per image. It abstains on the
third of traffic it cannot vouch for instead of guessing; abstentions are charged the abstention cost
and counted as uncovered, never as wrong.

Ablations isolate what each signal buys:

| variant | coverage | acc on answered | cost/img |
| --- | --- | --- | --- |
| no_uncertainty | 0.000 | — | 2.000 |
| no_drift_signal | 0.867 | 0.9056 | 1.577 |
| no_cost_awareness | 1.000 | 0.8413 | 2.328 |
| no_disagreement | 0.652 | 0.9387 | 1.387 |
| no_escalation | 0.481 | 0.9002 | 1.518 |
| clean_only_subset | 0.963 | 0.9460 | 1.062 |

## Architecture

```
                    ┌──────────────────────────────────────────────────────────┐
  image ──────────► │  ReliabilityPipeline  (one object, three entry points)    │
                    │                                                          │
                    │  1  validate · decode · quality signals                  │
                    │  2  lightweight inference ──► logits ──► temperature     │
                    │  3  uncertainty: entropy · max-prob · margin             │
                    │  4  drift monitor: PSI · KS · MMD · behaviour            │
                    │  5  error-risk model ──► P(wrong) per candidate model    │
                    │  6  routing: accept │ escalate │ review │ abstain        │
                    │  7  escalate ──► advanced inference ──► re-decide        │
                    │  8  persist evidence, decision, audit record             │
                    └───────────────┬──────────────────────┬───────────────────┘
                                    │                      │
                        FastAPI /api/v1            CLI (aether predict)
                                    │                      │
                          React dashboard          experiments/run_benchmarks.py
```

The API, the CLI and the offline experiments call the *same* pipeline, so a benchmark row and a
served request follow identical code paths — no second implementation to drift out of sync.

| layer | technology |
| --- | --- |
| core | Python 3.11, NumPy, SciPy, scikit-learn, PyTorch |
| serving | FastAPI, SQLAlchemy 2, Alembic, SQLite / PostgreSQL |
| dashboard | React 18, TypeScript, Vite, recharts |
| ops | Docker Compose, nginx, GitHub Actions |
| quality | pytest (98 tests), ruff, mypy clean over `aether/` |

## Quickstart

```bash
git clone https://github.com/officialarghya29/AETHER.git && cd AETHER
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make data          # materialise CIFAR-10 into data/raw (not committed)
make train         # fine-tune both tiers        ~60 min CPU
make calibrate     # temperatures + risk models
make reference     # freeze the drift reference window
make benchmarks    # full reliability study
```

Serve it:

```bash
make migrate       # create the schema
make api           # http://localhost:8000/docs
make frontend      # http://localhost:5173
```

Or the whole stack: `make docker-up`.

Single-shot inference from the terminal:

```bash
aether status
aether predict path/to/image.png --json
```

## API surface

All routes are under `/api/v1`.

| method | path | purpose |
| --- | --- | --- |
| `GET` | `/health` | liveness, model and drift availability |
| `GET` | `/system/status` | runtime, artifact and database state |
| `POST` | `/predict` | one image → prediction, uncertainty, decision, evidence |
| `POST` | `/predict/batch` | bounded batch with per-item routing |
| `GET` | `/predictions/{id}/uncertainty` | full uncertainty breakdown |
| `GET` | `/diagnostics/{id}` | why this decision was taken |
| `GET` | `/drift/status`, `/drift/events` | live drift state and history |
| `GET` | `/routing/metrics` | action mix, escalation and abstention rates |
| `GET`/`POST` | `/reviews`, `/feedback` | human verification queue and labelled feedback |
| `GET` | `/reports/{name}` | generated benchmark artifacts |

Interactive schema at `/docs`; design notes in [`docs/API.md`](docs/API.md).

## Dashboard

Eight pages over the same evidence store: Overview, Live Inference, Uncertainty & Reliability,
Drift Monitoring, Routing Intelligence, Failure Diagnosis, Human Review, Experiment Centre.

## Reproducibility contract

Splits are fixed by seed and persisted with a SHA-256 fingerprint of the pixel arrays:

```
train 32,000 · validation 4,000 · calibration 4,000 · routing_dev 5,000 · holdout 5,000
official test 10,000 — reported only, never tuned
```

Temperatures are fitted on `calibration`; risk models on `calibration`, evaluated on `routing_dev`;
routing thresholds and drift bands chosen on `routing_dev`; the reference window comes from
`holdout`. The test split is touched once, for the reported numbers.

## Limitations

- CIFAR-10 at 64 px only. Nothing here is validated on natural-resolution or domain-specific imagery.
- 10 of the 15 CIFAR-10-C corruption families are implemented; the remainder are listed as out of
  scope in [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md).
- No adversarial robustness testing.
- Human review is simulated with a 2 % residual reviewer error rate — no real reviewer study.
- Drift detection says the input distribution moved, not that accuracy dropped; no label-free
  accuracy guarantee is claimed.
- Risk models are per-fleet and must be refitted for new architectures or label spaces.
- Feedback is stored and exportable, but retraining is a manual, deliberate step.
- API authentication is configurable and disabled by default for local use.

## Documentation

| document | contents |
| --- | --- |
| [`docs/PRD.md`](docs/PRD.md) | problem, users, scope, success criteria |
| [`docs/SRS.md`](docs/SRS.md) | numbered functional and non-functional requirements |
| [`docs/TDD.md`](docs/TDD.md) | module contracts and test strategy |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | components, data flow, deployment topology |
| [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) | uncertainty, drift, risk and routing mathematics |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | protocol, splits, seeds, how to reproduce |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | architecture decision records |
| [`docs/SECURITY.md`](docs/SECURITY.md) | threat model and controls |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | local, container and production paths |
| [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md) | what is built, verified, or planned |

## License

MIT — © 2026 Arghya Bose. See [`LICENSE`](LICENSE).
