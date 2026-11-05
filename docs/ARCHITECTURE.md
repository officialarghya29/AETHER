# Architecture — AETHER-ML

## 1. Design principles

1. **One pipeline.** `aether.pipeline.ReliabilityPipeline` is the only orchestration path. The API,
   the CLI and the experiments all call it, so measured behaviour and served behaviour cannot drift
   apart.
2. **Evidence is explicit.** Confidence, uncertainty, drift, risk and correctness are separate
   fields with separate types. No module may collapse them.
3. **No label access at inference.** Routing sees uncertainty, drift, disagreement and cost — never
   ground truth, never the test split.
4. **Deterministic decisions.** Policies are pure functions of an evidence structure; identical
   evidence yields identical actions.
5. **Degrade honestly.** Missing artifacts produce `reference_unavailable`/`uncalibrated` states
   rather than invented numbers.

## 2. Component view

```
                      ┌──────────────────────────────────────────────┐
  image ──► validate ─►│            ReliabilityPipeline               │
                      │                                              │
                      │  adapter(light) ─► logits ─► temperature     │
                      │                         │                    │
                      │                         ├─► uncertainty      │
                      │                         ├─► drift window ────┼──► monitor (PSI/KS/MMD)
                      │                         └─► risk model       │
                      │                                              │
                      │  evidence ─► RoutingController ─► decision    │
                      │                    │                         │
                      │                    ├─ ESCALATE ─► adapter(advanced)
                      │                    ├─ HUMAN_REVIEW           │
                      │                    ├─ ABSTAIN / MONITOR      │
                      │                    └─ ACCEPT                 │
                      │  diagnosis ◄─ quality + uncertainty + drift  │
                      └──────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼───────────────────┐
                    ▼                  ▼                   ▼
              PredictionService   CLI / experiments   ReviewService
                    │                                      │
              PostgreSQL (SQLite locally)             review cases
```

| Layer | Module | Responsibility |
| --- | --- | --- |
| Adapters | `aether.adapters` | Torch inference, embeddings, latency, metadata |
| Models | `aether.models.registry` | Lazy loading, warm-up, versioned checkpoints |
| Uncertainty | `aether.uncertainty` | Entropy, calibration, metrics, MC dropout |
| Drift | `aether.drift` | Reference, statistics, embeddings, monitor |
| Routing | `aether.routing` | Cost model, risk models, policies, controller |
| Diagnosis | `aether.diagnosis` | Image quality and diagnostic rules |
| Feedback | `aether.feedback` | Review queue, verified labels, dataset builder |
| Recovery | `aether.recovery` | Evaluation gate, promotion, registry, rollback |
| Evaluation | `aether.evaluation` | Classification, robustness, routing simulation |
| Service | `backend.app` | HTTP contract, persistence, auth, rate limiting |
| Dashboard | `frontend/` | Live operational views over the API |

## 3. Inference flow

1. **Validate** — decode with byte and pixel limits, centre-crop, reject malformed or oversized input.
2. **Lightweight inference** — batch through the adapter; collect logits, probabilities, embedding.
3. **Calibrate** — apply the model's fitted temperature; compute entropy, MSP, margin.
4. **Drift** — append the observation to a rolling window; evaluate at the configured interval
   against the frozen reference.
5. **Estimate risk** — fitted logistic risk model for the lightweight role, or a documented
   heuristic when no artifact exists.
6. **Estimate advanced risk** — the same evidence through the advanced role model, used only to
   price escalation.
7. **Route** — the configured policy selects an action under budget and availability constraints.
8. **Escalate (conditional)** — run the advanced model, compare predictive distributions, record
   measured advanced risk and disagreement, re-decide once.
9. **Diagnose** — derive findings from quality, uncertainty, drift and escalation outcome.
10. **Record** — persist prediction, uncertainty, routing decision, diagnostics, drift event and a
    review case when required.

Escalation cannot loop: `max_escalations` bounds the second call, and after it runs the advanced
model is marked unavailable for that request.

## 4. Routing flow

```
evidence ─► policy ─┬─ rule_based : ordered thresholds (risk, drift, availability)
                     ├─ cost_aware : argmax expected utility over feasible actions
                     └─ learned    : cost-aware + minimum-benefit gate on escalation
                                │
                        budget checks: max_escalations, latency_budget_ms
                                │
                        RoutingDecision(action, risk, reasons, explanation, utilities)
```

Feasibility is enforced before utility comparison: an unaffordable escalation is removed from the
option set, not merely penalised. When neither escalation nor review is available, the controller
abstains if configured to do so, otherwise returns the prediction with its risk attached.

## 5. Feedback and recovery flow

```
PREDICTION ─► UNCERTAIN CASE ─► REVIEW CASE (pending)
                                    │ verified label + reviewer
                                    ▼
                              RESOLVED CASE ─► CANDIDATE DATASET (manifest + digest)
                                    │
                                    ▼
                         OFFLINE EVALUATION (accuracy, ECE, sample floor)
                                    │ pass + named approver
                                    ▼
                     MODEL REGISTRY (candidate ─► active, previous retired)
                                    │ regression detected
                                    ▼
                                ROLLBACK (history entry)
```

Nothing in this path is automatic: an arriving label never replaces a production model.

## 6. Persistence

Eleven tables: `model_versions`, `predictions`, `uncertainty_records`, `routing_decisions`,
`diagnostic_records`, `drift_events`, `review_cases`, `human_feedback`, `experiments`,
`audit_events`, `dataset_candidates`. Images are stored as files under `artifacts/uploads` with
generated names; the database keeps only a relative reference.

## 7. Deployment view

```
            ┌──────────────┐        ┌──────────────────┐      ┌─────────────┐
  browser ─►│ nginx :80    │──/api─►│ FastAPI :8000    │─────►│ PostgreSQL  │
            │ React bundle │        │ ReliabilityPipeline     │  :5432     │
            └──────────────┘        └──────────────────┘      └─────────────┘
                                             │
                                      artifacts/ (checkpoints, calibration,
                                      reference, reports, uploads)
```

The backend runs as a non-root user, applies migrations on start, and exposes `/health` for
orchestrators. CPU-only operation is the default; CUDA is opt-in through configuration.

## 8. Failure behaviour

| Failure | Behaviour |
| --- | --- |
| Missing checkpoint | Service starts degraded; `/health` reports the missing component; predictions return a sanitised 503 |
| Missing calibration | Predictions proceed uncalibrated; temperature defaults to 1.0 and the response says so |
| Missing drift reference | `status: reference_unavailable`, severity `none`, warning attached |
| Advanced model fails mid-escalation | Recorded as a critical diagnostic; action falls back to review/abstain |
| Database unavailable | Requests fail with a sanitised 500 and a logged cause; no partial success is reported |
| Rate limit exceeded | 429 with a stable error code |
