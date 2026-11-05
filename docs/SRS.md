# Software Requirements Specification — AETHER-ML

Requirement IDs are stable and referenced by tests and by `docs/IMPLEMENTATION_STATUS.md`.

## 1. Functional requirements

### Model layer

| ID | Requirement |
| --- | --- |
| FR-1.1 | The system shall expose one adapter contract returning class index, label, logits, probabilities, embedding (when available), latency and model id. |
| FR-1.2 | The system shall load MobileNetV3-Small (lightweight) and ResNet50 (advanced) from versioned checkpoints. |
| FR-1.3 | The system shall validate checkpoint metadata (architecture, class count, input size) at load time. |
| FR-1.4 | All models shall share the CIFAR-10 class ordering; mismatched label spaces shall be rejected. |
| FR-1.5 | Inference shall run on CPU, and on CUDA when available and configured. |

### Uncertainty

| ID | Requirement |
| --- | --- |
| FR-2.1 | The system shall compute predictive entropy, normalised entropy, maximum softmax probability and top-2 margin. |
| FR-2.2 | Uncertainty routines shall be numerically stable for zero probabilities and extreme logits. |
| FR-2.3 | Temperature scaling shall be fitted on the calibration split only and refuse a test split. |
| FR-2.4 | The system shall report ECE, MCE, Brier, NLL, reliability curves and risk-coverage/AURC. |
| FR-2.5 | Calibration artifacts shall record model id, split, sample count, temperature and before/after metrics. |
| FR-2.6 | MC dropout shall be available and disabled by default. |

### Drift

| ID | Requirement |
| --- | --- |
| FR-3.1 | The system shall persist a frozen reference window built from the holdout split. |
| FR-3.2 | The system shall compute PSI, two-sample KS and permutation MMD over input statistics and embeddings. |
| FR-3.3 | The system shall monitor prediction behaviour: entropy, confidence, class mix and disagreement. |
| FR-3.4 | Multiple comparisons shall be corrected before composing a severity score. |
| FR-3.5 | Severity bands shall be configurable and ordered (low < moderate < severe). |
| FR-3.6 | The system shall report `reference_unavailable` and `insufficient_window` instead of a fabricated score. |
| FR-3.7 | The system shall never report accuracy drift from unlabelled traffic. |

### Routing

| ID | Requirement |
| --- | --- |
| FR-4.1 | The router shall support ACCEPT, MONITOR, ESCALATE, HUMAN_REVIEW and ABSTAIN. |
| FR-4.2 | Policy selection shall be configurable (`rule_based`, `cost_aware`, `learned`). |
| FR-4.3 | Decisions shall be deterministic for identical evidence and configuration. |
| FR-4.4 | Escalation shall be bounded by `max_escalations` and by the latency budget. |
| FR-4.5 | A predicted advanced-model risk may price escalation; only a completed evaluation may justify returning the advanced risk. |
| FR-4.6 | Every decision shall carry reason codes and a human-readable explanation. |
| FR-4.7 | The learned policy's gate shall suppress escalation when predicted benefit is below the configured minimum. |

### Diagnosis

| ID | Requirement |
| --- | --- |
| FR-5.1 | Diagnostics shall include code, severity, evidence, explanation and recommended action. |
| FR-5.2 | Image quality shall report blur, brightness, contrast, dimensions and channel statistics. |
| FR-5.3 | Explanations shall not present correlation as proven causation. |

### Feedback and recovery

| ID | Requirement |
| --- | --- |
| FR-6.1 | Uncertain predictions shall create review cases with model evidence and an image reference. |
| FR-6.2 | A verified label shall require a non-empty reviewer identity and a label inside the class space. |
| FR-6.3 | Review state transitions shall be append-only and auditable. |
| FR-6.4 | Candidate datasets shall include only verified cases and record a manifest and digest. |
| FR-6.5 | Promotion shall require a passing offline gate and a named approver. |
| FR-6.6 | Rollback shall restore the previous active version and be recorded in history. |

## 2. API requirements

| ID | Requirement |
| --- | --- |
| FR-7.1 | Requests and responses shall be validated by Pydantic v2 models. |
| FR-7.2 | Errors shall use one shape: `{"error": {"code", "message", "details", "request_id"}}`. |
| FR-7.3 | Uploads shall be decoded safely, size- and pixel-limited, and never trusted by filename. |
| FR-7.4 | Every response shall carry an `X-Request-Id` header. |
| FR-7.5 | List endpoints shall paginate with a bounded page size. |
| FR-7.6 | Protected review and feedback operations shall support bearer authentication when enabled. |
| FR-7.7 | Prediction responses shall include id, class, probabilities, uncertainty, drift, model, action, explanation, version and latency. |
| FR-7.8 | No response shall expose filesystem paths, secrets or raw stack traces. |

## 3. Non-functional requirements

| ID | Requirement |
| --- | --- |
| NFR-1 | CPU-only inference shall be supported for the whole stack. |
| NFR-2 | A single-image prediction shall complete within the configured timeout on the reference CPU. |
| NFR-3 | Training and evaluation shall be reproducible from a recorded seed, fingerprint and config. |
| NFR-4 | Configuration shall be validated at startup; invalid combinations shall refuse to start. |
| NFR-5 | The service shall start in degraded mode when an artifact is missing, and `/health` shall say so. |
| NFR-6 | The database schema shall be created by Alembic migrations in production. |
| NFR-7 | Logs shall be structured and shall not contain secrets or uploaded payloads. |

## 4. Security requirements

| ID | Requirement |
| --- | --- |
| SR-1 | Secrets shall come from the environment; none shall be committed. |
| SR-2 | CORS origins shall be explicit and configurable. |
| SR-3 | Requests shall be rate limited per client. |
| SR-4 | Uploaded data shall never be executed and shall be stored under generated names. |
| SR-5 | Review and feedback endpoints shall be authenticated before public exposure. |
| SR-6 | The container shall run as a non-root user. |

## 5. Error handling and logging

| ID | Requirement |
| --- | --- |
| ER-1 | Domain errors shall map to stable HTTP statuses: 400 validation, 404 missing, 409 conflict, 413 too large, 422 invalid input, 429 rate limit. |
| ER-2 | Unexpected exceptions shall be logged with the request id and returned as a sanitised 500. |
| ER-3 | Routing, prediction and review actions shall emit structured events. |
| ER-4 | Exceptions shall never be swallowed without a log entry. |

## 6. Scientific integrity requirements

| ID | Requirement |
| --- | --- |
| SI-1 | The official test split shall be used for reporting only. |
| SI-2 | Thresholds and policies shall be developed on the routing_dev split. |
| SI-3 | Temperatures shall be fitted on the calibration split. |
| SI-4 | Every reported number shall be traceable to a command and an artifact. |
| SI-5 | Unverified claims shall be labelled as such, not omitted silently. |
