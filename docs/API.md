# API — AETHER-ML

Base path: `/api/v1`. Interactive schema: `/docs`. All responses carry `X-Request-Id`; a client may
send one to correlate logs.

## Errors

```json
{
  "error": {
    "code": "invalid_input",
    "message": "unable to decode image: not a supported format",
    "details": null,
    "request_id": "5f0c1d2e..."
  }
}
```

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | `validation_failed` | Domain rule violated (e.g. blank reviewer) |
| 404 | `not_found` | Unknown prediction, review, experiment or report |
| 409 | `conflict` | Illegal state transition (duplicate review submission) |
| 413 | `too_large` | Upload exceeds byte or pixel limits |
| 422 | `invalid_input` | Undecodable image, bad batch composition |
| 429 | `rate_limited` | Client exceeded the configured rate |
| 503 | `model_unavailable` | Checkpoint missing or failed to load |
| 500 | `internal_error` | Sanitised unexpected failure |

## Endpoints

### Health and system

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | `ok` or `degraded`; per-component status |
| GET | `/api/v1/system/status` | Runtime, models, artifacts, policy |
| GET | `/api/v1/models` | Registered adapters with metadata |
| GET | `/api/v1/models/{model_id}` | One model |

```bash
curl -s localhost:8000/health | jq
```

### Prediction

```bash
curl -s -X POST localhost:8000/api/v1/predict \
  -F "file=@sample.png"
```

```json
{
  "prediction_id": "6f1c...e9",
  "action": "HUMAN_REVIEW",
  "predicted_index": 3,
  "predicted_label": "cat",
  "probabilities": [0.02, 0.01, 0.05, 0.41, 0.03, 0.19, 0.04, 0.02, 0.19, 0.04],
  "uncertainty": {"predictive_entropy": 1.82, "max_probability": 0.41, "prediction_margin": 0.22},
  "risk": {"value": 0.44, "source": "learned"},
  "drift": {"severity": "none", "score": 0.0, "status": "ok", "window_size": 200},
  "model_id": "lightweight:mobilenet_v3_small@1.0.0",
  "model_version": "1.0.0",
  "decision": {"policy": "cost_aware", "explanation": "...", "reason_codes": ["UTILITY_HUMAN_REVIEW"]},
  "diagnostics": [{"code": "HIGH_ENTROPY", "severity": "medium", "evidence": {"predictive_entropy_nats": 1.82}}],
  "latency_ms": 12.4,
  "review_id": "9c2a...",
  "created_at": "2026-11-05T00:00:00Z"
}
```

| Method | Path | Notes |
| --- | --- | --- |
| POST | `/predict` | Single image, max 5 MB / 4.1 MP |
| POST | `/predict/batch` | Up to 16 files, identical dimensions |
| GET | `/predictions` | Paginated history (`limit`, `offset`, `action`) |
| GET | `/predictions/{id}/uncertainty` | Raw + calibrated uncertainty, temperature, risk |
| GET | `/diagnostics/{id}` | Findings for a prediction |

### Drift, routing, metrics

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/drift/status` | Latest assessment, or `reference_unavailable` |
| GET | `/drift/events` | Recorded moderate/severe events |
| GET | `/routing/decisions` | Recent decisions with explanations |
| GET | `/routing/metrics` | Action distribution, escalation rate, latency |
| GET | `/metrics/summary` | Counters, backlog, service gauges, budget-miss rate |

Drift endpoints never report accuracy impact: labels are not available at inference time.

### Human review and feedback

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/reviews?status=pending` | Paginated queue |
| GET | `/reviews/{review_id}` | Case with history |
| POST | `/reviews/{review_id}/submit` | `{"label": 3, "reviewer": "name", "notes": "..."}` |
| GET | `/feedback` | Verified labels with provenance |
| POST | `/feedback` | Direct feedback submission (auth required when enabled) |

```bash
curl -s -X POST localhost:8000/api/v1/reviews/9c2a.../submit \
  -H "Content-Type: application/json" \
  -d '{"label": 3, "reviewer": "qa.reviewer", "notes": "verified against source"}'
```

A second submission for the same case returns `409`. A blank reviewer returns `400`. A label outside
the class space returns `400`.

### Experiments and reports

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/experiments` | Recorded runs |
| GET | `/experiments/{id}` | Configuration and metrics |
| POST | `/experiments` | Register a run |
| GET | `/reports` | Names of generated reports |
| GET | `/reports/{name}` | Allow-listed report JSON (`benchmarks`, `calibration`, `reference`, `training_lightweight`, `training_advanced`) |

Reports outside the allow-list return `404`; the endpoint cannot read arbitrary files.

## Authentication

Set `api.auth.enabled: true` and export `AETHER_API_TOKEN`. Review submission, feedback and
experiment registration then require `Authorization: Bearer <token>`. Comparison is constant-time.
The dashboard is expected to sit behind the same gateway.
