# Deployment — AETHER-ML

## 1. Prerequisites

- Python 3.11 or 3.12, Node 20+, Docker 24+ with Compose v2 (for the container path).
- ~2 GB disk for CIFAR-10 plus artifacts; ~1 GB RAM for CPU inference.
- No GPU required. CUDA is used only when `runtime.device: cuda` and CUDA is present.

## 2. Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make data            # CIFAR-10 -> data/raw/cifar10.npz (fingerprinted)
make train           # checkpoints -> artifacts/checkpoints
make reference       # drift reference -> artifacts/reference
make calibrate       # temperatures + risk models -> artifacts/calibration
make benchmarks      # reports -> artifacts/reports

make api             # http://localhost:8000  (docs at /docs)
make frontend        # http://localhost:5173
```

SQLite is the default database and is created automatically outside production. Migrations:

```bash
python -m alembic upgrade head
```

## 3. Container stack

```bash
cp .env.example .env          # optional: set POSTGRES_PASSWORD, AETHER_API_TOKEN
docker compose up --build -d
```

| Service | URL | Notes |
| --- | --- | --- |
| dashboard | http://localhost:8080 | nginx, proxies `/api` and `/health` |
| API | http://localhost:8000 | FastAPI, `/docs` |
| database | internal | PostgreSQL 16, named volume |

The backend applies `alembic upgrade head` on start and runs as uid 10001. Artifacts are mounted
from `./artifacts`; the process needs read access to checkpoints, calibration, reference and reports,
and write access to `artifacts/uploads` and `artifacts/db`.

**The container does not train.** Produce artifacts first (`make data train calibrate reference`),
or mount a directory that already contains them.

## 4. Configuration

Environment variables override YAML:

```bash
AETHER_ENV=production
DATABASE_URL=postgresql+psycopg://aether:secret@db:5432/aether
AETHER_API_TOKEN=<generated>
AETHER_CORS_ORIGINS=https://aether.example.com
```

Generate a token with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

## 5. Production checklist

1. `application.environment: production` (disables automatic `create_all`; migrations become the only
   schema path).
2. `api.auth.enabled: true` with `AETHER_API_TOKEN` set before exposing review or feedback routes.
3. `api.cors_origins` restricted to the dashboard origin.
4. TLS terminated at a reverse proxy or ingress in front of the dashboard.
5. Database backups and a tested restore procedure.
6. Artifact directory on durable storage, versioned alongside model releases.
7. Log aggregation for `routing.decision`, `prediction.completed` and `review.*` events.
8. Health checks wired to the orchestrator; investigate `degraded` states rather than restarting blindly.

## 6. Health and observability

```bash
curl -s localhost:8000/health | jq
curl -s localhost:8000/api/v1/system/status | jq
curl -s localhost:8000/api/v1/metrics/summary | jq
```

`/health` returns `degraded` when an artifact is missing or the database is unreachable, with the
component detail. Metrics include prediction totals, action counts, review backlog, latency
observations and the last drift score.

## 7. Scaling notes

- The service is stateless apart from SQLite; with PostgreSQL it scales horizontally behind a proxy.
- Each replica loads its own copy of the models. On CPU this is ~100 MB for ResNet50; plan memory per
  replica accordingly.
- Drift windows are per-process. With multiple replicas each holds a partial window, so either
  evaluate drift in a dedicated replica or feed windows from a shared store. This is a documented
  limitation of v1.0.0, not a solved problem.
- Rate limiting is per process, not global.

## 8. Rollback

Application rollback is an image/tag change. Model rollback is a registry operation and does not
require a restart of the training pipeline:

```python
from aether.recovery.registry import ModelRegistryStore
from aether.recovery.rollback import rollback_model

store = ModelRegistryStore("artifacts/registry/models.json")
rollback_model(store, role="lightweight", actor="oncall")
```

Restart the API after a checkpoint change so adapters reload, and confirm `/api/v1/system/status`
reports the expected version.

## 9. Troubleshooting

| Symptom | Cause | Remedy |
| --- | --- | --- |
| `/health` reports `degraded` | checkpoint or database missing | run the artifact stages, check `DATABASE_URL` |
| Predictions are slow on first call | lazy advanced-model load | warm it at startup or accept the first-call cost |
| Drift always `reference_unavailable` | reference artifact absent | `make reference` |
| All confidences look uncalibrated | temperature artifact absent | `make calibrate` |
| `429` under light load | low `rate_limit_per_minute` | raise it or run multiple replicas |
| `503 model_unavailable` | checkpoint path wrong | verify `models.*.checkpoint` and permissions |
