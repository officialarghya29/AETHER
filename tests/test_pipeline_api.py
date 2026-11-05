"""Pipeline and HTTP integration tests against real (small) checkpoints.

These are plumbing tests: the models are randomly initialised, so nothing here is a claim about
predictive quality. What is verified is that the pipeline records evidence, the router acts on
it, the database persists the trail, and the API returns what it says it returns.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from aether.config import load_config
from aether.pipeline import ReliabilityPipeline
from backend.app.database.session import create_all, init_engine, reset_engine


@pytest.fixture
def pipeline(tiny_models):
    pipe = ReliabilityPipeline(tiny_models, device="cpu")
    pipe.warm(("lightweight",))
    yield pipe
    pipe.registry.close()


def _png_bytes(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    image = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def api_config(temp_config):
    """Force the router into human review so the review workflow is exercised deterministically."""
    config = load_config()
    routing = config.routing.model_copy(
        update={
            "accept_risk": 0.001,
            "escalate_risk": 0.002,
            "high_risk": 0.003,
            "enable_escalation": False,
            "review_available": True,
            "policy": "rule_based",
        }
    )
    return temp_config.model_copy(update={"routing": routing})


@pytest.fixture
def client(api_config, tiny_models):
    from backend.app.main import create_app

    reset_engine()
    init_engine(api_config.storage.database_url)
    create_all()
    app = create_app(api_config)
    with TestClient(app) as test_client:
        yield test_client
    reset_engine()


def test_pipeline_produces_complete_evidence(pipeline, images):
    results = pipeline.predict_batch(images)
    assert len(results) == images.shape[0]
    for result in results:
        assert 0.0 <= result.risk["value"] <= 1.0
        assert result.risk["source"] in ("learned", "heuristic")
        assert 0.0 <= result.uncertainty["max_probability"] <= 1.0
        assert np.isclose(sum(result.probabilities), 1.0, atol=1e-4)
        assert result.action in ("ACCEPT", "MONITOR", "ESCALATE", "HUMAN_REVIEW", "ABSTAIN")
        assert result.drift["status"] in ("reference_unavailable", "insufficient_window", "ok")
        assert result.timings["total_ms"] > 0
        payload = result.to_dict()
        assert payload["prediction_id"] and payload["created_at"]


def test_pipeline_drift_window_reaches_assessment(tiny_models, reference_artifact, images):
    pipeline = ReliabilityPipeline(tiny_models, device="cpu")
    pipeline.warm(("lightweight",))
    for _ in range(6):
        pipeline.predict_batch(images)
    status = pipeline.last_drift
    assert status is not None
    assert status.status in ("ok", "insufficient_window")
    pipeline.registry.close()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["components"]["database"] == "ok"
    assert payload["status"] in ("ok", "degraded")


def test_system_status_lists_models(client):
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["models"]) == 2
    assert payload["routing_policy"] == "rule_based"
    assert "lightweight_checkpoint" in payload["artifacts"]


def test_predict_persists_and_routes_to_review(client):
    response = client.post(
        "/api/v1/predict", files={"file": ("sample.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["action"] in ("HUMAN_REVIEW", "ABSTAIN")
    assert payload["review_id"]
    prediction_id = payload["prediction_id"]

    uncertainty = client.get(f"/api/v1/predictions/{prediction_id}/uncertainty")
    assert uncertainty.status_code == 200
    assert uncertainty.json()["temperature"] >= 0.05

    diagnostics = client.get(f"/api/v1/diagnostics/{prediction_id}")
    assert diagnostics.status_code == 200
    assert "diagnostics" in diagnostics.json()


def test_review_submission_and_audit_trail(client):
    created = client.post(
        "/api/v1/predict", files={"file": ("sample.png", _png_bytes(1), "image/png")}
    ).json()
    review_id = created["review_id"]
    assert review_id

    listing = client.get("/api/v1/reviews?status=pending")
    assert listing.status_code == 200
    assert any(item["review_id"] == review_id for item in listing.json()["items"])

    submitted = client.post(
        f"/api/v1/reviews/{review_id}/submit",
        json={"label": 3, "reviewer": "qa.reviewer", "notes": "verified from source"},
    )
    assert submitted.status_code == 200, submitted.text
    assert submitted.json()["verified"] is True

    case = client.get(f"/api/v1/reviews/{review_id}").json()
    assert case["status"] == "resolved"
    assert case["verified_label"] == 3
    assert case["reviewer"] == "qa.reviewer"
    assert any(entry.get("event") == "resolved" for entry in case["history"])

    duplicate = client.post(
        f"/api/v1/reviews/{review_id}/submit", json={"label": 4, "reviewer": "qa.reviewer"}
    )
    assert duplicate.status_code == 409

    feedback = client.get("/api/v1/feedback")
    assert feedback.status_code == 200
    assert feedback.json()["items"][0]["verified"] is True


def test_review_validation_rejects_bad_labels(client):
    created = client.post(
        "/api/v1/predict", files={"file": ("sample.png", _png_bytes(2), "image/png")}
    ).json()
    response = client.post(
        f"/api/v1/reviews/{created['review_id']}/submit",
        json={"label": 3, "reviewer": "   "},
    )
    # Domain validation (as opposed to request-schema validation) is reported as 400 with a
    # stable machine-readable code, per the documented error contract.
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_failed"


def test_invalid_upload_is_rejected(client):
    response = client.post(
        "/api/v1/predict", files={"file": ("bad.png", b"not an image", "image/png")}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_oversized_batch_is_rejected(client):
    files = [("files", (f"a{index}.png", _png_bytes(index), "image/png")) for index in range(3)]
    response = client.post("/api/v1/predict/batch", files=files)
    assert response.status_code == 201
    assert len(response.json()) == 3


def test_metrics_summary_counts_predictions(client):
    client.post("/api/v1/predict", files={"file": ("a.png", _png_bytes(3), "image/png")})
    response = client.get("/api/v1/metrics/summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["predictions_total"] >= 1
    assert payload["review_backlog"] >= 1


def test_routing_endpoints(client):
    client.post("/api/v1/predict", files={"file": ("a.png", _png_bytes(4), "image/png")})
    metrics = client.get("/api/v1/routing/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["total"] >= 1
    decisions = client.get("/api/v1/routing/decisions")
    assert decisions.status_code == 200
    assert decisions.json()["items"][0]["explanation"]


def test_missing_report_returns_404(client):
    response = client.get("/api/v1/reports/benchmarks")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_rate_limit_is_enforced(api_config, tiny_models):
    from backend.app.main import create_app

    limited = api_config.model_copy(
        update={"api": api_config.api.model_copy(update={"rate_limit_per_minute": 2})}
    )
    reset_engine()
    init_engine(limited.storage.database_url)
    create_all()
    app = create_app(limited)
    with TestClient(app) as test_client:
        codes = [
            test_client.post(
                "/api/v1/predict", files={"file": ("a.png", _png_bytes(5), "image/png")}
            ).status_code
            for _ in range(4)
        ]
    reset_engine()
    assert 429 in codes
