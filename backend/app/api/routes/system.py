"""Health, system status and model inventory endpoints."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from aether.runtime import resolve_device
from backend.app.api.dependencies import ConfigDep, PipelineDep, SessionDep
from backend.app.api.schemas.models import HealthResponse, ModelInfo, SystemStatus

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(config: ConfigDep, session: SessionDep) -> HealthResponse:
    """Liveness plus dependency readiness. Returns ``degraded`` when artifacts are missing."""
    components: dict[str, str] = {}
    try:
        session.execute(text("SELECT 1"))
        components["database"] = "ok"
    except Exception:
        components["database"] = "error"
    artifacts = _artifact_status(config)
    components["models"] = "ok" if artifacts.get("lightweight_checkpoint") else "missing"
    components["calibration"] = "ok" if artifacts.get("calibration") else "missing"
    components["reference"] = "ok" if artifacts.get("reference") else "missing"
    overall = "ok"
    if components["database"] != "ok":
        overall = "error"
    elif any(value == "missing" for value in components.values()):
        overall = "degraded"
    return HealthResponse(
        status=overall,
        version=config.application.version,
        environment=config.application.environment,
        components=components,
        timestamp=datetime.now(UTC),
    )


@router.get("/system/status", response_model=SystemStatus)
def system_status(
    config: ConfigDep, pipeline: PipelineDep, session: SessionDep
) -> SystemStatus:
    """Full runtime description for the dashboard."""
    import torch

    device = resolve_device(config.runtime.device)
    models = pipeline.registry.describe()
    return SystemStatus(
        application=config.application.name,
        version=config.application.version,
        environment=config.application.environment,
        device=str(device),
        torch_version=torch.__version__,
        artifacts=pipeline.artifact_status,
        routing_policy=config.routing.policy,
        drift_window_size=config.drift.evaluation_window_size,
        uptime_seconds=round(time.time() - _started_at(), 1),
        models=[ModelInfo(**entry) for entry in models],
    )


@router.get("/models", response_model=list[ModelInfo])
def list_models(config: ConfigDep, pipeline: PipelineDep) -> list[ModelInfo]:
    """Describe configured models without loading them."""
    return [ModelInfo(**entry) for entry in pipeline.registry.describe()]


@router.get("/models/{model_id}", response_model=ModelInfo)
def get_model(model_id: str, config: ConfigDep, pipeline: PipelineDep) -> ModelInfo:
    """Describe a single model by role, architecture or full model id."""
    for entry in pipeline.registry.describe():
        if model_id in {entry.get("role"), entry.get("architecture"), entry.get("model_id")}:
            return ModelInfo(**entry)
    raise HTTPException(status_code=404, detail=f"model '{model_id}' is not configured")


def _artifact_status(config) -> dict[str, bool]:
    return {
        "lightweight_checkpoint": config.resolve(config.models.lightweight.checkpoint).exists(),
        "advanced_checkpoint": config.resolve(config.models.advanced.checkpoint).exists(),
        "calibration": config.resolve(config.uncertainty.temperature_artifact).exists(),
        "reference": config.resolve(config.drift.reference_artifact).exists(),
        "risk_model": config.resolve(config.routing.learned_policy.artifact).exists(),
    }


def _started_at() -> float:
    import backend.app.main as main_module

    return main_module.STARTED_AT
