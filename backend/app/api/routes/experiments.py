"""Experiment centre endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.app.api.dependencies import SessionDep
from backend.app.api.schemas.models import ExperimentCreate, ExperimentResponse
from backend.app.core.security import require_auth
from backend.app.services.experiment_service import ExperimentService

router = APIRouter(tags=["experiments"])


@router.get("/experiments")
def list_experiments(
    session: SessionDep, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)
) -> dict:
    service = ExperimentService(session)
    rows, total = service.list(limit=limit, offset=offset)
    return {
        "items": [ExperimentResponse.model_validate(row).model_dump(mode="json") for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: str, session: SessionDep) -> ExperimentResponse:
    return ExperimentResponse.model_validate(ExperimentService(session).get(experiment_id))


@router.post(
    "/experiments", response_model=ExperimentResponse, status_code=201,
    dependencies=[Depends(require_auth)],
)
def create_experiment(payload: ExperimentCreate, session: SessionDep) -> ExperimentResponse:
    """Register a completed experiment so the dashboard shows only real measurements."""
    row = ExperimentService(session).upsert(payload, notes=payload.notes)
    session.flush()
    return ExperimentResponse.model_validate(row)
