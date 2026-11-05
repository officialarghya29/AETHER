"""Inference endpoints: single, batch, uncertainty and diagnostics."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, File, Query, UploadFile
from sqlalchemy import func, select

from aether.errors import InvalidInputError, NotFoundError
from aether.imaging import centre_crop_square, decode_image
from aether.monitoring.metrics import METRICS
from aether.utils.logging import get_logger, request_id_var
from backend.app.api.dependencies import (
    ConfigDep,
    PipelineDep,
    RateLimitDep,
    SessionDep,
)
from backend.app.api.schemas.models import (
    PredictionListResponse,
    PredictionResponse,
    UncertaintyResponse,
)
from backend.app.database.models import (
    DiagnosticRecordModel,
    PredictionModel,
    UncertaintyRecordModel,
)
from backend.app.services.prediction_service import PredictionService, prediction_to_dict

router = APIRouter(tags=["predictions"])
logger = get_logger("aether.api.predictions")

MAX_BATCH_FILES = 16


def _decode_uploads(files: list[UploadFile], config, require_square: bool = False) -> tuple[np.ndarray, list[bytes]]:
    images: list[np.ndarray] = []
    payloads: list[bytes] = []
    for upload in files:
        payload = upload.file.read(config.storage.upload_max_bytes + 1)
        decoded = decode_image(
            payload,
            max_bytes=config.storage.upload_max_bytes,
            max_pixels=config.storage.upload_max_pixels,
        )
        pixels = decoded.pixels
        if require_square and pixels.shape[0] != pixels.shape[1]:
            raise InvalidInputError("image must be square")
        images.append(centre_crop_square(pixels))
        payloads.append(payload)
    shapes = {image.shape[:2] for image in images}
    if len(shapes) > 1:
        raise InvalidInputError("all images in a batch must share the same dimensions")
    return np.stack(images).astype(np.uint8), payloads


def _to_response(result, review_id: str | None) -> PredictionResponse:
    payload = result.to_dict()
    payload["review_id"] = review_id
    return PredictionResponse(**payload)


@router.post("/predict", response_model=PredictionResponse, status_code=201)
def predict(
    config: ConfigDep,
    pipeline: PipelineDep,
    session: SessionDep,
    _rate: RateLimitDep,
    file: UploadFile = File(..., description="PNG/JPEG/WebP/BMP image"),
    error_budget: float | None = Query(
        None,
        gt=0,
        lt=1,
        description="Optional selective-error budget; returns the calibrated acceptance threshold",
    ),
) -> PredictionResponse:
    """Run the full reliability pipeline for a single image."""
    images, payloads = _decode_uploads([file], config)
    service = PredictionService(session, pipeline, config)
    results = service.run(
        images,
        request_id=request_id_var.get(),
        raw_payloads=payloads,
        error_budget=error_budget,
    )
    result, review_id = results[0]
    return _to_response(result, review_id)


@router.post("/predict/batch", response_model=list[PredictionResponse], status_code=201)
def predict_batch(
    config: ConfigDep,
    pipeline: PipelineDep,
    session: SessionDep,
    _rate: RateLimitDep,
    files: list[UploadFile] = File(...),
    error_budget: float | None = Query(None, gt=0, lt=1),
) -> list[PredictionResponse]:
    """Run the pipeline over a batch of images."""
    if not files:
        raise InvalidInputError("at least one file is required")
    if len(files) > MAX_BATCH_FILES:
        raise InvalidInputError(
            f"batch size {len(files)} exceeds the limit of {MAX_BATCH_FILES} files"
        )
    images, payloads = _decode_uploads(files, config)
    service = PredictionService(session, pipeline, config)
    results = service.run(
        images,
        request_id=request_id_var.get(),
        raw_payloads=payloads,
        error_budget=error_budget,
    )
    METRICS.increment("batch_predictions", len(results))
    return [_to_response(result, review_id) for result, review_id in results]


@router.get("/predictions", response_model=PredictionListResponse)
def list_predictions(
    session: SessionDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    action: str | None = Query(None),
) -> PredictionListResponse:
    """Paginated prediction history for the dashboard."""
    statement = select(PredictionModel)
    count_statement = select(func.count()).select_from(PredictionModel)
    if action:
        statement = statement.where(PredictionModel.action == action.upper())
        count_statement = count_statement.where(PredictionModel.action == action.upper())
    rows = list(
        session.scalars(
            statement.order_by(PredictionModel.created_at.desc()).limit(limit).offset(offset)
        )
    )
    total = int(session.scalar(count_statement) or 0)
    return PredictionListResponse(
        items=[prediction_to_dict(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/predictions/{prediction_id}/uncertainty", response_model=UncertaintyResponse)
def get_uncertainty(prediction_id: str, session: SessionDep) -> UncertaintyResponse:
    """Uncertainty evidence recorded for a prediction."""
    prediction = session.scalar(
        select(PredictionModel).where(PredictionModel.id == prediction_id)
    )
    if prediction is None:
        raise NotFoundError(f"prediction {prediction_id} not found")
    record = session.scalar(
        select(UncertaintyRecordModel).where(
            UncertaintyRecordModel.prediction_id == prediction_id
        )
    )
    if record is None:
        raise NotFoundError(f"no uncertainty record for prediction {prediction_id}")
    return UncertaintyResponse(
        prediction_id=prediction_id,
        uncertainty={
            "predictive_entropy": record.predictive_entropy,
            "max_probability": record.max_probability,
            "prediction_margin": record.prediction_margin,
        },
        calibrated_uncertainty=prediction.uncertainty,
        temperature=record.temperature,
        risk=prediction.risk,
    )


@router.get("/diagnostics/{prediction_id}")
def get_diagnostics(prediction_id: str, session: SessionDep) -> dict:
    """Structured diagnostics for a prediction."""
    prediction = session.scalar(
        select(PredictionModel).where(PredictionModel.id == prediction_id)
    )
    if prediction is None:
        raise NotFoundError(f"prediction {prediction_id} not found")
    rows = list(
        session.scalars(
            select(DiagnosticRecordModel)
            .where(DiagnosticRecordModel.prediction_id == prediction_id)
            .order_by(DiagnosticRecordModel.severity)
        )
    )
    return {
        "prediction_id": prediction_id,
        "action": prediction.action,
        "predicted_label": prediction.predicted_label,
        "diagnostics": [
            {
                "code": row.code,
                "severity": row.severity,
                "evidence": row.evidence,
                "explanation": row.explanation,
                "recommended_action": row.recommended_action,
            }
            for row in rows
        ],
    }
