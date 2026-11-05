"""Human review and feedback endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from aether.errors import ValidationFailedError
from backend.app.api.dependencies import SessionDep
from backend.app.api.schemas.models import (
    FeedbackRequest,
    FeedbackResponse,
    ReviewCaseResponse,
    ReviewSubmitRequest,
)
from backend.app.core.security import require_auth
from backend.app.services.review_service import ReviewService

router = APIRouter(tags=["reviews"])


def _case_payload(case) -> ReviewCaseResponse:
    return ReviewCaseResponse(
        review_id=case.review_id,
        prediction_id=case.prediction_id,
        status=case.status,
        predicted_index=case.predicted_index,
        predicted_label=case.predicted_label,
        confidence=case.confidence,
        uncertainty=case.uncertainty,
        drift_severity=case.drift_severity,
        diagnostics=case.diagnostics,
        created_at=case.created_at,
        resolved_at=case.resolved_at,
        verified_label=case.verified_label,
        reviewer=case.reviewer,
        notes=case.notes,
        history=case.history or [],
    )


@router.get("/reviews")
def list_reviews(
    session: SessionDep,
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """Paginated review queue."""
    service = ReviewService(session)
    cases, total = service.list(status=status, limit=limit, offset=offset)
    return {
        "items": [_case_payload(case).model_dump(mode="json") for case in cases],
        "total": total,
        "limit": limit,
        "offset": offset,
        "pending": service.backlog(),
    }


@router.get("/reviews/{review_id}", response_model=ReviewCaseResponse)
def get_review(review_id: str, session: SessionDep) -> ReviewCaseResponse:
    return _case_payload(ReviewService(session).get(review_id))


@router.post(
    "/reviews/{review_id}/submit",
    response_model=FeedbackResponse,
    dependencies=[Depends(require_auth)],
)
def submit_review(
    review_id: str, payload: ReviewSubmitRequest, session: SessionDep
) -> FeedbackResponse:
    """Record a verified label. Requires authentication when enabled."""
    service = ReviewService(session)
    case, feedback = service.submit(review_id, payload.label, payload.reviewer, payload.notes)
    session.flush()
    return FeedbackResponse(
        feedback_id=feedback.id,
        review_id=case.review_id,
        label=feedback.label,
        reviewer=feedback.reviewer,
        verified=feedback.verified,
        created_at=feedback.created_at,
    )


@router.get("/feedback")
def list_feedback(
    session: SessionDep, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)
) -> dict:
    service = ReviewService(session)
    rows = service.list_feedback(limit=limit, offset=offset)
    return {
        "items": [
            {
                "feedback_id": row.id,
                "review_id": row.review_id,
                "label": row.label,
                "reviewer": row.reviewer,
                "verified": row.verified,
                "notes": row.notes,
                "created_at": row.created_at.isoformat(),
                "used_in_dataset": row.used_in_dataset,
            }
            for row in rows
        ],
        "limit": limit,
        "offset": offset,
    }


@router.post("/feedback", response_model=FeedbackResponse, dependencies=[Depends(require_auth)])
def create_feedback(payload: FeedbackRequest, session: SessionDep) -> FeedbackResponse:
    """Submit verified feedback for an existing review case."""
    service = ReviewService(session)
    try:
        feedback = service.submit_feedback(
            payload.review_id, payload.label, payload.reviewer, payload.notes
        )
    except Exception as exc:
        if isinstance(exc, ValidationFailedError):
            raise
        raise
    case = service.get(payload.review_id)
    session.flush()
    return FeedbackResponse(
        feedback_id=feedback.id,
        review_id=case.review_id,
        label=feedback.label,
        reviewer=feedback.reviewer,
        verified=feedback.verified,
        created_at=feedback.created_at,
    )
