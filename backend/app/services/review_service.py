"""Database-backed human review queue and feedback recording."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aether.errors import ConflictError, NotFoundError, ValidationFailedError
from aether.feedback.review import ReviewStatus, validate_verified_label
from aether.monitoring.events import review_event
from aether.pipeline import InferenceResult
from aether.utils.logging import get_logger
from backend.app.database.models import HumanFeedbackModel, ReviewCaseModel

logger = get_logger("aether.api.reviews")


def build_review_case_model(result: InferenceResult, *, image_ref: str | None) -> ReviewCaseModel:
    """Create a review case row from a routed prediction."""
    return ReviewCaseModel(
        review_id=str(uuid.uuid4()),
        prediction_id=result.prediction_id,
        status=ReviewStatus.PENDING.value,
        predicted_index=result.predicted_index,
        predicted_label=result.predicted_label,
        confidence=float(result.uncertainty.get("max_probability", 0.0)),
        uncertainty=result.uncertainty,
        drift_severity=str(result.drift.get("severity", "none")),
        diagnostics=[item.get("code", "") for item in result.diagnostics],
        image_ref=image_ref,
        history=[
            {
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": "created",
                "status": ReviewStatus.PENDING.value,
                "action": result.action,
            }
        ],
    )


class ReviewService:
    """Review queue operations with an auditable history."""

    def __init__(self, session: Session, num_classes: int = 10) -> None:
        self.session = session
        self.num_classes = num_classes

    def list(
        self, *, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> tuple[list[ReviewCaseModel], int]:
        statement = select(ReviewCaseModel)
        count_statement = select(func.count()).select_from(ReviewCaseModel)
        if status:
            statement = statement.where(ReviewCaseModel.status == status)
            count_statement = count_statement.where(ReviewCaseModel.status == status)
        statement = (
            statement.order_by(ReviewCaseModel.created_at.desc()).limit(limit).offset(offset)
        )
        cases = list(self.session.scalars(statement))
        total = int(self.session.scalar(count_statement) or 0)
        return cases, total

    def get(self, review_id: str) -> ReviewCaseModel:
        case = self.session.scalar(
            select(ReviewCaseModel).where(ReviewCaseModel.review_id == review_id)
        )
        if case is None:
            raise NotFoundError(f"review case {review_id} not found")
        return case

    def backlog(self) -> int:
        return int(
            self.session.scalar(
                select(func.count())
                .select_from(ReviewCaseModel)
                .where(ReviewCaseModel.status == ReviewStatus.PENDING.value)
            )
            or 0
        )

    def submit(
        self, review_id: str, label: int, reviewer: str, notes: str | None
    ) -> tuple[ReviewCaseModel, HumanFeedbackModel]:
        """Record a verified label and the resulting feedback row atomically."""
        label_value, reviewer_name = validate_verified_label(
            label, self.num_classes, reviewer, notes
        )
        case = self.get(review_id)
        if case.status == ReviewStatus.RESOLVED.value:
            raise ConflictError("review case is already resolved")
        now = datetime.now(UTC)
        history = list(case.history or [])
        history.append(
            {
                "at": now.isoformat(),
                "event": "resolved",
                "status": ReviewStatus.RESOLVED.value,
                "predicted_index": case.predicted_index,
                "verified_label": label_value,
                "reviewer": reviewer_name,
            }
        )
        case.verified_label = label_value
        case.reviewer = reviewer_name
        case.notes = notes
        case.status = ReviewStatus.RESOLVED.value
        case.resolved_at = now
        case.history = history
        feedback = HumanFeedbackModel(
            review_id=case.id,
            label=label_value,
            reviewer=reviewer_name,
            verified=True,
            notes=notes,
        )
        self.session.add(feedback)
        self.session.flush()
        review_event(
            review_id,
            "resolved",
            predicted_index=case.predicted_index,
            verified_label=label_value,
            reviewer=reviewer_name,
        )
        return case, feedback

    def submit_feedback(
        self, review_id: str, label: int, reviewer: str, notes: str | None
    ) -> HumanFeedbackModel:
        """Standalone feedback path; verified labels still require a reviewer identity."""
        return self.submit(review_id, label, reviewer, notes)[1]

    def list_feedback(self, limit: int = 50, offset: int = 0) -> list[HumanFeedbackModel]:
        return list(
            self.session.scalars(
                select(HumanFeedbackModel)
                .order_by(HumanFeedbackModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        )

    def pending_count(self) -> int:
        return self.backlog()


def ensure_notes(notes: str | None) -> str | None:
    if notes is not None and len(notes) > 2000:
        raise ValidationFailedError("notes too long")
    return notes
