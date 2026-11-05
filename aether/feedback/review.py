"""Human review domain logic.

A review case is created only when the router cannot answer reliably. A human supplies a
verified label, which is recorded with reviewer identity and timestamp. Unverified or rejected
feedback never reaches training data; :mod:`aether.feedback.dataset_builder` enforces that at
the boundary.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from aether.errors import ConflictError, NotFoundError, ValidationFailedError


class ReviewStatus(StrEnum):
    """Lifecycle of a review case."""

    PENDING = "pending"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ReviewCase:
    """A prediction awaiting verified human feedback."""

    review_id: str
    prediction_id: str
    model_id: str
    predicted_index: int
    predicted_label: str
    confidence: float
    uncertainty: dict[str, float]
    drift_severity: str
    diagnostics: list[str]
    status: str = ReviewStatus.PENDING.value
    image_ref: str | None = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    resolved_at: str | None = None
    verified_label: int | None = None
    reviewer: str | None = None
    notes: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_verified_label(
    label: Any, num_classes: int, reviewer: str, notes: str | None = None
) -> tuple[int, str]:
    """Validate a submitted label and reviewer identity."""
    if not reviewer or not reviewer.strip():
        raise ValidationFailedError("reviewer identity is required for a verified label")
    if len(reviewer) > 128:
        raise ValidationFailedError("reviewer identity is too long")
    try:
        value = int(label)
    except (TypeError, ValueError) as exc:
        raise ValidationFailedError("label must be an integer class index") from exc
    if value < 0 or value >= num_classes:
        raise ValidationFailedError(
            "label is outside the model's class space",
            details={"label": value, "num_classes": num_classes},
        )
    if notes is not None and len(notes) > 2000:
        raise ValidationFailedError("review notes are too long")
    return value, reviewer.strip()


class InMemoryReviewQueue:
    """Reference review queue used by tests and offline workflows.

    The FastAPI backend uses the database-backed implementation with the same semantics.
    """

    def __init__(self, num_classes: int = 10) -> None:
        self.num_classes = num_classes
        self._cases: dict[str, ReviewCase] = {}
        self._lock = threading.Lock()

    def create(self, case: ReviewCase) -> ReviewCase:
        with self._lock:
            if case.review_id in self._cases:
                raise ConflictError(f"review case {case.review_id} already exists")
            case.history.append(
                {"at": case.created_at, "event": "created", "status": case.status}
            )
            self._cases[case.review_id] = case
            return case

    def get(self, review_id: str) -> ReviewCase:
        with self._lock:
            case = self._cases.get(review_id)
        if case is None:
            raise NotFoundError(f"review case {review_id} not found")
        return case

    def list(self, status: str | None = None, limit: int = 50, offset: int = 0) -> list[ReviewCase]:
        with self._lock:
            cases = sorted(self._cases.values(), key=lambda item: item.created_at, reverse=True)
        if status:
            cases = [case for case in cases if case.status == status]
        return cases[offset : offset + limit]

    def count(self, status: str | None = None) -> int:
        with self._lock:
            if status is None:
                return len(self._cases)
            return sum(1 for case in self._cases.values() if case.status == status)

    def submit(
        self, review_id: str, label: Any, reviewer: str, notes: str | None = None
    ) -> ReviewCase:
        """Record a verified label and resolve the case."""
        label_value, reviewer_name = validate_verified_label(
            label, self.num_classes, reviewer, notes
        )
        with self._lock:
            case = self._cases.get(review_id)
            if case is None:
                raise NotFoundError(f"review case {review_id} not found")
            if case.status == ReviewStatus.RESOLVED.value:
                raise ConflictError("review case is already resolved")
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            previous = case.predicted_index
            case.verified_label = label_value
            case.reviewer = reviewer_name
            case.notes = notes
            case.status = ReviewStatus.RESOLVED.value
            case.resolved_at = now
            case.history.append(
                {
                    "at": now,
                    "event": "resolved",
                    "status": case.status,
                    "predicted_index": previous,
                    "verified_label": label_value,
                    "reviewer": reviewer_name,
                }
            )
            return case

    def reject(self, review_id: str, reviewer: str, reason: str) -> ReviewCase:
        with self._lock:
            case = self._cases.get(review_id)
            if case is None:
                raise NotFoundError(f"review case {review_id} not found")
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            case.status = ReviewStatus.REJECTED.value
            case.reviewer = reviewer
            case.notes = reason
            case.resolved_at = now
            case.history.append(
                {"at": now, "event": "rejected", "reason": reason, "reviewer": reviewer}
            )
            return case


def build_case(
    *,
    prediction_id: str,
    model_id: str,
    predicted_index: int,
    predicted_label: str,
    confidence: float,
    uncertainty: dict[str, float],
    drift_severity: str,
    diagnostics: list[str],
    image_ref: str | None = None,
) -> ReviewCase:
    """Construct a review case from a routed prediction."""
    return ReviewCase(
        review_id=str(uuid.uuid4()),
        prediction_id=prediction_id,
        model_id=model_id,
        predicted_index=int(predicted_index),
        predicted_label=predicted_label,
        confidence=float(confidence),
        uncertainty={key: float(value) for key, value in uncertainty.items()},
        drift_severity=drift_severity,
        diagnostics=list(diagnostics),
        image_ref=image_ref,
    )
