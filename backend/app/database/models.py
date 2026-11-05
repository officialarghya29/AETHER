"""SQLAlchemy 2.0 persistence models.

Logical entities mirror the specification one-to-one. Images are never stored in the database:
only a reference to a controlled local artifact is kept.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base."""


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )


class ModelVersionModel(Base, TimestampMixin):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    role: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    architecture: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="candidate")
    checkpoint: Mapped[str | None] = mapped_column(String(512))
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    approved_by: Mapped[str | None] = mapped_column(String(128))
    notes: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("ix_model_role_version", "role", "version", unique=True),)


class PredictionModel(Base, TimestampMixin):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    action: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    predicted_index: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(64), nullable=False)
    probabilities: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    input_size: Mapped[int] = mapped_column(Integer, nullable=False)
    drift_severity: Mapped[str] = mapped_column(String(16), nullable=False, default="none", index=True)
    risk: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    uncertainty: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    decision: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    image_ref: Mapped[str | None] = mapped_column(String(512))
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)

    uncertainty_records: Mapped[list[UncertaintyRecordModel]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan"
    )
    routing_decision: Mapped[RoutingDecisionModel | None] = relationship(
        back_populates="prediction", cascade="all, delete-orphan", uselist=False
    )
    diagnostics: Mapped[list[DiagnosticRecordModel]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan"
    )
    review_case: Mapped[ReviewCaseModel | None] = relationship(
        back_populates="prediction", cascade="all, delete-orphan", uselist=False
    )


class UncertaintyRecordModel(Base, TimestampMixin):
    __tablename__ = "uncertainty_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    prediction_id: Mapped[str] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    predictive_entropy: Mapped[float] = mapped_column(Float, nullable=False)
    max_probability: Mapped[float] = mapped_column(Float, nullable=False)
    prediction_margin: Mapped[float] = mapped_column(Float, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    calibrated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    prediction: Mapped[PredictionModel] = relationship(back_populates="uncertainty_records")


class DriftEventModel(Base, TimestampMixin):
    __tablename__ = "drift_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    window_size: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ok")
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    model_id: Mapped[str | None] = mapped_column(String(128), index=True)


class RoutingDecisionModel(Base, TimestampMixin):
    __tablename__ = "routing_decisions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    prediction_id: Mapped[str] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    action: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    policy: Mapped[str] = mapped_column(String(32), nullable=False)
    risk: Mapped[float] = mapped_column(Float, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False, default="")
    reason_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    utilities: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)

    prediction: Mapped[PredictionModel] = relationship(back_populates="routing_decision")


class DiagnosticRecordModel(Base, TimestampMixin):
    __tablename__ = "diagnostic_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    prediction_id: Mapped[str] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    explanation: Mapped[str] = mapped_column(Text, nullable=False, default="")
    recommended_action: Mapped[str] = mapped_column(Text, nullable=False, default="")

    prediction: Mapped[PredictionModel] = relationship(back_populates="diagnostics")


class ReviewCaseModel(Base, TimestampMixin):
    __tablename__ = "review_cases"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    review_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    prediction_id: Mapped[str] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    predicted_index: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    uncertainty: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    drift_severity: Mapped[str] = mapped_column(String(16), nullable=False, default="none")
    diagnostics: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    image_ref: Mapped[str | None] = mapped_column(String(512))
    verified_label: Mapped[int | None] = mapped_column(Integer)
    reviewer: Mapped[str | None] = mapped_column(String(128))
    notes: Mapped[str | None] = mapped_column(Text)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    history: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)

    prediction: Mapped[PredictionModel] = relationship(back_populates="review_case")
    feedback: Mapped[list[HumanFeedbackModel]] = relationship(
        back_populates="review", cascade="all, delete-orphan"
    )


class HumanFeedbackModel(Base, TimestampMixin):
    __tablename__ = "human_feedback"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    review_id: Mapped[str] = mapped_column(
        ForeignKey("review_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[int] = mapped_column(Integer, nullable=False)
    reviewer: Mapped[str] = mapped_column(String(128), nullable=False)
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    notes: Mapped[str | None] = mapped_column(Text)
    used_in_dataset: Mapped[str | None] = mapped_column(String(64))

    review: Mapped[ReviewCaseModel] = relationship(back_populates="feedback")


class ExperimentModel(Base, TimestampMixin):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="completed")
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    dataset_id: Mapped[str | None] = mapped_column(String(64))
    seed: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text)


class AuditEventModel(Base, TimestampMixin):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    event: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="info")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    actor: Mapped[str | None] = mapped_column(String(128))


ALL_MODELS = (
    ModelVersionModel,
    PredictionModel,
    UncertaintyRecordModel,
    DriftEventModel,
    RoutingDecisionModel,
    DiagnosticRecordModel,
    ReviewCaseModel,
    HumanFeedbackModel,
    ExperimentModel,
    AuditEventModel,
)
