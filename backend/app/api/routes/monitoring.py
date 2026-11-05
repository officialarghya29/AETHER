"""Drift, routing and metrics endpoints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aether.monitoring.metrics import METRICS
from backend.app.api.dependencies import PipelineDep, SessionDep
from backend.app.api.schemas.models import (
    DriftEventResponse,
    DriftStatusResponse,
    MetricsSummary,
    RoutingMetrics,
)
from backend.app.database.models import (
    DriftEventModel,
    PredictionModel,
    ReviewCaseModel,
    RoutingDecisionModel,
)

router = APIRouter(tags=["monitoring"])


def _budget_and_tta_counts(session: Session) -> tuple[int, int, int]:
    """Budgeted traffic, budget misses and paid mirror passes across stored decisions."""
    budgeted = 0
    misses = 0
    tta_calls = 0
    for decision in session.scalars(select(PredictionModel.decision)):
        if not decision:
            continue
        budget = decision.get("risk_budget")
        if budget:
            budgeted += 1
            misses += 0 if budget.get("meets_threshold") else 1
        if decision.get("test_time_augmentation"):
            tta_calls += 1
    return budgeted, misses, tta_calls


@router.get("/drift/status", response_model=DriftStatusResponse)
def drift_status(pipeline: PipelineDep) -> DriftStatusResponse:
    """Current drift assessment for the live request window."""
    report = getattr(pipeline, "last_drift", None)
    if report is None:
        return DriftStatusResponse(
            severity="none",
            score=0.0,
            window_size=len(pipeline.window),
            status="insufficient_window",
            warnings=["no drift assessment has been produced yet"],
            interpretation=(
                "Shift is measured without labels. Accuracy impact cannot be inferred from "
                "unlabelled traffic and is not reported here."
            ),
        )
    payload = report.to_dict()
    return DriftStatusResponse(**payload)


@router.get("/drift/events", response_model=list[DriftEventResponse])
def drift_events(
    session: SessionDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    severity: str | None = Query(None),
) -> list[DriftEventResponse]:
    statement = select(DriftEventModel)
    if severity:
        statement = statement.where(DriftEventModel.severity == severity)
    rows = list(
        session.scalars(
            statement.order_by(DriftEventModel.created_at.desc()).limit(limit).offset(offset)
        )
    )
    return [DriftEventResponse.model_validate(row) for row in rows]


@router.get("/routing/decisions")
def routing_decisions(
    session: SessionDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """Recent routing decisions with their explanations."""
    rows = list(
        session.execute(
            select(RoutingDecisionModel, PredictionModel.decision)
            .join(PredictionModel, PredictionModel.id == RoutingDecisionModel.prediction_id)
            .order_by(RoutingDecisionModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    )
    total = int(session.scalar(select(func.count()).select_from(RoutingDecisionModel)) or 0)
    return {
        "items": [
            {
                "id": row.id,
                "prediction_id": row.prediction_id,
                "action": row.action,
                "policy": row.policy,
                "risk": row.risk,
                "explanation": row.explanation,
                "reason_codes": row.reason_codes,
                "utilities": row.utilities,
                "test_time_augmentation": bool(
                    (prediction_decision or {}).get("test_time_augmentation", False)
                ),
                "risk_budget": (prediction_decision or {}).get("risk_budget"),
                "created_at": row.created_at.isoformat(),
            }
            for row, prediction_decision in rows
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/routing/metrics", response_model=RoutingMetrics)
def routing_metrics(session: SessionDep) -> RoutingMetrics:
    """Aggregate routing behaviour over all recorded decisions."""
    total = int(session.scalar(select(func.count()).select_from(RoutingDecisionModel)) or 0)
    if total == 0:
        return RoutingMetrics(
            total=0,
            action_distribution={},
            mean_risk=0.0,
            mean_latency_ms=0.0,
            escalation_rate=0.0,
            abstention_rate=0.0,
            review_rate=0.0,
        )
    distribution = {
        action: int(count)
        for action, count in session.execute(
            select(RoutingDecisionModel.action, func.count()).group_by(RoutingDecisionModel.action)
        )
    }
    mean_risk = float(session.scalar(select(func.avg(RoutingDecisionModel.risk))) or 0.0)
    mean_latency = float(session.scalar(select(func.avg(PredictionModel.latency_ms))) or 0.0)
    budgeted, misses, tta_calls = _budget_and_tta_counts(session)
    return RoutingMetrics(
        total=total,
        action_distribution=distribution,
        mean_risk=round(mean_risk, 4),
        mean_latency_ms=round(mean_latency, 3),
        escalation_rate=round(distribution.get("ESCALATE", 0) / total, 4),
        abstention_rate=round(distribution.get("ABSTAIN", 0) / total, 4),
        review_rate=round(distribution.get("HUMAN_REVIEW", 0) / total, 4),
        budgeted_total=budgeted,
        budget_misses=misses,
        budget_miss_rate=round(misses / budgeted, 4) if budgeted else 0.0,
        tta_calls=tta_calls,
    )


@router.get("/metrics/summary", response_model=MetricsSummary)
def metrics_summary(session: SessionDep) -> MetricsSummary:
    """Operational summary combining in-process metrics and persisted counts."""
    snapshot = METRICS.snapshot()
    backlog = int(
        session.scalar(
            select(func.count())
            .select_from(ReviewCaseModel)
            .where(ReviewCaseModel.status == "pending")
        )
        or 0
    )
    since = datetime.now(UTC) - timedelta(hours=24)
    drift_events = int(
        session.scalar(
            select(func.count())
            .select_from(DriftEventModel)
            .where(DriftEventModel.created_at >= since)
        )
        or 0
    )
    predictions_total = int(session.scalar(select(func.count()).select_from(PredictionModel)) or 0)
    budgeted, misses, _ = _budget_and_tta_counts(session)
    return MetricsSummary(
        uptime_seconds=snapshot["uptime_seconds"],
        counters=snapshot["counters"],
        gauges=snapshot["gauges"],
        latency=snapshot["latency"],
        review_backlog=backlog,
        drift_events_24h=drift_events,
        predictions_total=predictions_total,
        budgeted_total=budgeted,
        budget_misses=misses,
        budget_miss_rate=round(misses / budgeted, 4) if budgeted else 0.0,
    )
