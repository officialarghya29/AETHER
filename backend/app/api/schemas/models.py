"""Request and response schemas (Pydantic v2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    version: str
    environment: str
    components: dict[str, str]
    timestamp: datetime


class ModelInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    role: str
    architecture: str
    version: str
    num_classes: int
    checkpoint_present: bool
    loaded: bool
    model_id: str | None = None
    parameters: int | None = None
    device: str | None = None
    val_accuracy: float | None = None


class SystemStatus(BaseModel):
    application: str
    version: str
    environment: str
    device: str
    torch_version: str
    artifacts: dict[str, bool]
    routing_policy: str
    drift_window_size: int
    uptime_seconds: float
    models: list[ModelInfo]


class UncertaintyPayload(BaseModel):
    predictive_entropy: float
    max_probability: float
    prediction_margin: float


class RiskPayload(BaseModel):
    value: float
    source: str
    lightweight: float
    advanced: float | None = None


class DecisionPayload(BaseModel):
    action: str
    policy: str
    risk: float
    reason_codes: list[str] = Field(default_factory=list)
    explanation: str = ""
    utilities: list[dict[str, Any]] = Field(default_factory=list)


class DiagnosticPayload(BaseModel):
    code: str
    severity: str
    title: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    explanation: str = ""
    recommended_action: str = ""


class PredictionResponse(BaseModel):
    prediction_id: str
    action: str
    predicted_index: int
    predicted_label: str
    probabilities: list[float]
    model_id: str
    model_version: str
    uncertainty: dict[str, float]
    calibrated_uncertainty: dict[str, float]
    temperature: float
    risk: RiskPayload
    drift: dict[str, Any]
    decision: DecisionPayload | dict[str, Any]
    diagnostics: list[DiagnosticPayload | dict[str, Any]]
    quality: dict[str, Any]
    lightweight: dict[str, Any]
    advanced: dict[str, Any] | None = None
    timings: dict[str, float]
    input_size: int
    created_at: str
    latency_ms: float
    review_id: str | None = None
    image_ref: str | None = None
    risk_budget: dict[str, Any] | None = None


class PredictionListResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int
    limit: int
    offset: int


class UncertaintyResponse(BaseModel):
    prediction_id: str
    uncertainty: dict[str, float]
    calibrated_uncertainty: dict[str, float]
    temperature: float
    risk: RiskPayload
    reliability_diagram: list[dict[str, Any]] | None = None


class DriftStatusResponse(BaseModel):
    severity: str
    score: float
    window_size: int
    status: str
    signals: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    input_shift: dict[str, Any] = Field(default_factory=dict)
    prediction_shift: dict[str, Any] = Field(default_factory=dict)
    interpretation: str = ""


class DriftEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    severity: str
    score: float
    window_size: int
    status: str
    details: dict[str, Any]
    created_at: datetime


class ReviewCaseResponse(BaseModel):
    review_id: str
    prediction_id: str
    status: str
    predicted_index: int
    predicted_label: str
    confidence: float
    uncertainty: dict[str, Any]
    drift_severity: str
    diagnostics: list[str]
    created_at: datetime
    resolved_at: datetime | None = None
    verified_label: int | None = None
    reviewer: str | None = None
    notes: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


class ReviewSubmitRequest(BaseModel):
    label: int = Field(ge=0, le=9)
    reviewer: str = Field(min_length=1, max_length=128)
    notes: str | None = Field(default=None, max_length=2000)


class FeedbackRequest(BaseModel):
    review_id: str = Field(min_length=1, max_length=64)
    label: int = Field(ge=0, le=9)
    reviewer: str = Field(min_length=1, max_length=128)
    notes: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    feedback_id: str
    review_id: str
    label: int
    reviewer: str
    verified: bool
    created_at: datetime


class MetricsSummary(BaseModel):
    uptime_seconds: float
    counters: dict[str, float]
    gauges: dict[str, float]
    latency: dict[str, dict[str, float]]
    review_backlog: int
    drift_events_24h: int
    predictions_total: int


class Pagination(BaseModel):
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class RoutingMetrics(BaseModel):
    total: int
    action_distribution: dict[str, int]
    mean_risk: float
    mean_latency_ms: float
    escalation_rate: float
    abstention_rate: float
    review_rate: float
    budgeted_total: int = 0
    budget_misses: int = 0
    budget_miss_rate: float = 0.0
    tta_calls: int = 0


class ExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    experiment_id: str
    name: str
    status: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    dataset_id: str | None = None
    seed: int | None = None
    created_at: datetime


class ExperimentCreate(BaseModel):
    experiment_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    dataset_id: str | None = None
    seed: int | None = None
    notes: str | None = None
