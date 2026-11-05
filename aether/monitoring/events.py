"""Structured domain events.

Events are the audit trail: routing decisions, drift alerts, review actions and model
promotions. They are emitted through the standard logger with a stable ``event`` field and are
also persisted by the backend's ``AuditEvent`` table for durable history.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from aether.utils.logging import get_logger, request_id_var

logger = get_logger("aether.events")


@dataclass(frozen=True)
class Event:
    """A typed, timestamped domain event."""

    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    severity: str = "info"
    request_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return data


def emit(name: str, severity: str = "info", **payload: Any) -> Event:
    """Emit an event to the log stream and return it for persistence."""
    event = Event(
        name=name,
        payload=payload,
        severity=severity,
        request_id=request_id_var.get(),
    )
    log = logger.info if severity in ("info", "debug") else logger.warning
    log(name, extra={"event": name, "event_payload": payload})
    return event


def prediction_event(prediction_id: str, **payload: Any) -> Event:
    return emit("prediction.completed", prediction_id=prediction_id, **payload)


def routing_event(prediction_id: str, action: str, **payload: Any) -> Event:
    return emit("routing.decision", prediction_id=prediction_id, action=action, **payload)


def drift_event(severity: str, score: float, **payload: Any) -> Event:
    return emit(
        "drift.detected",
        severity="warning" if severity in ("moderate", "severe") else "info",
        drift_severity=severity,
        drift_score=score,
        **payload,
    )


def review_event(review_id: str, action: str, **payload: Any) -> Event:
    return emit(f"review.{action}", review_id=review_id, **payload)


def model_event(action: str, model_id: str, **payload: Any) -> Event:
    return emit(f"model.{action}", model_id=model_id, **payload)
