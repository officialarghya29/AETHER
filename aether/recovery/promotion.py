"""Candidate promotion.

Promotion requires all three: a passed offline evaluation, a named human approver, and a
registered candidate version. Feedback arriving from the review queue can never activate a model
by itself — that is the entire point of the gate.
"""

from __future__ import annotations

from aether.errors import ValidationFailedError
from aether.recovery.evaluation import CandidateEvaluation
from aether.recovery.registry import ModelRegistryStore
from aether.utils.logging import get_logger

logger = get_logger("aether.recovery.promotion")


def promote_candidate(
    store: ModelRegistryStore,
    *,
    role: str,
    version: str,
    evaluation: CandidateEvaluation,
    approver: str,
    reason: str = "",
) -> dict:
    """Activate a candidate version if, and only if, the gate passes."""
    if not approver or not approver.strip():
        raise ValidationFailedError("promotion requires a named approver")
    if not evaluation.approved:
        raise ValidationFailedError(
            "candidate did not pass offline evaluation",
            details={"reasons": evaluation.reasons},
        )
    store.activate(role, version, approver=approver.strip(), reason=reason)
    logger.info(
        "candidate promoted",
        extra={
            "role": role,
            "version": version,
            "approver": approver,
            "accuracy": evaluation.candidate.get("accuracy"),
        },
    )
    return {
        "role": role,
        "version": version,
        "approved_by": approver,
        "evaluation": evaluation.to_dict(),
        "active": store.active(role),
    }
