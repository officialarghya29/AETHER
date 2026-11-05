"""Learned-policy artifact.

The learned policy deliberately separates two learned quantities:

1. **Risk models** (``aether.routing.risk_model``) estimate ``P(error)`` for the lightweight and
   advanced models from inference-time evidence.
2. **A decision rule** (``aether.routing.policy.LearnedPolicy``) maps those risks and configured
   costs to an action by expected utility.

Both are fitted on the routing-development split and stored as one auditable JSON artifact that
records the split, sample count and quality metrics of each component. No ground-truth label is
required at inference time, and no test data participates in fitting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.errors import ArtifactNotFoundError
from aether.routing.risk_model import RiskModel, RiskModelArtifact
from aether.utils.io import atomic_write_json, read_json
from aether.utils.logging import get_logger

logger = get_logger("aether.routing.learned_policy")


@dataclass(frozen=True)
class LearnedPolicyArtifact:
    """Container for the flight and post-escalation risk models.

    ``lightweight`` estimates the risk of the first call, ``advanced`` the risk of the advanced
    model *after* it has answered, and ``advanced_precall`` the risk the advanced model would
    carry, estimated from lightweight evidence only. The pre-call head is what makes escalation a
    priced decision instead of an oracle: no output of the second model is consulted before it is
    paid for.
    """

    policy_name: str
    split: str
    samples: int
    lightweight: dict[str, Any]
    advanced: dict[str, Any]
    expected_benefit: float
    advanced_precall: dict[str, Any] | None = None
    fitted_at: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "split": self.split,
            "samples": self.samples,
            "lightweight": self.lightweight,
            "advanced": self.advanced,
            "advanced_precall": self.advanced_precall,
            "expected_benefit": self.expected_benefit,
            "fitted_at": self.fitted_at,
            "notes": self.notes,
            "schema_version": 2,
        }


def build_policy_artifact(
    *,
    lightweight: RiskModel,
    advanced: RiskModel,
    samples: int,
    advanced_precall: RiskModel | None = None,
    split: str = "calibration",
    expected_benefit: float = 0.0,
    notes: list[str] | None = None,
) -> LearnedPolicyArtifact:
    """Package the fitted risk models into a single versioned artifact."""
    return LearnedPolicyArtifact(
        policy_name="learned",
        split=split,
        samples=samples,
        lightweight=lightweight.describe(),
        advanced=advanced.describe(),
        advanced_precall=advanced_precall.describe() if advanced_precall else None,
        expected_benefit=float(expected_benefit),
        fitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        notes=notes or [],
    )


def save_policy_artifact(artifact: LearnedPolicyArtifact, path: Path) -> None:
    atomic_write_json(path, artifact.to_dict())
    logger.info(
        "learned policy artifact written",
        extra={"path": str(path), "samples": artifact.samples},
    )


def load_policy_artifact(path: Path) -> LearnedPolicyArtifact:
    payload = read_json(path, default=None)
    if not payload:
        raise ArtifactNotFoundError(
            f"learned policy artifact not found: {path.name}",
            details={"expected": str(path), "remedy": "run training/calibrate.py"},
        )
    return LearnedPolicyArtifact(
        policy_name=payload.get("policy_name", "learned"),
        split=payload.get("split", "routing_dev"),
        samples=int(payload.get("samples", 0)),
        lightweight=payload["lightweight"],
        advanced=payload["advanced"],
        advanced_precall=payload.get("advanced_precall"),
        expected_benefit=float(payload.get("expected_benefit", 0.0)),
        fitted_at=payload.get("fitted_at", ""),
        notes=payload.get("notes", []),
    )


def load_risk_models(path: Path) -> tuple[RiskModel, RiskModel]:
    """Load the first-call and post-escalation risk models."""
    payload = read_json(path, default=None)
    if not payload:
        raise ArtifactNotFoundError(f"learned policy artifact not found: {path.name}")
    return (
        RiskModel(RiskModelArtifact(**payload["lightweight"])),
        RiskModel(RiskModelArtifact(**payload["advanced"])),
    )


def load_precall_risk_model(path: Path) -> RiskModel | None:
    """Load the pre-call advanced risk head, if the artifact was fitted with one."""
    payload = read_json(path, default=None)
    if not payload or not payload.get("advanced_precall"):
        return None
    return RiskModel(RiskModelArtifact(**payload["advanced_precall"]))
