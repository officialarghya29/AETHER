"""Offline candidate evaluation.

A retrained candidate is only promotable if it beats the incumbent on a held-out evaluation set
by a configured margin, without a material calibration regression. The gate is deliberately
conservative and reports every reason it failed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PromotionCriteria:
    """Acceptance thresholds for a candidate model."""

    min_accuracy: float = 0.70
    min_accuracy_gain: float = 0.0
    max_ece: float = 0.15
    max_ece_regression: float = 0.02
    min_samples: int = 200


@dataclass(frozen=True)
class CandidateEvaluation:
    """Comparison of a candidate against the incumbent on the same evaluation set."""

    candidate: dict[str, float]
    incumbent: dict[str, float]
    samples: int
    approved: bool
    reasons: list[str]
    criteria: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_candidate(
    candidate: dict[str, float],
    incumbent: dict[str, float],
    *,
    samples: int,
    criteria: PromotionCriteria | None = None,
) -> CandidateEvaluation:
    """Apply the promotion gate and return a fully explained decision."""
    criteria = criteria or PromotionCriteria()
    reasons: list[str] = []
    if samples < criteria.min_samples:
        reasons.append(
            f"evaluation set has {samples} samples; at least {criteria.min_samples} required"
        )
    candidate_accuracy = float(candidate.get("accuracy", 0.0))
    incumbent_accuracy = float(incumbent.get("accuracy", 0.0))
    candidate_ece = float(candidate.get("ece", 1.0))
    incumbent_ece = float(incumbent.get("ece", 1.0))
    if candidate_accuracy < criteria.min_accuracy:
        reasons.append(
            f"candidate accuracy {candidate_accuracy:.4f} below minimum {criteria.min_accuracy:.4f}"
        )
    if candidate_accuracy - incumbent_accuracy < criteria.min_accuracy_gain:
        reasons.append(
            f"accuracy gain {candidate_accuracy - incumbent_accuracy:+.4f} below required "
            f"{criteria.min_accuracy_gain:+.4f}"
        )
    if candidate_ece > criteria.max_ece:
        reasons.append(f"candidate ECE {candidate_ece:.4f} above maximum {criteria.max_ece:.4f}")
    if candidate_ece - incumbent_ece > criteria.max_ece_regression:
        reasons.append(
            f"calibration regressed by {candidate_ece - incumbent_ece:+.4f}, "
            f"maximum {criteria.max_ece_regression:+.4f}"
        )
    return CandidateEvaluation(
        candidate=candidate,
        incumbent=incumbent,
        samples=int(samples),
        approved=not reasons,
        reasons=reasons,
        criteria=asdict(criteria),
    )
