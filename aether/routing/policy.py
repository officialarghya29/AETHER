"""Routing policies.

Three policies share one interface and one evidence structure, so a benchmark can swap them
without touching the pipeline:

* ``RuleBasedPolicy`` — ordered thresholds. The transparent baseline.
* ``CostAwarePolicy`` — maximises expected utility over all feasible actions.
* ``LearnedPolicy`` — cost-aware utility driven by fitted risks, with a minimum expected
  benefit gate so escalation is refused when the predicted improvement is not material.

All policies are pure functions of the supplied state: identical evidence and configuration
produce an identical decision, which is what makes the router testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from aether.config import RoutingConfig
from aether.routing.cost import Action, ActionOption, best_option, evaluate_options
from aether.utils.logging import get_logger

logger = get_logger("aether.routing.policy")

# Tie-break order: prefer answering over spending, and prefer spending over refusing.
PRIORITY: list[Action] = [
    Action.ACCEPT,
    Action.MONITOR,
    Action.ESCALATE,
    Action.HUMAN_REVIEW,
    Action.ABSTAIN,
]


@dataclass(frozen=True)
class RoutingState:
    """Everything a policy may consider. No labels, no future information."""

    risk_light: float
    entropy: float
    max_probability: float
    margin: float
    drift_score: float
    drift_severity: str
    risk_advanced: float | None = None
    disagreement: float | None = None
    escalation_count: int = 0
    advanced_available: bool = False
    # ``risk_advanced`` can be a prediction made before paying for the second call. Only a
    # completed evaluation may be used as evidence that the advanced model answered well.
    advanced_evaluated: bool = False
    escalation_affordable: bool = True
    elapsed_ms: float = 0.0
    remaining_budget_ms: float = float("inf")


@dataclass(frozen=True)
class RoutingDecision:
    """A routing decision with its complete justification."""

    action: Action
    policy: str
    risk: float
    reason_codes: tuple[str, ...] = ()
    explanation: str = ""
    utilities: tuple[ActionOption, ...] = ()
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "policy": self.policy,
            "risk": round(float(self.risk), 4),
            "reason_codes": list(self.reason_codes),
            "explanation": self.explanation,
            "utilities": [option.to_dict() for option in self.utilities],
            "details": self.details,
        }


class DecisionPolicy(Protocol):
    """Policy contract."""

    name: str

    def decide(self, state: RoutingState, config: RoutingConfig) -> RoutingDecision: ...


def _escalation_allowed(state: RoutingState, config: RoutingConfig) -> bool:
    return (
        config.enable_escalation
        and state.advanced_available
        and state.escalation_count < config.max_escalations
        and state.escalation_affordable
    )


class RuleBasedPolicy:
    """Ordered-threshold baseline policy."""

    name = "rule_based"

    def decide(self, state: RoutingState, config: RoutingConfig) -> RoutingDecision:
        reasons: list[str] = []
        risk = float(state.risk_light)
        can_escalate = _escalation_allowed(state, config)

        if risk <= config.accept_risk:
            if state.drift_severity == "severe":
                reasons.append("SEVERE_DRIFT_CONFIDENT_PREDICTION")
                return RoutingDecision(
                    action=Action.MONITOR,
                    policy=self.name,
                    risk=risk,
                    reason_codes=tuple(reasons + ["ACCEPT_RISK_BELOW_THRESHOLD"]),
                    explanation=(
                        "Prediction risk is within the acceptance budget, but the input window "
                        "shows severe distribution shift. The prediction is returned and the "
                        "anomaly is recorded for monitoring."
                    ),
                )
            reasons.append("ACCEPT_RISK_BELOW_THRESHOLD")
            return RoutingDecision(
                action=Action.ACCEPT,
                policy=self.name,
                risk=risk,
                reason_codes=tuple(reasons),
                explanation=(
                    f"Estimated error risk {risk:.3f} is at or below the acceptance threshold "
                    f"{config.accept_risk:.2f}."
                ),
            )

        if can_escalate:
            reasons.append("RISK_ABOVE_ACCEPT_THRESHOLD")
            reasons.append("ESCALATION_WITHIN_BUDGET")
            return RoutingDecision(
                action=Action.ESCALATE,
                policy=self.name,
                risk=risk,
                reason_codes=tuple(reasons),
                explanation=(
                    f"Estimated error risk {risk:.3f} exceeds {config.accept_risk:.2f}; "
                    "a second model call fits the latency budget."
                ),
            )

        if state.advanced_evaluated and state.risk_advanced is not None:
            reasons.append("ADVANCED_MODEL_EVALUATED")
            return RoutingDecision(
                action=Action.ACCEPT,
                policy=self.name,
                risk=float(state.risk_advanced),
                reason_codes=tuple(reasons + ["ESCALATION_EXHAUSTED"]),
                explanation=(
                    "The advanced model has already run and the escalation budget is spent; its "
                    "prediction is returned with the measured risk attached."
                ),
            )

        if config.review_available:
            reasons.append("RISK_ABOVE_ACCEPT_THRESHOLD")
            reasons.append("ESCALATION_UNAVAILABLE_OR_UNBUDGETED")
            reasons.append("ROUTED_TO_HUMAN_REVIEW")
            return RoutingDecision(
                action=Action.HUMAN_REVIEW,
                policy=self.name,
                risk=risk,
                reason_codes=tuple(reasons),
                explanation=(
                    f"Estimated error risk {risk:.3f} is too high to answer automatically and "
                    "escalation is not available within budget, so a human reviewer decides."
                ),
            )

        reasons.append("NO_RELIABLE_ACTION_AVAILABLE")
        action = Action.ABSTAIN if config.enable_abstention else Action.ACCEPT
        return RoutingDecision(
            action=action,
            policy=self.name,
            risk=risk,
            reason_codes=tuple(reasons),
            explanation=(
                "Neither escalation nor human review is available; the system declines to "
                "answer automatically."
                if action is Action.ABSTAIN
                else "No review path is configured; returning the prediction with its risk."
            ),
        )


class CostAwarePolicy:
    """Expected-utility policy over all feasible actions."""

    name = "cost_aware"

    def decide(self, state: RoutingState, config: RoutingConfig) -> RoutingDecision:
        can_escalate = _escalation_allowed(state, config)
        # Once the second call has run its risk is measured, not estimated: price ACCEPT and
        # HUMAN_REVIEW against that evidence instead of the stale first-call risk. Otherwise the
        # policy pays for an escalation and then declines to use the answer it bought.
        decision_risk = (
            float(state.risk_advanced)
            if state.advanced_evaluated and state.risk_advanced is not None
            else state.risk_light
        )
        options = evaluate_options(
            risk_light=decision_risk,
            risk_advanced=state.risk_advanced,
            costs=config.costs,
            reviewer_error_rate=config.reviewer_error_rate,
            escalation_available=can_escalate,
            escalation_affordable=state.escalation_affordable,
            review_available=config.review_available,
            abstention_enabled=config.enable_abstention,
            escalation_reason="latency budget exhausted",
        )
        chosen = best_option(options, PRIORITY)
        reasons = [f"UTILITY_{chosen.action.value}"]
        if state.drift_severity in ("moderate", "severe"):
            reasons.append(f"DRIFT_{state.drift_severity.upper()}")
        if state.disagreement is not None and state.disagreement > 0.5:
            reasons.append("MODEL_DISAGREEMENT")
        # An abstention option carries no expected risk of its own, so the explanation quotes the
        # risk the decision was actually made against.
        assessed_risk = decision_risk if chosen.action is Action.ABSTAIN else chosen.expected_risk
        explanation = (
            f"Expected utility of {chosen.action.value} is {chosen.utility:.3f} "
            f"at estimated risk {assessed_risk:.3f}; "
            f"alternatives: "
            + ", ".join(
                f"{option.action.value}={option.utility:.3f}"
                for option in options
                if option.action is not chosen.action
            )
            + "."
        )
        return RoutingDecision(
            action=chosen.action,
            policy=self.name,
            risk=chosen.expected_risk,
            reason_codes=tuple(reasons),
            explanation=explanation,
            utilities=tuple(options),
        )


class LearnedPolicy(CostAwarePolicy):
    """Cost-aware policy with a learned minimum-benefit gate on escalation.

    The gate answers a different question from the utility comparison: *how much* risk would
    escalation have to remove before its cost is worth paying? If the fitted reduction is below
    ``min_expected_benefit`` the escalation option is withdrawn, which suppresses pointless
    second model calls near the decision boundary.
    """

    name = "learned"

    def decide(self, state: RoutingState, config: RoutingConfig) -> RoutingDecision:
        decision = super().decide(state, config)
        threshold = config.learned_policy.min_expected_benefit
        if state.risk_advanced is None or state.escalation_count > 0:
            return decision
        expected_benefit = state.risk_light - state.risk_advanced
        if decision.action is Action.ESCALATE and expected_benefit < threshold:
            return RoutingDecision(
                action=Action.ACCEPT,
                policy=self.name,
                risk=state.risk_light,
                reason_codes=(
                    "ESCALATION_BENEFIT_TOO_SMALL",
                    f"EXPECTED_BENEFIT_{expected_benefit:.3f}",
                ),
                explanation=(
                    f"Escalation was projected to reduce error risk by only {expected_benefit:.3f} "
                    f"(below the {threshold:.2f} minimum benefit), so the model call is skipped."
                ),
                utilities=decision.utilities,
            )
        return decision


POLICIES: dict[str, type] = {
    "rule_based": RuleBasedPolicy,
    "cost_aware": CostAwarePolicy,
    "learned": LearnedPolicy,
}


def build_policy(name: str) -> DecisionPolicy:
    """Instantiate a policy by configured name."""
    if name not in POLICIES:
        raise ValueError(f"unknown routing policy '{name}'; available: {sorted(POLICIES)}")
    return POLICIES[name]()
