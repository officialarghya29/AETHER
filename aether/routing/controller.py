"""Routing controller.

The controller is the only component that turns evidence into an action. It owns the policy
choice, enforces the escalation budget, and guarantees termination: at most
``routing.max_escalations`` model calls occur per request, so an escalation loop is structurally
impossible.
"""

from __future__ import annotations

from aether.config import RoutingConfig
from aether.routing.policy import (
    Action,
    DecisionPolicy,
    RoutingDecision,
    RoutingState,
    build_policy,
)
from aether.utils.logging import get_logger

logger = get_logger("aether.routing.controller")


class RoutingController:
    """Deterministic, budget-aware action selection."""

    def __init__(self, config: RoutingConfig, policy_name: str | None = None) -> None:
        self.config = config
        self.policy: DecisionPolicy = build_policy(policy_name or config.policy)
        logger.info("router configured", extra={"policy": self.policy.name})

    def decide(self, state: RoutingState) -> RoutingDecision:
        """Select an action for the supplied evidence."""
        # Defensive enforcement: even a buggy policy cannot exceed the escalation budget.
        if state.escalation_count >= self.config.max_escalations:
            state = RoutingState(**{**state.__dict__, "advanced_available": False})
        decision = self.policy.decide(state, self.config)
        logger.debug(
            "routing decision",
            extra={
                "action": decision.action.value,
                "risk": round(decision.risk, 4),
                "drift": state.drift_severity,
                "policy": self.policy.name,
            },
        )
        return decision

    def escalation_allowed(
        self,
        *,
        escalation_count: int,
        advanced_available: bool,
        estimated_advanced_latency_ms: float,
        elapsed_ms: float,
        remaining_budget_ms: float | None = None,
    ) -> bool:
        """Whether another model call fits the configured budget and limits."""
        if not self.config.enable_escalation or not advanced_available:
            return False
        if escalation_count >= self.config.max_escalations:
            return False
        budget = (
            self.config.latency_budget_ms if remaining_budget_ms is None else remaining_budget_ms
        )
        return elapsed_ms + estimated_advanced_latency_ms <= budget

    @property
    def action_space(self) -> list[str]:
        return [action.value for action in Action]
