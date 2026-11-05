"""Cost model and expected-utility accounting for routing decisions.

Every action is scored in the same currency, so trade-offs are explicit rather than implicit in
threshold folklore. Utilities are in "correct answer" units; the cost of an incorrect automatic
answer dominates by default (``error=10`` vs ``correct=1``).

    U(ACCEPT)        = (1-r)·correct − r·error
    U(ESCALATE)      = (1-r_adv)·correct − r_adv·error − escalate
    U(HUMAN_REVIEW)  = (1-r_h)·correct − r_h·error − review
    U(ABSTAIN)       = −abstain

where ``r`` is the estimated risk of an incorrect answer and ``r_h`` the residual risk after a
human review (``risk × reviewer_error_rate``). Latency is enforced as a hard constraint, not
priced into the utility, because an over-budget answer is not a slow answer — it is a failure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

from aether.config import CostsConfig


class Action(StrEnum):
    """All actions the router may select."""

    ACCEPT = "ACCEPT"
    ESCALATE = "ESCALATE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    ABSTAIN = "ABSTAIN"
    MONITOR = "MONITOR"
    REJECT = "REJECT"


@dataclass(frozen=True)
class ActionOption:
    """One candidate action with its estimated utility and cost breakdown."""

    action: Action
    utility: float
    expected_risk: float
    added_cost: float
    feasible: bool = True
    reason: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["action"] = self.action.value
        return payload


def accept_utility(risk: float, costs: CostsConfig) -> float:
    return (1.0 - risk) * costs.correct - risk * costs.error


def escalate_utility(risk: float, costs: CostsConfig) -> float:
    return (1.0 - risk) * costs.correct - risk * costs.error - costs.escalate


def review_utility(risk: float, costs: CostsConfig, reviewer_error_rate: float) -> float:
    residual = risk * reviewer_error_rate
    return (1.0 - residual) * costs.correct - residual * costs.error - costs.review


def abstain_utility(costs: CostsConfig) -> float:
    return -costs.abstain


def evaluate_options(
    *,
    risk_light: float,
    risk_advanced: float | None,
    costs: CostsConfig,
    reviewer_error_rate: float,
    escalation_available: bool,
    escalation_affordable: bool,
    review_available: bool,
    abstention_enabled: bool,
    escalation_reason: str = "",
) -> list[ActionOption]:
    """Score every feasible action. Higher utility is better; ties break by a fixed order."""
    options = [ActionOption(Action.ACCEPT, accept_utility(risk_light, costs), risk_light, 0.0)]
    if escalation_available and risk_advanced is not None:
        options.append(
            ActionOption(
                Action.ESCALATE,
                escalate_utility(risk_advanced, costs),
                risk_advanced,
                costs.escalate,
                feasible=escalation_affordable,
                reason="" if escalation_affordable else escalation_reason,
            )
        )
    if review_available:
        options.append(
            ActionOption(
                Action.HUMAN_REVIEW,
                review_utility(risk_light, costs, reviewer_error_rate),
                risk_light * reviewer_error_rate,
                costs.review,
            )
        )
    if abstention_enabled:
        options.append(ActionOption(Action.ABSTAIN, abstain_utility(costs), 0.0, costs.abstain))
    return options


def best_option(options: list[ActionOption], priority: list[Action]) -> ActionOption:
    """Pick the highest-utility feasible option, breaking ties by ``priority`` order."""
    feasible = [option for option in options if option.feasible]
    if not feasible:
        return ActionOption(Action.ABSTAIN, abstain_utility(CostsConfig()), 0.0, 0.0)
    rank = {action: index for index, action in enumerate(priority)}
    return max(feasible, key=lambda option: (option.utility, -rank.get(option.action, 99)))
