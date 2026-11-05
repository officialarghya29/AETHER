"""Routing: determinism, budget enforcement, action coverage and cost accounting."""

from __future__ import annotations

import numpy as np
import pytest

from aether.config import load_config
from aether.evaluation.routing import PolicyRecord, simulate_policy
from aether.routing.controller import RoutingController
from aether.routing.cost import Action, evaluate_options
from aether.routing.policy import RoutingState, build_policy
from aether.routing.risk_model import RiskModel, features_from_evidence


@pytest.fixture
def routing_config():
    return load_config().routing


def _state(**overrides) -> RoutingState:
    defaults = {
        "risk_light": 0.05,
        "entropy": 0.2,
        "max_probability": 0.9,
        "margin": 0.7,
        "drift_score": 0.0,
        "drift_severity": "none",
        "risk_advanced": None,
        "disagreement": None,
        "escalation_count": 0,
        "advanced_available": True,
        "escalation_affordable": True,
        "elapsed_ms": 5.0,
    }
    defaults.update(overrides)
    return RoutingState(**defaults)


def test_low_risk_is_accepted(routing_config):
    decision = RoutingController(routing_config, policy_name="rule_based").decide(
        _state(risk_light=0.02)
    )
    assert decision.action is Action.ACCEPT
    assert "ACCEPT_RISK_BELOW_THRESHOLD" in decision.reason_codes


def test_high_risk_escalates_when_affordable(routing_config):
    decision = RoutingController(routing_config, policy_name="rule_based").decide(
        _state(risk_light=0.6)
    )
    assert decision.action is Action.ESCALATE


def test_high_risk_without_escalation_goes_to_review(routing_config):
    decision = RoutingController(routing_config, policy_name="rule_based").decide(
        _state(risk_light=0.6, advanced_available=False)
    )
    assert decision.action is Action.HUMAN_REVIEW


def test_abstains_when_review_is_unavailable(routing_config):
    config = routing_config.model_copy(update={"review_available": False})
    decision = RoutingController(config).decide(_state(risk_light=0.8, advanced_available=False))
    assert decision.action is Action.ABSTAIN


def test_severe_drift_with_confident_prediction_monitors(routing_config):
    decision = RoutingController(routing_config, policy_name="rule_based").decide(
        _state(risk_light=0.01, drift_severity="severe", drift_score=8.0)
    )
    assert decision.action is Action.MONITOR
    assert "SEVERE_DRIFT_CONFIDENT_PREDICTION" in decision.reason_codes


def test_escalation_budget_is_enforced(routing_config):
    decision = RoutingController(routing_config).decide(
        _state(risk_light=0.9, escalation_count=1, advanced_available=True)
    )
    assert decision.action is not Action.ESCALATE


def test_controller_strips_advanced_after_budget(routing_config):
    controller = RoutingController(routing_config)
    state = _state(risk_light=0.9, escalation_count=99, advanced_available=True)
    decision = controller.decide(state)
    assert decision.action in (Action.HUMAN_REVIEW, Action.ABSTAIN)


def test_decisions_are_deterministic(routing_config):
    controller = RoutingController(routing_config, policy_name="rule_based")
    runs = {controller.decide(_state(risk_light=0.42)).action for _ in range(20)}
    assert len(runs) == 1


def test_escalation_affordability_respects_latency_budget(routing_config):
    controller = RoutingController(routing_config)
    assert not controller.escalation_allowed(
        escalation_count=0,
        advanced_available=True,
        estimated_advanced_latency_ms=500.0,
        elapsed_ms=10.0,
    )
    assert controller.escalation_allowed(
        escalation_count=0,
        advanced_available=True,
        estimated_advanced_latency_ms=5.0,
        elapsed_ms=10.0,
    )


def test_cost_aware_utility_prefers_escalation_when_it_pays(routing_config):
    options = evaluate_options(
        risk_light=0.7,
        risk_advanced=0.05,
        costs=routing_config.costs,
        reviewer_error_rate=routing_config.reviewer_error_rate,
        escalation_available=True,
        escalation_affordable=True,
        review_available=True,
        abstention_enabled=True,
    )
    utilities = {option.action: option.utility for option in options}
    assert utilities[Action.ESCALATE] > utilities[Action.ACCEPT]
    assert utilities[Action.ESCALATE] > utilities[Action.ABSTAIN]


def test_learned_policy_skips_pointless_escalation(routing_config):
    config = routing_config.model_copy(
        update={
            "policy": "learned",
            "learned_policy": routing_config.learned_policy.model_copy(
                update={"min_expected_benefit": 0.5}
            ),
        }
    )
    # Chosen so escalation wins on utility (0.11 x (error+correct) > escalate cost) while the
    # learned gate still refuses it because the benefit is below the configured minimum.
    state = _state(risk_light=0.27, risk_advanced=0.16)
    decision = RoutingController(config, policy_name="learned").decide(state)
    assert decision.action is not Action.ESCALATE
    assert "ESCALATION_BENEFIT_TOO_SMALL" in decision.reason_codes


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        build_policy("magic")


def test_risk_model_fit_requires_routing_dev_split():
    features = {
        "entropy": np.array([0.1, 0.9, 0.5, 1.2]),
        "max_probability": np.array([0.9, 0.4, 0.7, 0.3]),
        "margin": np.array([0.8, 0.1, 0.4, 0.05]),
        "drift_score": np.zeros(4),
        "shifted_flag": np.zeros(4),
        "disagreement": np.zeros(4),
    }
    labels = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="routing-development"):
        RiskModel.fit(features, labels, model_id="m", split="test")
    model = RiskModel.fit(features, labels, model_id="m")
    probability = model.predict(features_from_evidence(
        entropy=0.5, max_probability=0.7, margin=0.4, drift_score=0.0,
        severity="none", disagreement=0.0,
    ))
    assert 0.0 <= probability <= 1.0


def test_risk_model_requires_both_classes():
    features = {
        name: np.zeros(4)
        for name in ("entropy", "max_probability", "margin", "drift_score", "shifted_flag", "disagreement")
    }
    with pytest.raises(ValueError):
        RiskModel.fit(features, np.zeros(4, dtype=int), model_id="m")


def test_simulation_prices_abstention_like_the_router_does():
    """Abstaining costs the abstention cost, not an additional wrong-answer penalty."""
    config = load_config()
    record = PolicyRecord(
        light_probabilities=np.array([0.6, 0.4]),
        advanced_probabilities=np.array([0.6, 0.4]),
        label=1,  # the lightweight model is wrong on this sample
    )
    abstained = simulate_policy([record], config, "aether", constant_risk=0.95)
    assert abstained["action_distribution"] == {"ABSTAIN": 1}
    assert abstained["expected_cost_per_sample"] == pytest.approx(config.routing.costs.abstain)
    assert abstained["coverage"] == 0.0

    wrong_answer = simulate_policy([record], config, "always_light")
    assert wrong_answer["expected_cost_per_sample"] == pytest.approx(config.routing.costs.error)


def test_cost_aware_policy_uses_advanced_risk_after_escalation(routing_config):
    """A completed second call must be priced with its measured risk, not the stale light risk."""
    controller = RoutingController(routing_config, policy_name="cost_aware")
    decision = controller.decide(
        _state(
            risk_light=0.55,
            risk_advanced=0.02,
            advanced_evaluated=True,
            advanced_available=False,
            escalation_count=1,
        )
    )
    assert decision.action is Action.ACCEPT
    assert decision.risk == pytest.approx(0.02)
