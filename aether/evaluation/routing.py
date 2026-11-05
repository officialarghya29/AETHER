"""Offline routing evaluation.

Logged evidence rows are replayed through each candidate policy using the *same* controller the
service uses. That makes the comparison meaningful: the baselines and AETHER differ only in
decision logic, never in preprocessing, temperature or risk plumbing.

Assumption stated explicitly: a human-reviewed sample is resolved correctly with probability
``1 - reviewer_error_rate``. Reviews are not free (configured cost) and not instant, so a policy
that routes everything to humans scores well on accuracy and badly on cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aether.config import AetherConfig, RoutingConfig
from aether.routing.controller import RoutingController
from aether.routing.cost import Action
from aether.routing.policy import RoutingState
from aether.routing.risk_model import default_risk, features_from_evidence
from aether.uncertainty.entropy import (
    max_probability,
    normalized_entropy,
    prediction_margin,
    predictive_entropy,
)


@dataclass(frozen=True)
class PolicyRecord:
    """One logged observation: model outputs, drift evidence and the truth."""

    light_probabilities: np.ndarray
    advanced_probabilities: np.ndarray
    label: int
    drift_severity: str = "none"
    drift_score: float = 0.0
    light_latency_ms: float = 1.0
    advanced_latency_ms: float = 10.0

    @property
    def light_correct(self) -> bool:
        return int(np.argmax(self.light_probabilities)) == int(self.label)

    @property
    def advanced_correct(self) -> bool:
        return int(np.argmax(self.advanced_probabilities)) == int(self.label)


BASELINES = ("always_light", "always_advanced", "fixed_threshold", "uncertainty_only", "shift_aware", "aether")


def _uncertainty(probabilities: np.ndarray) -> dict[str, float]:
    return {
        "predictive_entropy": float(predictive_entropy(probabilities)),
        "max_probability": float(max_probability(probabilities)),
        "prediction_margin": float(prediction_margin(probabilities)),
    }


def simulate_policy(
    records: list[PolicyRecord],
    config: AetherConfig,
    policy: str,
    *,
    confidence_threshold: float = 0.85,
    risk_light_model=None,
    risk_advanced_model=None,
    risk_precall_model=None,
    constant_risk: float | None = None,
    ignore_disagreement: bool = False,
) -> dict:
    """Replay records through one policy and return accuracy, coverage, cost and latency."""
    if not records:
        raise ValueError("no records to evaluate")
    routing_config: RoutingConfig = config.routing
    controller = _controller_for(policy, routing_config)

    answered = 0
    correct = 0
    escalated = 0
    reviewed = 0
    abstained = 0
    cost = 0.0
    latencies: list[float] = []
    actions: dict[str, int] = {}
    risks: list[float] = []

    for record in records:
        light_probs = np.asarray(record.light_probabilities, dtype=np.float64)
        advanced_probs = np.asarray(record.advanced_probabilities, dtype=np.float64)
        light_unc = _uncertainty(light_probs)
        advanced_unc = _uncertainty(advanced_probs)
        drift_severity = record.drift_severity
        drift_score = float(record.drift_score)

        if policy == "always_light":
            action = Action.ACCEPT
            chosen_correct = record.light_correct
            added_latency = record.light_latency_ms
            added_cost = 0.0
        elif policy == "always_advanced":
            action = Action.ACCEPT
            chosen_correct = record.advanced_correct
            added_latency = record.advanced_latency_ms
            added_cost = routing_config.costs.escalate
        else:
            disagreement = (
                0.0
                if ignore_disagreement
                else float(0.5 * np.abs(light_probs - advanced_probs).sum())
            )
            # The lightweight risk model was fitted without disagreement (unknown before a second
            # call); the advanced model was fitted with it, so inference must match each contract.
            risk_light = _risk(risk_light_model, light_unc, drift_severity, drift_score, 0.0)
            # Before the second call is paid for, its risk is estimated from lightweight evidence
            # alone. The advanced model's own probabilities must not enter this decision.
            risk_precall = _risk(
                risk_precall_model, light_unc, drift_severity, drift_score, 0.0
            )
            if constant_risk is not None:
                # Ablation: removes uncertainty information without touching the predictions.
                risk_light = risk_precall = float(constant_risk)
            if policy == "uncertainty_only":
                # Deliberately blind to shift evidence: isolates the value of drift signals.
                drift_severity, drift_score = "none", 0.0
                risk_light = float(normalized_entropy(light_probs))
                risk_precall = float(normalized_entropy(light_probs))
            state = RoutingState(
                risk_light=risk_light,
                risk_advanced=risk_precall,
                entropy=light_unc["predictive_entropy"],
                max_probability=light_unc["max_probability"],
                margin=light_unc["prediction_margin"],
                drift_score=drift_score,
                drift_severity=drift_severity,
                disagreement=disagreement,
                escalation_count=0,
                advanced_available=True,
            )
            action = controller.decide(state).action
            risks.append(risk_light)
            escalated_flag = False
            if action is Action.ESCALATE:
                escalated_flag = True
                # The call is now paid for: re-decide with the advanced model's own, fully
                # observed risk (advanced uncertainty + disagreement).
                risk_advanced = _risk(
                    risk_advanced_model, advanced_unc, drift_severity, drift_score, disagreement
                )
                post = RoutingState(
                    risk_light=risk_light,
                    risk_advanced=risk_advanced,
                    entropy=advanced_unc["predictive_entropy"],
                    max_probability=advanced_unc["max_probability"],
                    margin=advanced_unc["prediction_margin"],
                    drift_score=drift_score,
                    drift_severity=drift_severity,
                    disagreement=state.disagreement,
                    escalation_count=1,
                    advanced_available=False,
                    advanced_evaluated=True,
                )
                action = controller.decide(post).action
            if policy == "fixed_threshold":
                action = (
                    Action.ACCEPT
                    if light_unc["max_probability"] >= confidence_threshold
                    else Action.ABSTAIN
                )

            escalated_used = escalated_flag
            chosen_correct = record.light_correct
            added_latency = record.light_latency_ms
            added_cost = 0.0
            if escalated_used:
                escalated += 1
            if escalated_used and action in (Action.ACCEPT, Action.MONITOR):
                # Escalation ran and its answer is the one returned.
                chosen_correct = record.advanced_correct
                added_latency += record.advanced_latency_ms
                added_cost += routing_config.costs.escalate
            elif action is Action.HUMAN_REVIEW:
                reviewed += 1
                # Residual reviewer error is the only remaining risk after verification.
                chosen_correct = True
                added_latency += record.light_latency_ms
                added_cost += routing_config.costs.review
            elif action is Action.ABSTAIN:
                abstained += 1
                chosen_correct = False
                added_cost += routing_config.costs.abstain

        actions[action.value] = actions.get(action.value, 0) + 1
        # Cost accounting mirrors the router's utility model: abstention is charged its own
        # configured cost (not an additional wrong-answer penalty), and a review carries the
        # review cost plus the expected residual error that survives the review.
        if action is Action.ABSTAIN:
            cost += added_cost
        else:
            cost += added_cost + (0.0 if chosen_correct else routing_config.costs.error)
            if action is Action.HUMAN_REVIEW:
                cost += routing_config.costs.error * routing_config.reviewer_error_rate
        latencies.append(added_latency)
        if action in (Action.ACCEPT, Action.MONITOR):
            answered += 1
            correct += int(chosen_correct)

    total = len(records)
    answered_safe = max(1, answered)
    accuracy_on_answered = correct / answered_safe if answered else 0.0
    accuracy_overall = (correct + reviewed * (1 - routing_config.reviewer_error_rate)) / total
    return {
        "policy": policy,
        "samples": total,
        "coverage": answered / total,
        "accuracy_on_answered": accuracy_on_answered,
        "accuracy_including_reviews": accuracy_overall,
        # Share of traffic where the advanced model was actually invoked, whether or not the
        # second call resolved the case. ``unresolved_escalation_rate`` counts samples whose
        # final action is still ESCALATE (no answer produced).
        "escalation_rate": escalated / total,
        "unresolved_escalation_rate": actions.get("ESCALATE", 0) / total,
        "review_rate": reviewed / total,
        "abstention_rate": abstained / total,
        "action_distribution": actions,
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "expected_cost_per_sample": cost / total,
        "light_correct_rate": float(np.mean([record.light_correct for record in records])),
        "advanced_correct_rate": float(np.mean([record.advanced_correct for record in records])),
    }


def _risk(
    model,
    uncertainty: dict[str, float],
    severity: str,
    drift_score: float,
    disagreement: float | None = None,
) -> float:
    evidence = features_from_evidence(
        entropy=uncertainty["predictive_entropy"],
        max_probability=uncertainty["max_probability"],
        margin=uncertainty["prediction_margin"],
        drift_score=drift_score,
        severity=severity,
        disagreement=disagreement,
    )
    return float(model.predict(evidence)) if model is not None else float(default_risk(evidence))


def _controller_for(policy: str, config: RoutingConfig) -> RoutingController:
    name = {
        "aether": config.policy,
        "shift_aware": "rule_based",
        "uncertainty_only": "cost_aware",
        "fixed_threshold": "rule_based",
    }.get(policy, "cost_aware")
    return RoutingController(config, policy_name=name)
