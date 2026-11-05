"""Failure diagnosis engine.

Each rule maps evidence to a diagnostic with a stable code, a severity, the exact numbers that
triggered it, a cautious explanation and a recommended action. Rules never assert causality:
``HIGH_EMBEDDING_SHIFT`` says the embedding distribution moved, not that the model is wrong for
that reason.

Diagnostic codes are part of the API contract, so they are added rather than renamed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from aether.diagnosis.image_quality import ImageQuality

SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Documented rule thresholds.

    These are engineering defaults, and the reliability study reports how often each rule fires
    on clean data so an operator can judge the false-alarm rate before changing them.
    """

    high_entropy: float = 1.20          # nats, 10 classes (ln 10 = 2.30)
    low_confidence: float = 0.55
    low_margin: float = 0.15
    drift_moderate: float = 3.0
    drift_severe: float = 6.0
    disagreement_high: float = 0.50
    blur_low_sharpness: float = 0.0008
    brightness_dark: float = 0.20
    brightness_bright: float = 0.80
    contrast_low: float = 0.06


@dataclass(frozen=True)
class Diagnostic:
    """One evidence-backed finding."""

    code: str
    severity: str
    title: str
    evidence: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    recommended_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiagnosisEngine:
    """Applies the diagnostic rule set to one prediction's evidence."""

    def __init__(self, thresholds: DiagnosticThresholds | None = None) -> None:
        self.thresholds = thresholds or DiagnosticThresholds()

    def diagnose(
        self,
        *,
        entropy: float,
        max_probability: float,
        margin: float,
        predicted_label: str,
        drift_severity: str = "none",
        drift_score: float = 0.0,
        disagreement: float | None = None,
        quality: ImageQuality | None = None,
        advanced_used: bool = False,
        advanced_failed: bool = False,
        advanced_agrees: bool | None = None,
    ) -> list[Diagnostic]:
        """Return every diagnostic supported by the evidence, most severe first."""
        findings: list[Diagnostic] = []
        thresholds = self.thresholds

        if entropy >= thresholds.high_entropy:
            findings.append(
                Diagnostic(
                    code="HIGH_ENTROPY",
                    severity="high" if entropy >= thresholds.high_entropy * 1.4 else "medium",
                    title="Prediction ambiguity",
                    evidence={
                        "predictive_entropy_nats": round(entropy, 4),
                        "threshold": thresholds.high_entropy,
                        "predicted_class": predicted_label,
                    },
                    explanation=(
                        "The predictive distribution is spread across several classes. This "
                        "indicates the model does not concentrate probability on one outcome; "
                        "it does not by itself imply the prediction is wrong."
                    ),
                    recommended_action="Route to human review or escalate before answering.",
                )
            )
        if max_probability <= thresholds.low_confidence:
            findings.append(
                Diagnostic(
                    code="LOW_CONFIDENCE",
                    severity="medium" if max_probability > 0.35 else "high",
                    title="Weak top-class confidence",
                    evidence={
                        "max_probability": round(max_probability, 4),
                        "threshold": thresholds.low_confidence,
                    },
                    explanation=(
                        "The highest class probability is low. Softmax confidence is not a "
                        "probability of correctness, so this is treated as a warning signal "
                        "rather than proof of error."
                    ),
                    recommended_action="Treat the answer as advisory; prefer review.",
                )
            )
        if margin <= thresholds.low_margin:
            findings.append(
                Diagnostic(
                    code="AMBIGUOUS_TOP2",
                    severity="medium",
                    title="Small gap between top two classes",
                    evidence={"margin": round(margin, 4), "threshold": thresholds.low_margin},
                    explanation=(
                        "The top two classes are nearly tied, so a small change in input could "
                        "flip the answer."
                    ),
                    recommended_action="Escalate for a second opinion.",
                )
            )
        if drift_severity in ("moderate", "severe"):
            findings.append(
                Diagnostic(
                    code="EMBEDDING_SHIFT",
                    severity="high" if drift_severity == "severe" else "medium",
                    title="Input distribution differs from reference",
                    evidence={
                        "drift_severity": drift_severity,
                        "drift_score": round(drift_score, 4),
                        "thresholds": {
                            "moderate": thresholds.drift_moderate,
                            "severe": thresholds.drift_severe,
                        },
                    },
                    explanation=(
                        "Recent inputs are statistically distinguishable from the reference "
                        "window in feature or prediction space. Accuracy on shifted data is "
                        "unknown without verified labels."
                    ),
                    recommended_action="Monitor closely; consider reviewing a sample of cases.",
                )
            )
        if quality is not None:
            if quality.sharpness <= thresholds.blur_low_sharpness:
                findings.append(
                    Diagnostic(
                        code="IMAGE_BLUR",
                        severity="medium",
                        title="Low image sharpness",
                        evidence={
                            "laplacian_variance": round(quality.sharpness, 6),
                            "threshold": thresholds.blur_low_sharpness,
                        },
                        explanation=(
                            "Laplacian variance is low, which is consistent with a blurred or "
                            "out-of-focus acquisition. Smooth scenes produce the same signal, so "
                            "this is not a confirmed capture defect."
                        ),
                        recommended_action="Check the acquisition pipeline for this source.",
                    )
                )
            if quality.brightness <= thresholds.brightness_dark:
                findings.append(
                    Diagnostic(
                        code="IMAGE_TOO_DARK",
                        severity="low",
                        title="Low brightness",
                        evidence={
                            "brightness": round(quality.brightness, 4),
                            "threshold": thresholds.brightness_dark,
                        },
                        explanation=(
                            "Mean pixel intensity is far below the reference range, which can "
                            "degrade feature extraction."
                        ),
                        recommended_action="Verify illumination settings for this camera.",
                    )
                )
            elif quality.brightness >= thresholds.brightness_bright:
                findings.append(
                    Diagnostic(
                        code="IMAGE_TOO_BRIGHT",
                        severity="low",
                        title="High brightness",
                        evidence={
                            "brightness": round(quality.brightness, 4),
                            "threshold": thresholds.brightness_bright,
                        },
                        explanation=(
                            "Mean pixel intensity is far above the reference range; "
                            "highlights may be saturated."
                        ),
                        recommended_action="Verify exposure settings for this camera.",
                    )
                )
            if quality.contrast <= thresholds.contrast_low:
                findings.append(
                    Diagnostic(
                        code="LOW_CONTRAST",
                        severity="low",
                        title="Low contrast",
                        evidence={
                            "contrast": round(quality.contrast, 4),
                            "threshold": thresholds.contrast_low,
                        },
                        explanation=(
                            "Pixel intensity variance is low, which reduces discriminative "
                            "detail available to the model."
                        ),
                        recommended_action="Check the sensor or scene contrast.",
                    )
                )

        if disagreement is not None and disagreement >= thresholds.disagreement_high:
            findings.append(
                Diagnostic(
                    code="HIGH_MODEL_DISAGREEMENT",
                    severity="high",
                    title="Models disagree",
                    evidence={
                        "disagreement": round(disagreement, 4),
                        "threshold": thresholds.disagreement_high,
                    },
                    explanation=(
                        "The lightweight and advanced models predict different classes. "
                        "A second opinion is not automatically more correct, so this is treated "
                        "as instability rather than proof that either model is wrong."
                    ),
                    recommended_action="Prefer human review over automatic escalation.",
                )
            )
        if advanced_used and advanced_agrees is False:
            findings.append(
                Diagnostic(
                    code="ESCALATION_DISAGREEMENT",
                    severity="medium",
                    title="Escalation changed the answer",
                    evidence={"advanced_agrees": advanced_agrees},
                    explanation=(
                        "The advanced model produced a different class, and neither model's "
                        "answer can be verified without a label."
                    ),
                    recommended_action="Queue the case for human review.",
                )
            )
        if advanced_failed:
            findings.append(
                Diagnostic(
                    code="ADVANCED_MODEL_FAILURE",
                    severity="critical",
                    title="Escalation failed",
                    evidence={"advanced_failed": True},
                    explanation=(
                        "The advanced model could not be executed, so the decision is based on "
                        "lightweight evidence only."
                    ),
                    recommended_action="Check model availability and latency budget.",
                )
            )
        if drift_severity == "severe" and max_probability > 0.95:
            findings.append(
                Diagnostic(
                    code="CONFIDENT_UNDER_SHIFT",
                    severity="medium",
                    title="High confidence on shifted input",
                    evidence={
                        "drift_severity": drift_severity,
                        "max_probability": round(max_probability, 4),
                    },
                    explanation=(
                        "The model is highly confident while the input distribution is shifted. "
                        "This is the pattern most often associated with silent failures, though "
                        "the prediction may still be correct."
                    ),
                    recommended_action="Prioritise review sampling for this window.",
                )
            )
        findings.sort(key=lambda item: -SEVERITY_ORDER.get(item.severity, 0))
        return findings


def summarise(findings: list[Diagnostic]) -> dict[str, Any]:
    """Compact summary used in API responses and the dashboard."""
    if not findings:
        return {"count": 0, "highest_severity": None, "codes": []}
    return {
        "count": len(findings),
        "highest_severity": findings[0].severity,
        "codes": [finding.code for finding in findings],
    }
