"""Diagnosis engine and image quality indicators."""

from __future__ import annotations

import numpy as np
import pytest

from aether.diagnosis.image_quality import analyze_image
from aether.diagnosis.rules import DiagnosisEngine, DiagnosticThresholds, summarise


def test_image_quality_measures_a_flat_image():
    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    quality = analyze_image(flat)
    assert quality.brightness == pytest.approx(128 / 255, rel=1e-3)
    assert quality.contrast == pytest.approx(0.0, abs=1e-6)
    assert quality.sharpness == pytest.approx(0.0, abs=1e-9)
    assert quality.is_effectively_greyscale


def test_sharpness_orders_blurred_below_sharp():
    from scipy import ndimage

    rng = np.random.default_rng(0)
    image = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    blurred = ndimage.gaussian_filter(image.astype(np.float64), sigma=(2, 2, 0)).astype(np.uint8)
    assert analyze_image(blurred).sharpness < analyze_image(image).sharpness


def test_image_quality_rejects_bad_shapes():
    with pytest.raises(ValueError):
        analyze_image(np.zeros((32, 32), dtype=np.uint8))
    with pytest.raises(ValueError):
        analyze_image(np.zeros((32, 32, 2), dtype=np.uint8))


def test_image_quality_rejects_unscaled_float():
    with pytest.raises(ValueError):
        analyze_image(np.full((8, 8, 3), 200.0))


def test_high_entropy_produces_diagnostic():
    engine = DiagnosisEngine()
    findings = engine.diagnose(
        entropy=1.9, max_probability=0.3, margin=0.02, predicted_label="cat"
    )
    codes = {finding.code for finding in findings}
    assert "HIGH_ENTROPY" in codes
    assert "LOW_CONFIDENCE" in codes
    assert "AMBIGUOUS_TOP2" in codes
    entropy_finding = next(finding for finding in findings if finding.code == "HIGH_ENTROPY")
    assert entropy_finding.evidence["predictive_entropy_nats"] == pytest.approx(1.9)
    assert "does not by itself imply" in entropy_finding.explanation


def test_confident_clean_prediction_produces_no_findings():
    engine = DiagnosisEngine()
    findings = engine.diagnose(
        entropy=0.05, max_probability=0.98, margin=0.96, predicted_label="ship"
    )
    assert findings == []
    assert summarise(findings) == {"count": 0, "highest_severity": None, "codes": []}


def test_drift_and_blur_are_surfaced():
    engine = DiagnosisEngine()
    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    quality = analyze_image(flat)
    findings = engine.diagnose(
        entropy=0.2,
        max_probability=0.9,
        margin=0.8,
        predicted_label="bird",
        drift_severity="severe",
        drift_score=9.0,
        quality=quality,
    )
    codes = {finding.code for finding in findings}
    assert "EMBEDDING_SHIFT" in codes
    assert "IMAGE_BLUR" in codes
    assert "LOW_CONTRAST" in codes


def test_findings_are_sorted_by_severity():
    engine = DiagnosisEngine()
    findings = engine.diagnose(
        entropy=2.2, max_probability=0.2, margin=0.01, predicted_label="frog",
        drift_severity="severe", drift_score=10.0, advanced_failed=True,
    )
    severities = [finding.severity for finding in findings]
    order = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    assert severities == sorted(severities, key=lambda value: -order[value])


def test_advanced_failure_is_critical():
    engine = DiagnosisEngine()
    findings = engine.diagnose(
        entropy=0.1, max_probability=0.95, margin=0.9, predicted_label="deer",
        advanced_used=True, advanced_failed=True,
    )
    assert findings[0].code == "ADVANCED_MODEL_FAILURE"
    assert findings[0].severity == "critical"


def test_summary_reports_highest_severity():
    engine = DiagnosisEngine()
    findings = engine.diagnose(
        entropy=1.4, max_probability=0.4, margin=0.5, predicted_label="dog",
        drift_severity="severe", drift_score=7.0,
    )
    summary = summarise(findings)
    assert summary["count"] == len(findings)
    assert summary["highest_severity"] in {"high", "critical"}
    assert set(summary["codes"]) <= {finding.code for finding in findings}


def test_thresholds_are_configurable():
    strict = DiagnosisEngine(DiagnosticThresholds(high_entropy=0.1))
    findings = strict.diagnose(entropy=0.2, max_probability=0.9, margin=0.8, predicted_label="cat")
    assert "HIGH_ENTROPY" in {finding.code for finding in findings}
