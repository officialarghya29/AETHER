"""Human feedback, dataset building, promotion gates and rollback."""

from __future__ import annotations

import numpy as np
import pytest

from aether.errors import ConflictError, NotFoundError, ValidationFailedError
from aether.feedback.dataset_builder import build_candidate_dataset
from aether.feedback.review import (
    InMemoryReviewQueue,
    ReviewStatus,
    build_case,
    validate_verified_label,
)
from aether.recovery.evaluation import PromotionCriteria, evaluate_candidate
from aether.recovery.promotion import promote_candidate
from aether.recovery.registry import ModelRegistryStore, ModelVersionEntry
from aether.recovery.rollback import rollback_model


def _case(review_id: str = "r-1") -> object:
    return build_case(
        prediction_id=f"p-{review_id}",
        model_id="light:mobilenet@1.0.0",
        predicted_index=4,
        predicted_label="deer",
        confidence=0.41,
        uncertainty={"predictive_entropy": 1.4, "max_probability": 0.41, "prediction_margin": 0.06},
        drift_severity="moderate",
        diagnostics=["HIGH_ENTROPY", "AMBIGUOUS_TOP2"],
    )


def test_reviewer_identity_is_required():
    with pytest.raises(ValidationFailedError):
        validate_verified_label(3, 10, "   ")


def test_label_must_be_inside_class_space():
    with pytest.raises(ValidationFailedError):
        validate_verified_label(11, 10, "qa")
    with pytest.raises(ValidationFailedError):
        validate_verified_label("cat", 10, "qa")
    assert validate_verified_label(7, 10, " qa ")[1] == "qa"


def test_review_queue_lifecycle():
    queue = InMemoryReviewQueue(num_classes=10)
    case = queue.create(_case())
    assert queue.count(ReviewStatus.PENDING.value) == 1

    resolved = queue.submit(case.review_id, label=6, reviewer="qa", notes="checked")
    assert resolved.status == ReviewStatus.RESOLVED.value
    assert resolved.verified_label == 6
    assert resolved.history[-1]["event"] == "resolved"

    with pytest.raises(ConflictError):
        queue.submit(case.review_id, label=5, reviewer="qa")
    with pytest.raises(NotFoundError):
        queue.get("missing")


def test_rejected_case_is_not_resolvable():
    queue = InMemoryReviewQueue(num_classes=10)
    case = queue.create(_case("r-2"))
    queue.reject(case.review_id, reviewer="qa", reason="duplicate upload")
    assert queue.get(case.review_id).status == ReviewStatus.REJECTED.value


def test_candidate_dataset_excludes_unverified_cases():
    images = {
        "uploads/a.img": np.zeros((32, 32, 3), dtype=np.uint8),
        "uploads/b.img": np.ones((32, 32, 3), dtype=np.uint8),
        "uploads/c.img": np.full((32, 32, 3), 2, dtype=np.uint8),
    }
    cases = [
        {"review_id": "1", "status": "resolved", "verified_label": 2, "reviewer": "qa", "image_ref": "uploads/a.img"},
        {"review_id": "2", "status": "resolved", "verified_label": 3, "reviewer": "qa2", "image_ref": "uploads/b.img"},
        {"review_id": "3", "status": "pending", "verified_label": None, "reviewer": None, "image_ref": "uploads/c.img"},
        {"review_id": "4", "status": "resolved", "verified_label": 4, "reviewer": "qa", "image_ref": "missing.img"},
    ]
    dataset = build_candidate_dataset(cases, images.__getitem__, min_samples=2)
    assert dataset.size == 2
    assert dataset.labels.tolist() == [2, 3]
    assert set(dataset.reviewers) == {"qa", "qa2"}
    assert len(dataset.manifest["skipped"]) == 2
    assert len(dataset.digest) == 64


def test_candidate_dataset_requires_minimum_samples():
    with pytest.raises(ValidationFailedError):
        build_candidate_dataset([], lambda _: np.zeros((32, 32, 3), dtype=np.uint8), min_samples=5)


def test_promotion_gate_blocks_regressions():
    approved = evaluate_candidate(
        {"accuracy": 0.80, "ece": 0.05},
        {"accuracy": 0.75, "ece": 0.06},
        samples=500,
        criteria=PromotionCriteria(min_accuracy=0.70, min_accuracy_gain=0.01, max_ece=0.15),
    )
    assert approved.approved is True

    rejected = evaluate_candidate(
        {"accuracy": 0.74, "ece": 0.20},
        {"accuracy": 0.75, "ece": 0.06},
        samples=50,
        criteria=PromotionCriteria(min_accuracy=0.70, min_accuracy_gain=0.01, max_ece=0.15),
    )
    assert rejected.approved is False
    assert len(rejected.reasons) >= 3


def test_registry_promotion_and_rollback(tmp_path):
    store = ModelRegistryStore(tmp_path / "models.json")
    for version, accuracy in (("1.0.0", 0.75), ("1.1.0", 0.80)):
        store.register(
            ModelVersionEntry(
                role="lightweight",
                version=version,
                model_id=f"light:mobilenet@{version}",
                checkpoint=f"artifacts/checkpoints/{version}.pt",
                status="candidate",
                metrics={"accuracy": accuracy},
            )
        )
    store.activate("lightweight", "1.0.0", approver="mle-lead", reason="initial release")
    assert store.active("lightweight")["version"] == "1.0.0"

    evaluation = evaluate_candidate(
        {"accuracy": 0.80, "ece": 0.04}, {"accuracy": 0.75, "ece": 0.05}, samples=500
    )
    promoted = promote_candidate(
        store,
        role="lightweight",
        version="1.1.0",
        evaluation=evaluation,
        approver="mle-lead",
        reason="verified feedback retrain",
    )
    assert promoted["active"]["version"] == "1.1.0"
    assert store.versions("lightweight")[0]["status"] == "retired"

    rolled_back = rollback_model(store, role="lightweight", actor="oncall")
    assert rolled_back["active_version"] == "1.0.0"
    assert [entry["event"] for entry in store.history()][-2:] == ["activated", "rolled_back"]


def test_promotion_requires_approver_and_passing_gate(tmp_path):
    store = ModelRegistryStore(tmp_path / "models.json")
    store.register(
        ModelVersionEntry(
            role="lightweight",
            version="2.0.0",
            model_id="light:mobilenet@2.0.0",
            checkpoint="artifacts/checkpoints/2.0.0.pt",
            status="candidate",
            metrics={},
        )
    )
    failing = evaluate_candidate({"accuracy": 0.50, "ece": 0.3}, {"accuracy": 0.75, "ece": 0.05}, samples=500)
    with pytest.raises(ValidationFailedError):
        promote_candidate(store, role="lightweight", version="2.0.0", evaluation=failing, approver="lead")
    passing = evaluate_candidate({"accuracy": 0.78, "ece": 0.05}, {"accuracy": 0.75, "ece": 0.05}, samples=500)
    with pytest.raises(ValidationFailedError):
        promote_candidate(store, role="lightweight", version="2.0.0", evaluation=passing, approver="  ")
    with pytest.raises(NotFoundError):
        store.activate("lightweight", "9.9.9", approver="lead")
