"""Build validated retraining datasets from verified feedback.

Only cases with ``status=resolved`` and a validated label are eligible. The builder records the
exact case ids, the reviewer set and a content digest, so a candidate dataset can be audited and
reproduced. Nothing here retrains or deploys anything: it produces a dataset and a manifest.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aether.errors import ValidationFailedError
from aether.utils.io import atomic_write_json
from aether.utils.logging import get_logger

logger = get_logger("aether.feedback.dataset_builder")


@dataclass(frozen=True)
class CandidateDataset:
    """A reproducible snapshot of verified feedback ready for offline training."""

    dataset_id: str
    images: np.ndarray
    labels: np.ndarray
    case_ids: list[str]
    reviewers: list[str]
    created_at: str
    digest: str
    manifest: dict[str, Any]

    @property
    def size(self) -> int:
        return int(self.labels.size)


def _digest(images: np.ndarray, labels: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(images, dtype=np.uint8).tobytes())
    hasher.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes())
    return hasher.hexdigest()


def build_candidate_dataset(
    cases: Iterable[dict[str, Any]],
    image_loader,
    *,
    num_classes: int = 10,
    min_samples: int = 10,
    dataset_id: str | None = None,
) -> CandidateDataset:
    """Assemble verified cases into a candidate dataset.

    ``cases`` must contain ``status``, ``verified_label``, ``reviewer``, ``review_id`` and an
    image reference resolvable by ``image_loader``. Ambiguous or rejected cases are skipped and
    counted in the manifest rather than silently included.
    """
    images: list[np.ndarray] = []
    labels: list[int] = []
    used_cases: list[str] = []
    reviewers: set[str] = set()
    skipped: list[dict[str, str]] = []

    for case in cases:
        review_id = str(case.get("review_id", ""))
        if case.get("status") != "resolved":
            skipped.append({"review_id": review_id, "reason": "not resolved"})
            continue
        label = case.get("verified_label")
        reviewer = case.get("reviewer")
        if label is None or not reviewer:
            skipped.append({"review_id": review_id, "reason": "missing label or reviewer"})
            continue
        label_value = int(label)
        if not 0 <= label_value < num_classes:
            skipped.append({"review_id": review_id, "reason": "label outside class space"})
            continue
        image_ref = case.get("image_ref") or case.get("image_path")
        if not image_ref:
            skipped.append({"review_id": review_id, "reason": "no image reference"})
            continue
        try:
            image = np.asarray(image_loader(image_ref), dtype=np.uint8)
        except Exception as exc:  # unreadable artifact: skip, never fabricate pixels
            skipped.append({"review_id": review_id, "reason": f"image unavailable: {exc}"})
            continue
        if image.ndim != 3 or image.shape[-1] != 3:
            skipped.append({"review_id": review_id, "reason": "image is not HWC RGB"})
            continue
        images.append(image)
        labels.append(label_value)
        used_cases.append(review_id)
        reviewers.add(str(reviewer))

    if len(images) < min_samples:
        raise ValidationFailedError(
            f"only {len(images)} verified cases available; {min_samples} required to build a dataset",
            details={"skipped": len(skipped)},
        )

    image_array = np.stack(images).astype(np.uint8)
    label_array = np.asarray(labels, dtype=np.int64)
    digest = _digest(image_array, label_array)
    identifier = dataset_id or f"feedback-{digest[:12]}"
    counts = np.bincount(label_array, minlength=num_classes).tolist()
    manifest = {
        "dataset_id": identifier,
        "size": int(label_array.size),
        "class_counts": counts,
        "reviewers": sorted(reviewers),
        "case_ids": used_cases,
        "skipped": skipped,
        "digest": digest,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Verified human feedback only. Candidate data must pass offline evaluation and "
        "explicit approval before promotion.",
    }
    logger.info(
        "candidate dataset built",
        extra={"dataset_id": identifier, "size": int(label_array.size), "skipped": len(skipped)},
    )
    return CandidateDataset(
        dataset_id=identifier,
        images=image_array,
        labels=label_array,
        case_ids=used_cases,
        reviewers=sorted(reviewers),
        created_at=str(manifest["created_at"]),
        digest=digest,
        manifest=manifest,
    )


def persist_candidate(dataset: CandidateDataset, directory: Path) -> dict[str, str]:
    """Write images, labels and a manifest; never commit these to source control."""
    directory.mkdir(parents=True, exist_ok=True)
    images_path = directory / f"{dataset.dataset_id}_images.npy"
    labels_path = directory / f"{dataset.dataset_id}_labels.npy"
    manifest_path = directory / f"{dataset.dataset_id}.json"
    np.save(images_path, dataset.images)
    np.save(labels_path, dataset.labels)
    payload = asdict(dataset) | {"images": None, "labels": None}
    payload["manifest"] = dataset.manifest
    payload["files"] = {"images": images_path.name, "labels": labels_path.name}
    atomic_write_json(manifest_path, payload)
    return {
        "images": str(images_path),
        "labels": str(labels_path),
        "manifest": str(manifest_path),
    }
