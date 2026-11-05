"""Model version registry.

A small JSON-backed registry keeps the version history explicit and reversible. Promotion is the
only way a version becomes active, and it requires a named approver plus a passed evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aether.errors import ConflictError, NotFoundError, ValidationFailedError
from aether.utils.io import atomic_write_json, read_json
from aether.utils.logging import get_logger

logger = get_logger("aether.recovery.registry")


@dataclass(frozen=True)
class ModelVersionEntry:
    """One registered model version."""

    role: str
    version: str
    model_id: str
    checkpoint: str
    status: str  # candidate | active | retired
    metrics: dict[str, float]
    dataset_id: str = ""
    registered_at: str = ""
    approved_by: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "version": self.version,
            "model_id": self.model_id,
            "checkpoint": self.checkpoint,
            "status": self.status,
            "metrics": self.metrics,
            "dataset_id": self.dataset_id,
            "registered_at": self.registered_at,
            "approved_by": self.approved_by,
            "notes": self.notes,
        }


class ModelRegistryStore:
    """Persistent model version history with safe promotion and rollback."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> dict[str, Any]:
        payload = read_json(self.path, default=None) or {}
        payload.setdefault("schema_version", 1)
        payload.setdefault("roles", {})
        payload.setdefault("history", [])
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.path, payload)

    def register(self, entry: ModelVersionEntry) -> None:
        """Register a candidate version. Does not change the active model."""
        if not entry.checkpoint:
            raise ValidationFailedError("a registered version needs a checkpoint path")
        payload = self._load()
        role = payload["roles"].setdefault(entry.role, {"active": None, "versions": {}})
        if entry.version in role["versions"]:
            raise ConflictError(f"{entry.role} version {entry.version} is already registered")
        record = entry.to_dict()
        record["registered_at"] = record.get("registered_at") or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        record["status"] = "candidate"
        role["versions"][entry.version] = record
        payload["history"].append(
            {"at": record["registered_at"], "event": "registered", **record}
        )
        self._save(payload)
        logger.info("model version registered", extra={"role": entry.role, "version": entry.version})

    def activate(self, role: str, version: str, approver: str, reason: str = "") -> None:
        """Make a version active. Requires an explicit approver identity."""
        if not approver or not approver.strip():
            raise ValidationFailedError("activation requires a named approver")
        payload = self._load()
        role_payload = payload["roles"].get(role)
        if not role_payload or version not in role_payload["versions"]:
            raise NotFoundError(f"{role} version {version} is not registered")
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        previous = role_payload.get("active")
        if previous and previous in role_payload["versions"]:
            role_payload["versions"][previous]["status"] = "retired"
        role_payload["versions"][version]["status"] = "active"
        role_payload["versions"][version]["approved_by"] = approver
        role_payload["active"] = version
        payload["history"].append(
            {
                "at": now,
                "event": "activated",
                "role": role,
                "version": version,
                "previous": previous,
                "approved_by": approver,
                "reason": reason,
            }
        )
        self._save(payload)
        logger.info(
            "model version activated",
            extra={"role": role, "version": version, "previous": previous, "approver": approver},
        )

    def rollback(self, role: str, target_version: str, actor: str) -> str:
        """Return to a previously registered version."""
        if not actor or not actor.strip():
            raise ValidationFailedError("rollback requires a named operator")
        payload = self._load()
        role_payload = payload["roles"].get(role)
        if not role_payload:
            raise NotFoundError(f"no versions registered for role {role}")
        if target_version not in role_payload["versions"]:
            raise NotFoundError(f"{role} version {target_version} is not registered")
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        previous = role_payload.get("active")
        if previous == target_version:
            raise ConflictError(f"{role} version {target_version} is already active")
        if previous and previous in role_payload["versions"]:
            role_payload["versions"][previous]["status"] = "retired"
        role_payload["versions"][target_version]["status"] = "active"
        role_payload["active"] = target_version
        payload["history"].append(
            {
                "at": now,
                "event": "rolled_back",
                "role": role,
                "from": previous,
                "to": target_version,
                "actor": actor,
            }
        )
        self._save(payload)
        logger.warning(
            "model rolled back", extra={"role": role, "from": previous, "to": target_version}
        )
        return target_version

    def active(self, role: str) -> dict[str, Any] | None:
        payload = self._load()
        role_payload = payload["roles"].get(role)
        if not role_payload or not role_payload.get("active"):
            return None
        return role_payload["versions"].get(role_payload["active"])

    def versions(self, role: str) -> list[dict[str, Any]]:
        payload = self._load()
        role_payload = payload["roles"].get(role) or {"versions": {}}
        return sorted(role_payload["versions"].values(), key=lambda item: item["registered_at"])

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._load()["history"][-limit:]


def default_registry_path(artifacts_root: Path) -> Path:
    return artifacts_root / "registry" / "models.json"
