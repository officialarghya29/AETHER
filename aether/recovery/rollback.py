"""Model rollback.

Rollback is deliberately one call: an operator under incident pressure should not have to
reconstruct a version graph. The registry records who rolled back what, and the service reloads
the previous checkpoint on the next request.
"""

from __future__ import annotations

from aether.recovery.registry import ModelRegistryStore
from aether.utils.logging import get_logger

logger = get_logger("aether.recovery.rollback")


def rollback_model(
    store: ModelRegistryStore, *, role: str, actor: str, target_version: str | None = None
) -> dict:
    """Roll back to an explicit version, or to the most recent retired version."""
    if not target_version:
        retired = [entry for entry in store.versions(role) if entry["status"] == "retired"]
        if not retired:
            raise ValueError(f"no retired {role} version is available to roll back to")
        target_version = max(retired, key=lambda entry: entry["registered_at"])["version"]
    active = store.rollback(role, target_version, actor=actor)
    return {
        "role": role,
        "active_version": active,
        "actor": actor,
        "note": "Reload the model registry (or restart the service) to serve the restored version.",
    }
