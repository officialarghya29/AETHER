"""Model loading and registry.

The registry owns exactly one adapter instance per role and is the only component allowed to
load weights. Models load lazily so a CPU deployment never pays for the advanced model unless
the router escalates.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch

from aether.adapters.base import ModelAdapter
from aether.adapters.torchvision_adapter import TorchvisionAdapter
from aether.config import AetherConfig, ModelSpec
from aether.errors import ModelNotAvailableError
from aether.utils.logging import get_logger

logger = get_logger("aether.models.registry")


def spec_to_adapter(
    spec: ModelSpec,
    *,
    config: AetherConfig,
    device: torch.device,
) -> TorchvisionAdapter:
    """Instantiate an adapter for a configured model spec."""
    checkpoint = config.resolve(spec.checkpoint) if spec.checkpoint else None
    if checkpoint is None:
        raise ModelNotAvailableError(f"no checkpoint configured for role '{spec.role}'")
    adapter = TorchvisionAdapter(
        role=spec.role,
        architecture=spec.architecture,
        version=spec.version,
        checkpoint=checkpoint,
        input_size=spec.input_size or config.data.image_size,
        num_classes=spec.num_classes,
        device=device,
        max_batch_size=config.runtime.max_batch_size,
    )
    logger.info(
        "model loaded",
        extra={
            "model_id": adapter.metadata.model_id,
            "parameters": adapter.metadata.parameters,
            "device": str(device),
        },
    )
    return adapter


class ModelRegistry:
    """Thread-safe lazy registry of adapters keyed by role."""

    def __init__(self, config: AetherConfig, device: torch.device) -> None:
        self._config = config
        self._device = device
        self._specs: dict[str, ModelSpec] = {
            "lightweight": config.models.lightweight,
            "advanced": config.models.advanced,
        }
        if config.models.optional is not None:
            self._specs["optional"] = config.models.optional
        self._adapters: dict[str, TorchvisionAdapter] = {}
        self._failures: dict[str, str] = {}
        self._lock = threading.RLock()

    def available_roles(self) -> list[str]:
        return ["lightweight", "advanced"]

    def get(self, role: str) -> ModelAdapter:
        """Return (and lazily load) the adapter for ``role``."""
        with self._lock:
            if role in self._adapters:
                return self._adapters[role]
            if role in self._failures:
                raise ModelNotAvailableError(
                    f"model '{role}' previously failed to load: {self._failures[role]}"
                )
            spec = self._specs.get(role)
            if spec is None:
                raise ModelNotAvailableError(f"no model configured for role '{role}'")
            try:
                adapter = spec_to_adapter(spec, config=self._config, device=self._device)
            except Exception as exc:
                self._failures[role] = str(exc)
                logger.error("model load failed", extra={"role": role, "error": str(exc)})
                raise
            self._adapters[role] = adapter
            return adapter

    def try_get(self, role: str) -> ModelAdapter | None:
        """Return the adapter or ``None`` when unavailable (used for degraded operation)."""
        try:
            return self.get(role)
        except Exception:
            return None

    def is_loaded(self, role: str) -> bool:
        return role in self._adapters

    def describe(self) -> list[dict[str, Any]]:
        """Describe configured models, loading nothing."""
        descriptions: list[dict[str, Any]] = []
        for role in ("lightweight", "advanced"):
            spec = self._specs[role]
            checkpoint = self._config.resolve(spec.checkpoint)
            entry: dict[str, Any] = {
                "role": role,
                "architecture": spec.architecture,
                "version": spec.version,
                "num_classes": spec.num_classes,
                "checkpoint_present": Path(checkpoint).exists(),
                "loaded": role in self._adapters,
            }
            if role in self._adapters:
                entry.update(self._adapters[role].metadata.to_dict())
            descriptions.append(entry)
        return descriptions

    def warmup_all(self, roles: tuple[str, ...] = ("lightweight",)) -> None:
        for role in roles:
            adapter = self.try_get(role)
            if adapter is not None:
                adapter.warmup()  # type: ignore[attr-defined]

    def close(self) -> None:
        with self._lock:
            for adapter in self._adapters.values():
                adapter.close()  # type: ignore[attr-defined]
            self._adapters.clear()
