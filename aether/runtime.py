"""Runtime device and thread configuration.

CPU-only operation is a first-class requirement, so device selection never hardcodes CUDA.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, cast

import torch

from aether.config import RuntimeConfig
from aether.utils.logging import get_logger

logger = get_logger("aether.runtime")


@dataclass(frozen=True)
class RuntimeInfo:
    device: torch.device
    device_name: str
    torch_version: str
    cuda_available: bool
    threads: int

    def to_dict(self) -> dict[str, object]:
        return {
            "device": str(self.device),
            "device_name": self.device_name,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "threads": self.threads,
        }


def resolve_device(setting: str = "auto") -> torch.device:
    """Resolve ``auto|cpu|cuda`` to a concrete device, failing loudly when CUDA is absent."""
    if setting == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device is 'cuda' but CUDA is not available")
        return torch.device("cuda")
    if setting == "cpu":
        return torch.device("cpu")
    if setting not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unknown device setting '{setting}'")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PyTorch accepts an inter-op thread count once per process and refuses after parallel work has
# started, so the value is tracked here instead of being re-applied per pipeline construction.
_INTEROP_THREADS: int | None = None


def configure_runtime(runtime: RuntimeConfig) -> RuntimeInfo:
    """Apply thread configuration and return a description of the execution environment."""
    device = resolve_device(runtime.device)
    threads = runtime.num_threads
    torch.set_num_threads(threads)

    if device.type == "cpu":
        # Keep inter-op parallelism modest: DataLoader workers need cores too.
        global _INTEROP_THREADS
        target = max(1, min(4, threads // 2))
        if target != _INTEROP_THREADS:
            try:
                torch.set_num_interop_threads(target)
            except RuntimeError as error:
                logger.warning(
                    "inter-op thread count could not be applied",
                    extra={"requested": target, "reason": str(error)},
                )
            _INTEROP_THREADS = target

    backends = cast(Any, torch.backends)
    if hasattr(backends, "mkldnn"):
        backends.mkldnn.enabled = True

    name = "cpu"
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
    else:
        name = os.environ.get("AETHER_CPU_NAME", "cpu")

    return RuntimeInfo(
        device=device,
        device_name=name,
        torch_version=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
        threads=threads,
    )
