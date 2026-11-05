"""Model adapter contract.

The router, uncertainty engine and API all depend on this interface only, never on a concrete
framework. Supporting a new model family means implementing :class:`ModelAdapter`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from aether.errors import InvalidInputError


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable description of a served model, included in every prediction."""

    model_id: str
    architecture: str
    role: str
    version: str
    num_classes: int
    input_size: int
    device: str
    parameters: int
    classes: tuple[str, ...]
    checkpoint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "architecture": self.architecture,
            "role": self.role,
            "version": self.version,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "device": self.device,
            "parameters": self.parameters,
            "classes": list(self.classes),
            "checkpoint": self.checkpoint,
            **({"extra": self.extra} if self.extra else {}),
        }


@dataclass(frozen=True)
class Prediction:
    """A single-model prediction with logits, probabilities and optional embedding.

    ``latency_ms`` is the per-sample share of the batch's wall-clock inference time.
    """

    model_id: str
    model_version: str
    class_index: int
    class_label: str
    logits: np.ndarray
    probabilities: np.ndarray
    latency_ms: float
    embedding: np.ndarray | None = None

    @property
    def confidence(self) -> float:
        return float(self.probabilities[self.class_index])

    def top_k(self, k: int) -> list[tuple[str, int, float]]:
        order = np.argsort(-self.probabilities)[:k]
        return [(f"class_{int(i)}", int(i), float(self.probabilities[i])) for i in order]


class ModelAdapter(ABC):
    """Common interface for every served model."""

    def __init__(self, metadata: ModelMetadata) -> None:
        self._metadata = metadata

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def input_size(self) -> int:
        return self._metadata.input_size

    @property
    def num_classes(self) -> int:
        return self._metadata.num_classes

    @abstractmethod
    def predict(
        self, batch: torch.Tensor, *, return_embeddings: bool = True
    ) -> list[Prediction]:
        """Predict a normalised NCHW float32 batch of shape ``(N, 3, S, S)``."""

    @abstractmethod
    def warmup(self) -> None:
        """Run one throwaway forward pass so the first real request is not penalised."""

    def validate_batch(self, batch: torch.Tensor) -> None:
        if not isinstance(batch, torch.Tensor):
            raise InvalidInputError(f"expected a torch.Tensor batch, got {type(batch).__name__}")
        if batch.ndim != 4:
            raise InvalidInputError(
                "expected a 4-D NCHW batch", details={"shape": list(batch.shape)}
            )
        if batch.shape[1] != 3:
            raise InvalidInputError(
                "expected 3 colour channels", details={"shape": list(batch.shape)}
            )
        if batch.shape[2] != self.input_size or batch.shape[3] != self.input_size:
            raise InvalidInputError(
                f"expected spatial size {self.input_size}x{self.input_size}",
                details={"shape": list(batch.shape)},
            )
        if batch.shape[0] == 0:
            raise InvalidInputError("empty batch")
        if not torch.isfinite(batch).all():
            raise InvalidInputError("batch contains non-finite values")
