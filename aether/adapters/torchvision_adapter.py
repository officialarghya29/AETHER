"""Torchvision adapter: MobileNetV3-Small (lightweight) and ResNet50 (advanced).

Pretrained ImageNet weights are used for *initialisation during training only*. Inference
strictly requires a project checkpoint: serving ImageNet weights against a CIFAR-10 label space
would silently produce meaningless predictions, so a missing checkpoint is a hard error.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from aether.adapters.base import ModelAdapter, ModelMetadata, Prediction
from aether.data.cifar import CIFAR10_CLASSES
from aether.errors import ModelNotAvailableError

_WEIGHTS: dict[str, tuple[Any, str]] = {}


def _torchvision_factories() -> dict[str, Any]:
    from torchvision import models

    return {
        "mobilenet_v3_small": models.mobilenet_v3_small,
        "mobilenet_v3_large": models.mobilenet_v3_large,
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "efficientnet_b0": models.efficientnet_b0,
    }


def _imagenet_weights(architecture: str) -> Any:
    from torchvision import models

    mapping = {
        "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    }
    if architecture not in mapping:
        raise ValueError(f"no ImageNet weights registered for '{architecture}'")
    return mapping[architecture]


def build_architecture(
    architecture: str, num_classes: int, *, pretrained: bool = False
) -> nn.Module:
    """Instantiate an architecture with a fresh classifier head for ``num_classes``."""
    factories = _torchvision_factories()
    if architecture not in factories:
        raise ValueError(
            f"unsupported architecture '{architecture}'; "
            f"supported: {sorted(factories)}"
        )
    weights = _imagenet_weights(architecture) if pretrained else None
    model: nn.Module = factories[architecture](weights=weights)

    typed = cast(Any, model)
    if architecture.startswith("resnet"):
        head = cast(nn.Linear, typed.fc)
        typed.fc = nn.Linear(head.in_features, num_classes)
    else:  # efficientnet and mobilenet_v3 heads are the last entry of ``classifier``
        classifier = cast(nn.Sequential, typed.classifier)
        head = cast(nn.Linear, classifier[-1])
        classifier[-1] = nn.Linear(head.in_features, num_classes)
    return model


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    """Load a checkpoint payload, converting old-style bare state dicts."""
    if not path.exists():
        raise ModelNotAvailableError(
            f"model checkpoint not found: {path.name}",
            details={"expected": str(path), "remedy": "run training/train.py"},
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # corrupt file or incompatible pickling
        raise ModelNotAvailableError(f"checkpoint could not be read: {path.name}") from exc
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload
    if isinstance(payload, dict):
        return {"state_dict": payload, "metadata": {}}
    raise ModelNotAvailableError(f"unrecognised checkpoint format in {path.name}")


class TorchvisionAdapter(ModelAdapter):
    """Serves a trained torchvision classifier with embedding extraction."""

    def __init__(
        self,
        *,
        role: str,
        architecture: str,
        version: str,
        checkpoint: Path,
        input_size: int,
        num_classes: int = 10,
        classes: tuple[str, ...] = CIFAR10_CLASSES,
        device: torch.device | str = "cpu",
        max_batch_size: int = 64,
    ) -> None:
        self.device = torch.device(device)
        self.max_batch_size = max_batch_size
        payload = load_checkpoint_payload(checkpoint)
        stored = payload.get("metadata", {}) or {}
        architecture = stored.get("architecture", architecture)
        num_classes = int(stored.get("num_classes", num_classes))

        model = build_architecture(architecture, num_classes, pretrained=False)
        try:
            model.load_state_dict(payload["state_dict"], strict=True)
        except RuntimeError as exc:
            raise ModelNotAvailableError(
                f"checkpoint does not match architecture '{architecture}'"
            ) from exc
        model.to(self.device).eval()

        self._embedding: np.ndarray | None = None
        self._model = model
        target = self._pooling_layer(model)
        self._hook = target.register_forward_hook(self._capture_embedding)

        metadata = ModelMetadata(
            model_id=f"{role}:{architecture}@{version}",
            architecture=architecture,
            role=role,
            version=str(stored.get("version", version)),
            num_classes=num_classes,
            input_size=int(stored.get("input_size", input_size)),
            device=str(self.device),
            parameters=int(sum(p.numel() for p in model.parameters())),
            classes=tuple(stored.get("classes", classes)),
            checkpoint=str(checkpoint),
            extra={"val_accuracy": stored.get("val_accuracy")},
        )
        super().__init__(metadata)

    @staticmethod
    def _pooling_layer(model: nn.Module) -> nn.Module:
        if hasattr(model, "avgpool"):
            return model.avgpool  # type: ignore[return-value]
        raise ModelNotAvailableError("architecture does not expose a pooling layer")

    def _capture_embedding(self, module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
        self._embedding = output.detach().flatten(1).cpu().numpy()

    def predict(
        self, batch: torch.Tensor, *, return_embeddings: bool = True
    ) -> list[Prediction]:
        self.validate_batch(batch)
        batch = batch.to(self.device, non_blocking=False)
        predictions: list[Prediction] = []
        for start in range(0, batch.shape[0], self.max_batch_size):
            chunk = batch[start : start + self.max_batch_size]
            embedding: np.ndarray | None
            with torch.inference_mode():
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                logits = self._model(chunk)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                logits_np = logits.detach().float().cpu().numpy()
                if return_embeddings and self._embedding is not None:
                    embedding = self._embedding
                else:
                    embedding = None
                self._embedding = None
            probabilities = _softmax(logits_np)
            per_sample_ms = elapsed_ms / max(1, chunk.shape[0])
            for row in range(logits_np.shape[0]):
                class_index = int(np.argmax(probabilities[row]))
                predictions.append(
                    Prediction(
                        model_id=self.metadata.model_id,
                        model_version=self.metadata.version,
                        class_index=class_index,
                        class_label=self.metadata.classes[class_index],
                        logits=logits_np[row].astype(np.float32),
                        probabilities=probabilities[row].astype(np.float32),
                        latency_ms=per_sample_ms,
                        embedding=(
                            embedding[row].astype(np.float32) if embedding is not None else None
                        ),
                    )
                )
        return predictions

    def warmup(self) -> None:
        dummy = torch.zeros(1, 3, self.input_size, self.input_size, device=self.device)
        with torch.inference_mode():
            self._model(dummy)
        self._embedding = None

    def close(self) -> None:
        self._hook.remove()


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the class axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)
