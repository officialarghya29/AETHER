"""Adapter exports."""

from aether.adapters.base import ModelAdapter, ModelMetadata, Prediction
from aether.adapters.torchvision_adapter import TorchvisionAdapter, build_architecture

__all__ = [
    "ModelAdapter",
    "ModelMetadata",
    "Prediction",
    "TorchvisionAdapter",
    "build_architecture",
]
