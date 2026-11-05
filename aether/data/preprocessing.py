"""Deterministic, batch-level preprocessing shared by training, evaluation and serving.

All three paths call :func:`prepare_model_input`, which is the only place pixel scaling,
resizing and normalisation are defined. Divergent preprocessing between training and serving
is one of the most common silent reliability failures, so it is structurally prevented here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from aether.data.cifar import CIFAR10_MEAN, CIFAR10_STD


def normalization_tensors(device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Return the (mean, std) tensors shaped for broadcasting against NCHW input."""
    mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return mean, std


def images_to_tensor(images: np.ndarray) -> torch.Tensor:
    """Convert uint8 NHWC images to a float32 NCHW tensor in [0, 1] (no normalisation)."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected NHWC uint8 images, got shape {images.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(images))
    return tensor.permute(0, 3, 1, 2).float().div_(255.0)


def resize_batch(batch: torch.Tensor, image_size: int) -> torch.Tensor:
    """Bilinearly resize a float NCHW batch to the model's expected spatial size."""
    if batch.shape[-1] == image_size and batch.shape[-2] == image_size:
        return batch
    return F.interpolate(batch, size=(image_size, image_size), mode="bilinear", antialias=True)


def normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    """Apply CIFAR-10 channel normalisation."""
    mean, std = normalization_tensors(batch.device)
    return (batch - mean) / std


def augment_batch(
    batch: torch.Tensor, generator: torch.Generator, padding: int = 4, flip: bool = True
) -> torch.Tensor:
    """Reflect-pad, random-crop and random horizontal flip a training batch.

    Batch-level augmentation keeps the DataLoader workers free of per-sample Python overhead,
    which matters on CPU-only hardware.
    """
    padded = F.pad(batch, (padding, padding, padding, padding), mode="reflect")
    size = batch.shape[-1]
    max_offset = 2 * padding
    if max_offset > 0:
        offsets = torch.randint(
            0, max_offset + 1, (2,), generator=generator, dtype=torch.int64
        ).tolist()
        top, left = int(offsets[0]), int(offsets[1])
        padded = padded[..., top : top + size, left : left + size]
    if flip:
        mask = torch.rand(batch.shape[0], generator=generator) < 0.5
        if mask.any():
            padded = padded.clone()
            padded[mask] = torch.flip(padded[mask], dims=[3])
    return padded


def prepare_model_input(
    images: np.ndarray | torch.Tensor,
    image_size: int,
    *,
    train: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Full preprocessing path from raw uint8 NHWC images to normalised NCHW float32."""
    if isinstance(images, np.ndarray):
        batch = images_to_tensor(images)
    elif isinstance(images, torch.Tensor):
        batch = images.float()
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError(f"expected NCHW float tensor, got shape {tuple(batch.shape)}")
        if batch.max() > 1.5:  # tolerate uint8-valued tensors
            batch = batch.div_(255.0)
    else:
        raise TypeError(f"unsupported image container: {type(images).__name__}")

    batch = resize_batch(batch, image_size)
    if train:
        if generator is None:
            raise ValueError("a torch.Generator is required for training augmentation")
        batch = augment_batch(batch, generator)
    return normalize_batch(batch).contiguous()


class CifarSplitDataset(torch.utils.data.Dataset):
    """Dataset view over a uint8 NHWC array restricted to precomputed split indices.

    Returns ``(uint8 CHW tensor, label)``; resizing and normalisation happen on the batch in
    :func:`prepare_model_input` so workers stay cheap and numerics stay identical everywhere.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> None:
        self._images = images
        self._labels = labels
        self._indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        index = int(self._indices[item])
        image = torch.from_numpy(self._images[index].copy()).permute(2, 0, 1)
        return image, int(self._labels[index])
