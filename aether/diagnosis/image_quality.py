"""Image quality indicators used as diagnostic evidence.

These are descriptive measurements, not defect detectors: a low Laplacian variance is
consistent with a blurred acquisition, but it is also consistent with a naturally smooth
scene. The diagnosis engine therefore reports the measurement and a *possible* explanation
rather than a cause.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ImageQuality:
    """Quality measurements for one image (0-1 pixel scale)."""

    width: int
    height: int
    channels: int
    brightness: float
    contrast: float
    sharpness: float
    colourfulness: float
    is_effectively_greyscale: bool

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_image(image: np.ndarray) -> ImageQuality:
    """Measure brightness, contrast, sharpness and colourfulness.

    Accepts ``HWC`` uint8 or float arrays; float input is assumed to be in ``[0, 1]``.
    """
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"expected an HWC image, got shape {array.shape}")
    if array.shape[-1] not in (1, 3):
        raise ValueError(f"expected 1 or 3 channels, got {array.shape[-1]}")
    if array.dtype == np.uint8:
        pixels = array.astype(np.float32) / 255.0
    else:
        pixels = array.astype(np.float32)
        if float(np.min(pixels)) < 0 or float(np.max(pixels)) > 1.0:
            raise ValueError("float images must be scaled to [0, 1]")

    grey = pixels.mean(axis=-1)
    laplacian = ndimage.laplace(grey)
    brightness = float(grey.mean())
    contrast = float(grey.std())
    sharpness = float(laplacian.var())
    if pixels.shape[-1] == 3:
        channel_spread = float(
            np.mean(np.abs(pixels[..., 0] - pixels[..., 1]))
            + np.mean(np.abs(pixels[..., 1] - pixels[..., 2]))
        )
    else:
        channel_spread = 0.0
    return ImageQuality(
        width=int(pixels.shape[1]),
        height=int(pixels.shape[0]),
        channels=int(pixels.shape[-1]),
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        colourfulness=channel_spread,
        is_effectively_greyscale=channel_spread < 0.01,
    )
