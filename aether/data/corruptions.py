"""Deterministic corruption suite compatible with the CIFAR-10-C protocol.

The original CIFAR-10-C benchmark applies five severities of fifteen corruption families.
This module reimplements ten of those families locally so the robustness study is fully
reproducible without a 2 GB download. Parameters are recorded in the generated manifest;
they are severity-scaled in the spirit of Hendrycks & Dietterich (2019) but are *our* exact
values, and every reported robustness number names the corruption and severity it used.

Unimplemented families (fog, frost, snow, spatter, elastic) are deliberately absent rather
than approximated loosely: a badly matched corruption would make cross-paper comparison
misleading.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image
from scipy import ndimage

CORRUPTION_FAMILIES: tuple[str, ...] = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "gaussian_blur",
    "motion_blur",
    "brightness",
    "contrast",
    "pixelate",
    "jpeg_compression",
)

SEVERITIES: tuple[int, ...] = (1, 2, 3, 4, 5)

# Severity 1..5 parameter tables (documented, fixed, and reproducible).
PARAMETERS: dict[str, tuple[float, ...]] = {
    # standard deviation of additive Gaussian noise in [0, 1] pixel space
    "gaussian_noise": (0.04, 0.07, 0.10, 0.14, 0.18),
    # photon budget: pixel counts are drawn from Poisson(pixel_counts * factor) / factor
    "shot_noise": (60.0, 25.0, 12.0, 5.0, 3.0),
    # fraction of pixels replaced by salt-and-pepper noise
    "impulse_noise": (0.02, 0.04, 0.07, 0.11, 0.16),
    # disk radius for the defocus point-spread function
    "defocus_blur": (1.0, 1.6, 2.2, 3.0, 4.0),
    # sigma of an isotropic Gaussian blur
    "gaussian_blur": (0.5, 0.8, 1.1, 1.5, 2.0),
    # linear motion-blur kernel length in pixels
    "motion_blur": (3.0, 5.0, 7.0, 9.0, 12.0),
    # signed additive brightness delta in [0, 1] pixel space
    "brightness": (0.10, 0.18, 0.26, 0.36, 0.48),
    # multiplicative contrast factor applied about the per-image mean
    "contrast": (0.70, 0.55, 0.40, 0.28, 0.18),
    # relative resolution retained before nearest-neighbour upsampling
    "pixelate": (0.70, 0.55, 0.42, 0.32, 0.22),
    # JPEG quality factor
    "jpeg_compression": (45.0, 34.0, 25.0, 17.0, 10.0),
}


def corruption_parameters(family: str, severity: int) -> float:
    """Return the numeric parameter used for ``(family, severity)``."""
    if family not in PARAMETERS:
        raise ValueError(f"unknown corruption family '{family}'")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")
    return float(PARAMETERS[family][severity - 1])


def _as_float(images: np.ndarray) -> np.ndarray:
    if images.dtype != np.uint8:
        raise ValueError(f"corruption input must be uint8, got {images.dtype}")
    return images.astype(np.float32) / 255.0


def _to_uint8(images: np.ndarray) -> np.ndarray:
    return np.clip(images * 255.0, 0, 255).round().astype(np.uint8)


def _gaussian_noise(x: np.ndarray, param: float, rng: np.random.Generator) -> np.ndarray:
    return x + rng.normal(0.0, param, size=x.shape).astype(np.float32)


def _shot_noise(x: np.ndarray, param: float, rng: np.random.Generator) -> np.ndarray:
    """Photon-count noise: fewer effective photons at higher severity."""
    counts = np.clip(x * 255.0, 0, None) * param
    return (rng.poisson(counts).astype(np.float32) / param) / 255.0


def _impulse_noise(x: np.ndarray, param: float, rng: np.random.Generator) -> np.ndarray:
    out = x.copy()
    mask = rng.random(out.shape[:3]) < param
    salt = rng.random(out.shape[:3]) < 0.5
    out[mask & salt] = 1.0
    out[mask & ~salt] = 0.0
    return out


def _defocus_blur(x: np.ndarray, param: float, rng: np.random.Generator) -> np.ndarray:
    radius = int(round(param))
    size = 2 * radius + 1
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = ((xx**2 + yy**2) <= radius**2).astype(np.float32)
    kernel /= kernel.sum()
    return _convolve(x, kernel)


def _gaussian_blur(x: np.ndarray, param: float, rng: np.random.Generator) -> np.ndarray:
    return ndimage.gaussian_filter(x, sigma=(0, param, param, 0), mode="reflect")


def _motion_blur(x: np.ndarray, param: float, rng: np.random.Generator) -> np.ndarray:
    length = max(1, int(round(param)))
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    kernel /= kernel.sum()
    angle = float(rng.uniform(0, 180))
    kernel = np.asarray(ndimage.rotate(kernel, angle, reshape=False, order=1))
    kernel = np.clip(kernel, 0, None)
    kernel /= max(kernel.sum(), 1e-6)
    return _convolve(x, kernel)


def _brightness(x: np.ndarray, param: float, rng: np.random.Generator) -> np.ndarray:
    return x + param


def _contrast(x: np.ndarray, param: float, rng: np.random.Generator) -> np.ndarray:
    means = x.mean(axis=(1, 2, 3), keepdims=True)
    return (x - means) * param + means


def _pixelate(x: np.ndarray, param: float, rng: np.random.Generator) -> np.ndarray:
    height, width = x.shape[1], x.shape[2]
    small_h = max(1, int(round(height * param)))
    small_w = max(1, int(round(width * param)))
    out = np.empty_like(x)
    for index in range(x.shape[0]):
        small = np.stack(
            [
                np.asarray(
                    Image.fromarray(_to_uint8(x[index, :, :, c]), mode="L").resize(
                        (small_w, small_h), Image.Resampling.BILINEAR
                    )
                )
                for c in range(3)
            ],
            axis=-1,
        )
        out[index] = (
            np.stack(
                [
                    np.asarray(
                        Image.fromarray(small[:, :, c], mode="L").resize(
                            (width, height), Image.Resampling.NEAREST
                        )
                    )
                    for c in range(3)
                ],
                axis=-1,
            ).astype(np.float32)
            / 255.0
        )
    return out


def _jpeg_compression(x: np.ndarray, param: float, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(x)
    quality = int(round(param))
    for index in range(x.shape[0]):
        buffer = io.BytesIO()
        Image.fromarray(_to_uint8(x[index]), mode="RGB").save(
            buffer, format="JPEG", quality=quality
        )
        buffer.seek(0)
        out[index] = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32) / 255.0
    return out


def _convolve(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a 2-D kernel over each image and channel of an NHWC float batch."""
    out = np.empty_like(x)
    for channel in range(3):
        out[..., channel] = ndimage.convolve(
            x[..., channel], kernel, mode="reflect", axes=(1, 2)
        )
    return out


_IMPLEMENTATIONS: dict[str, Callable[[np.ndarray, float, np.random.Generator], np.ndarray]] = {
    "gaussian_noise": _gaussian_noise,
    "shot_noise": _shot_noise,
    "impulse_noise": _impulse_noise,
    "defocus_blur": _defocus_blur,
    "gaussian_blur": _gaussian_blur,
    "motion_blur": _motion_blur,
    "brightness": _brightness,
    "contrast": _contrast,
    "pixelate": _pixelate,
    "jpeg_compression": _jpeg_compression,
}


def apply_corruption(
    images: np.ndarray,
    family: str,
    severity: int,
    seed: int = 0,
) -> np.ndarray:
    """Apply one corruption at one severity to a uint8 NHWC batch deterministically."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected NHWC uint8 images, got shape {images.shape}")
    param = corruption_parameters(family, severity)
    rng = np.random.default_rng(seed)
    corrupted = _IMPLEMENTATIONS[family](_as_float(images), param, rng)
    if corrupted.shape != images.shape:
        raise RuntimeError(f"corruption '{family}' changed the image shape")
    if not np.isfinite(corrupted).all():
        raise RuntimeError(f"corruption '{family}' produced non-finite pixels")
    return _to_uint8(corrupted)


def corruption_manifest() -> dict[str, object]:
    """Describe the exact corruption suite used for reporting."""
    return {
        "families": list(CORRUPTION_FAMILIES),
        "severities": list(SEVERITIES),
        "parameters": {family: list(PARAMETERS[family]) for family in CORRUPTION_FAMILIES},
        "unimplemented_families": ["fog", "frost", "snow", "spatter", "elastic"],
        "reference": "Hendrycks & Dietterich (2019), Benchmarks for Corruption Robustness",
        "note": (
            "Ten families reimplemented locally with documented severity tables; the five "
            "unimplemented families are intentionally omitted rather than approximated."
        ),
    }
