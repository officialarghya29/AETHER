"""Safe image decoding for untrusted uploads.

Guards, in order: byte-size limit, decompression-bomb limit, format allow-list, pixel-count
limit, and explicit re-encoding through RGB. The client's filename is never used for anything,
and nothing uploaded is ever executed or written with an attacker-chosen name.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from aether.errors import InvalidInputError

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}
MAX_PIXELS = 4_096_000


@dataclass(frozen=True)
class DecodedImage:
    """A decoded image with its verified metadata."""

    pixels: np.ndarray  # uint8 HWC RGB
    width: int
    height: int
    format: str
    original_bytes: int

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "bytes": self.original_bytes,
        }


def decode_image(
    payload: bytes,
    *,
    max_bytes: int = 5_242_880,
    max_pixels: int = MAX_PIXELS,
) -> DecodedImage:
    """Decode an uploaded image or raise :class:`InvalidInputError` with a clear reason."""
    if not payload:
        raise InvalidInputError("uploaded file is empty")
    if len(payload) > max_bytes:
        raise InvalidInputError(
            "uploaded file exceeds the size limit",
            details={"bytes": len(payload), "limit": max_bytes},
        )

    # ``Image.open`` is lazy: verify() runs the parser without decoding pixel data.
    try:
        with Image.open(io.BytesIO(payload)) as probe:
            image_format = (probe.format or "").upper()
            width, height = probe.size
            probe.verify()
    except Exception as exc:  # PIL raises a variety of parser errors
        raise InvalidInputError("file is not a readable image") from exc

    if image_format not in ALLOWED_FORMATS:
        raise InvalidInputError(
            f"unsupported image format '{image_format.lower() or 'unknown'}'",
            details={"allowed": sorted(ALLOWED_FORMATS)},
        )
    if width <= 0 or height <= 0:
        raise InvalidInputError("image has invalid dimensions")
    if width * height > max_pixels:
        raise InvalidInputError(
            "image resolution exceeds the pixel limit",
            details={"pixels": width * height, "limit": max_pixels},
        )

    try:
        with Image.open(io.BytesIO(payload)) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            pixels = np.asarray(rgb, dtype=np.uint8)
    except Exception as exc:
        raise InvalidInputError("image could not be decoded to RGB") from exc
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise InvalidInputError("decoded image is not RGB")
    return DecodedImage(
        pixels=np.ascontiguousarray(pixels),
        width=int(pixels.shape[1]),
        height=int(pixels.shape[0]),
        format=image_format,
        original_bytes=len(payload),
    )


def centre_crop_square(pixels: np.ndarray) -> np.ndarray:
    """Centre-crop to a square so non-square uploads are not distorted arbitrarily."""
    height, width = pixels.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return pixels[top : top + side, left : left + side]
