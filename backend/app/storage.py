"""Image ingestion.

Two things matter here for annotation correctness:
1. EXIF orientation. Browsers rotate JPEGs according to EXIF, models usually
   don't. If we stored the raw file, boxes drawn in the browser would not line
   up with the pixels a model sees. So rotated images are baked upright on upload.
2. Non-8-bit images (16-bit TIFFs, common in remote sensing and microscopy) can't
   be displayed by browsers. They are stretched to 8 bit (2nd-98th percentile)
   and stored as PNG.
"""
from __future__ import annotations

import io
import uuid
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps, UnidentifiedImageError

from .config import IMAGE_DIR, MAX_UPLOAD_MB

PILImage.MAX_IMAGE_PIXELS = 200_000_000  # large aerial tiles are fine, decompression bombs are not

BROWSER_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
EXIF_ORIENTATION = 0x0112


class InvalidImage(ValueError):
    pass


def _to_8bit(im: PILImage.Image) -> PILImage.Image:
    arr = np.asarray(im).astype(np.float32)
    lo, hi = np.percentile(arr, (2, 98))
    if hi <= lo:
        hi = lo + 1
    arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return PILImage.fromarray(arr, mode="L")


def save_image(data: bytes) -> tuple[str, int, int]:
    """Validate, normalise and store an uploaded image. Returns (stored_name, width, height)."""
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise InvalidImage(f"file larger than {MAX_UPLOAD_MB} MB")
    try:
        im = PILImage.open(io.BytesIO(data))
        im.load()
    except (UnidentifiedImageError, OSError, PILImage.DecompressionBombError) as exc:
        raise InvalidImage(f"not a readable image ({exc.__class__.__name__})") from exc

    fmt = im.format or ""
    orientation = im.getexif().get(EXIF_ORIENTATION, 1)
    keep_raw = fmt in BROWSER_FORMATS and orientation == 1 and im.mode in ("RGB", "RGBA", "L", "P")

    if keep_raw:
        ext = BROWSER_FORMATS[fmt]
        name = f"{uuid.uuid4().hex}{ext}"
        (IMAGE_DIR / name).write_bytes(data)
        return name, im.width, im.height

    im = ImageOps.exif_transpose(im)
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        im = _to_8bit(im)
    elif im.mode not in ("RGB", "RGBA", "L"):
        im = im.convert("RGB")

    if fmt == "JPEG" and im.mode in ("RGB", "L"):
        name = f"{uuid.uuid4().hex}.jpg"
        im.save(IMAGE_DIR / name, format="JPEG", quality=95)
    else:
        name = f"{uuid.uuid4().hex}.png"
        im.save(IMAGE_DIR / name, format="PNG")
    return name, im.width, im.height


def image_path(stored_name: str) -> Path:
    return IMAGE_DIR / stored_name


def delete_file(stored_name: str) -> None:
    try:
        image_path(stored_name).unlink(missing_ok=True)
    except OSError:
        pass
