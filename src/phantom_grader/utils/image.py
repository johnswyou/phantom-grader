"""Image preprocessing utilities."""

from __future__ import annotations

import io
from pathlib import Path

from google.genai import types
from PIL import Image


def get_image_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) of an image."""
    with Image.open(path) as img:
        return img.size


def ensure_jpeg(path: Path, output_dir: Path | None = None) -> Path:
    """Convert an image to JPEG if it isn't already. Returns path to JPEG."""
    if path.suffix.lower() in (".jpg", ".jpeg"):
        return path
    out_dir = output_dir or path.parent
    out_path = out_dir / f"{path.stem}.jpg"
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.save(out_path, "JPEG", quality=90)
    return out_path


def crop_region(image_path: Path, bbox_pct: dict, pad_pct: float = 5.0) -> bytes:
    """Crop a region from an image and return JPEG bytes.

    Args:
        image_path: Path to the source image.
        bbox_pct: Dict with keys x_pct, y_pct, w_pct, h_pct (percentages of
            image dimensions).
        pad_pct: Padding around the region as a percentage of image dimensions.

    Returns:
        JPEG-encoded bytes of the cropped region.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size

        # Convert percentages to pixels
        x = bbox_pct["x_pct"] / 100.0 * w
        y = bbox_pct["y_pct"] / 100.0 * h
        rw = bbox_pct["w_pct"] / 100.0 * w
        rh = bbox_pct["h_pct"] / 100.0 * h

        # Apply padding
        pad_x = pad_pct / 100.0 * w
        pad_y = pad_pct / 100.0 * h

        # Clamp to image bounds
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(w, x + rw + pad_x)
        bottom = min(h, y + rh + pad_y)

        cropped = img.crop((left, top, right, bottom))

        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def crop_region_to_part(
    image_path: Path, bbox_pct: dict, pad_pct: float = 5.0
) -> types.Part:
    """Crop a region from an image and return a Gemini API Part.

    Same as crop_region but wraps the result in a google.genai types.Part
    ready for API calls.
    """
    data = crop_region(image_path, bbox_pct, pad_pct=pad_pct)
    return types.Part.from_bytes(data=data, mime_type="image/jpeg")
