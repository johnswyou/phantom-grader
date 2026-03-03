"""Image preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

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
