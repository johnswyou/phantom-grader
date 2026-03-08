"""PDF to image conversion utilities."""

from __future__ import annotations

from pathlib import Path

import pymupdf  # PyMuPDF

from .. import config


def pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = config.DEFAULT_PDF_DPI) -> list[Path]:
    """Convert a PDF to a directory of JPEG page images.

    Returns list of output image paths, sorted by page number.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    paths = []
    for i, page in enumerate(doc):
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        out_path = output_dir / f"page_{i+1:04d}.jpg"
        pix.save(str(out_path))
        paths.append(out_path)
    doc.close()
    return paths
