"""Vision API abstraction for Gemini."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from PIL import Image

from . import config

logger = logging.getLogger(__name__)

_flash_semaphore: asyncio.Semaphore | None = None
_pro_semaphore: asyncio.Semaphore | None = None


def _get_model_semaphore(model: str) -> asyncio.Semaphore:
    """Return the appropriate semaphore for the given model."""
    global _flash_semaphore, _pro_semaphore
    if "flash" in model.lower():
        if _flash_semaphore is None:
            _flash_semaphore = asyncio.Semaphore(config.FLASH_SEMAPHORE_LIMIT)
        return _flash_semaphore
    else:
        if _pro_semaphore is None:
            _pro_semaphore = asyncio.Semaphore(config.PRO_SEMAPHORE_LIMIT)
        return _pro_semaphore


def get_client(api_key: str) -> genai.Client:
    """Create a Gemini client."""
    return genai.Client(api_key=api_key)


def load_image_part(image_path: Path) -> types.Part:
    """Load an image file as a Gemini Part, resizing if needed."""
    path = Path(image_path)
    data = path.read_bytes()

    # Resize if over limit
    if len(data) > config.MAX_IMAGE_SIZE_BYTES:
        img = Image.open(path)
        quality = 85
        while len(data) > config.MAX_IMAGE_SIZE_BYTES and quality > 20:
            import io
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            data = buf.getvalue()
            quality -= 10

    suffix = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    mime = mime_map.get(suffix, "image/jpeg")

    return types.Part.from_bytes(data=data, mime_type=mime)


def load_images_from_dir(directory: Path, sort: bool = True) -> list[types.Part]:
    """Load all images from a directory as Gemini Parts."""
    d = Path(directory)
    exts = {".jpg", ".jpeg", ".png"}
    files = [f for f in d.iterdir() if f.suffix.lower() in exts]
    if sort:
        files.sort()
    return [load_image_part(f) for f in files]


def image_paths_from_dir(directory: Path, sort: bool = True) -> list[Path]:
    """Get sorted list of image paths from a directory."""
    d = Path(directory)
    exts = {".jpg", ".jpeg", ".png"}
    files = [f for f in d.iterdir() if f.suffix.lower() in exts]
    if sort:
        files.sort()
    return files


def _extract_text(response) -> str:
    """Extract text from a Gemini response, handling thinking models."""
    # Iterate through candidates/parts, skipping thought parts
    text_parts = []
    try:
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    # Skip thought parts from thinking models
                    if hasattr(part, "thought") and part.thought:
                        continue
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
    except (AttributeError, TypeError) as e:
        logger.debug("Error iterating response parts: %s", e)

    if text_parts:
        return "\n".join(text_parts)

    # Try .text as fallback
    try:
        if response.text is not None:
            return response.text
    except (AttributeError, ValueError):
        pass

    # Log details about the response for debugging
    finish_reason = None
    try:
        finish_reason = response.candidates[0].finish_reason if response.candidates else None
    except Exception:
        pass
    logger.debug("No text in response. finish_reason=%s", finish_reason)
    raise RuntimeError(f"No text content in Gemini response (finish_reason={finish_reason})")


async def call_vision(
    client: genai.Client,
    model: str,
    prompt: str,
    images: list[types.Part] | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 65536,
) -> str:
    """Call the Gemini vision API with retry logic.

    Returns the text response.
    """
    sem = _get_model_semaphore(model)

    contents: list[Any] = []
    if images:
        contents.extend(images)
    contents.append(prompt)

    # Build generation config
    gen_config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(
            thinking_budget=8192,
        ),
    )

    last_error = None
    for attempt in range(config.API_RETRY_ATTEMPTS):
        try:
            async with sem:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=gen_config,
                )
            return _extract_text(response)
        except Exception as e:
            last_error = e
            if attempt < config.API_RETRY_ATTEMPTS - 1:
                delay = config.API_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("Attempt %d failed: %s. Retrying in %ss...", attempt + 1, e, delay)
                await asyncio.sleep(delay)
            else:
                raise RuntimeError(
                    f"Gemini API call failed after {config.API_RETRY_ATTEMPTS} attempts: {last_error}"
                ) from last_error
    return ""  # unreachable


async def detect_content_regions(
    client: genai.Client,
    page_image_path: Path,
    page_number: int,
) -> list[dict]:
    """Detect bounding boxes of student-added content regions on a page.

    Returns list of dicts with keys: label, x_pct, y_pct, w_pct, h_pct, content_type.
    content_type is one of: 'handwriting', 'pasted_image', 'filled_bubbles', 'diagram'.
    All coordinates are percentages of page dimensions.
    """
    image_part = load_image_part(page_image_path)

    prompt = f"""Analyze this student assignment page (page {page_number}) and identify ALL regions
where the student has added their own content. Return bounding boxes as percentages of page
width and height.

Identify these types of student-added content:
- **handwriting**: Equations, text, calculations written by hand
- **pasted_image**: Pasted images of handwritten solutions
- **filled_bubbles**: Colored/filled MCQ bubbles that the student selected (distinct from template)
- **diagram**: Drawn diagrams, graphs, or figures

IMPORTANT: Ignore pre-printed template content (questions, headers, page numbers, pre-marked
answer keys). Only detect regions where the student added something.

Return bounding boxes as percentages (0-100) of the page dimensions:
- x_pct: left edge as % of page width
- y_pct: top edge as % of page height
- w_pct: width as % of page width
- h_pct: height as % of page height

Return ONLY valid JSON:
```json
{{
  "regions": [
    {{
      "label": "Handwritten solution for Q1",
      "x_pct": 10.0,
      "y_pct": 25.0,
      "w_pct": 80.0,
      "h_pct": 30.0,
      "content_type": "handwriting"
    }},
    {{
      "label": "MCQ bubble selection",
      "x_pct": 5.0,
      "y_pct": 60.0,
      "w_pct": 40.0,
      "h_pct": 5.0,
      "content_type": "filled_bubbles"
    }}
  ]
}}
```"""

    try:
        response = await call_vision(
            client, config.FLASH_MODEL, prompt, [image_part], temperature=0.1
        )
        data = extract_json_from_response(response)
        regions = data.get("regions", [])

        # Filter out regions smaller than MIN_REGION_SIZE_PCT by area
        min_area = config.MIN_REGION_SIZE_PCT
        regions = [
            r for r in regions
            if r.get("w_pct", 0) * r.get("h_pct", 0) >= min_area
        ]

        return regions
    except Exception as e:
        logger.debug("Region detection failed for page %d: %s", page_number, e)
        return []


def extract_json_from_response(text: str) -> Any:
    """Extract JSON from a Gemini response, handling markdown code fences."""
    if not text or not text.strip():
        raise ValueError("Empty response text, cannot extract JSON")

    # Try to find JSON in code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # Try to find a JSON object or array in the text
        # Look for the first { or [ and last } or ]
        json_str = text.strip()
        start = -1
        for i, c in enumerate(json_str):
            if c in ('{', '['):
                start = i
                break
        if start >= 0:
            end_char = '}' if json_str[start] == '{' else ']'
            end = json_str.rfind(end_char)
            if end > start:
                json_str = json_str[start:end + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Log a snippet for debugging
        snippet = json_str[:500] if len(json_str) > 500 else json_str
        raise ValueError(f"Failed to parse JSON from response: {e}\nSnippet: {snippet}") from e
