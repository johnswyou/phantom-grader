"""Configuration for phantom-grader."""

from __future__ import annotations

import os

# Models
FLASH_MODEL = "gemini-3-flash-preview"
PRO_MODEL = "gemini-3.1-pro-preview"

# Rate limiting
MAX_CONCURRENT_STUDENTS = 3
API_SEMAPHORE_LIMIT = 5  # max concurrent API calls
API_RETRY_ATTEMPTS = 3
API_RETRY_BASE_DELAY = 2.0  # seconds

# Image limits
MAX_IMAGE_SIZE_BYTES = 4 * 1024 * 1024  # 4MB Gemini limit


def get_api_key(cli_key: str | None = None) -> str:
    """Get the Gemini API key from CLI arg or env var."""
    key = cli_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "No API key provided. Set GEMINI_API_KEY env var or pass --api-key."
        )
    return key
