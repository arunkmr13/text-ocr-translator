# ocr_service.py
# Uses Google Gemini Vision API (free tier) instead of Claude.
# Free tier: 15 requests/min, 1500 requests/day — no credit card needed.
# Get API key at: https://aistudio.google.com/app/apikey
#
# Everything unchanged from Claude version:
#   - Data models (BoundingBox, TextRegion, OCRResult)
#   - Image loading
#   - Prompt builder
#   - Response parser
#   - Public extract_and_translate() signature
#
# Only changed:
#   - _call_claude_vision() → _call_gemini_vision()
#   - API key env var: GEMINI_API_KEY
#   - Removed: httpx, base64 (Gemini SDK handles encoding internally)
#   - Added: google-generativeai SDK

from __future__ import annotations
import json
import logging
import re
import os
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
from google import genai
from google.genai import types

from backend.services.language_service import LanguageResult

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini config
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash"  # Free tier, vision-capable, fast


# ---------------------------------------------------------------------------
# Data models (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass
class TextRegion:
    """A single detected text region with its translation."""
    original_text: str
    translated_text: str
    bounding_box: BoundingBox | None = None


@dataclass
class OCRResult:
    extracted_text: str                         # Full raw extracted text (source language)
    translated_text: str                        # Full translated text (English)
    language: LanguageResult | None             # Language metadata from language_service
    regions: list[TextRegion] = field(default_factory=list)
    structured_data: dict[str, Any] = field(default_factory=dict)
    warning: str | None = None


# ---------------------------------------------------------------------------
# Image loading
# Gemini SDK accepts raw bytes directly — no manual base64 encoding needed.
# ---------------------------------------------------------------------------

def _load_image_bytes(image: np.ndarray | str | Path) -> tuple[bytes, str]:
    """
    Load image as raw bytes for Gemini SDK.
    Returns: (image_bytes, mime_type)
    """
    if isinstance(image, (str, Path)):
        path = Path(image)
        mime_type = {
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png":  "image/png",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/jpeg")

        with open(path, "rb") as f:
            return f.read(), mime_type

    elif isinstance(image, np.ndarray):
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Failed to encode OpenCV image to JPEG")
        return buffer.tobytes(), "image/jpeg"

    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


# ---------------------------------------------------------------------------
# Prompt builder (unchanged)
# ---------------------------------------------------------------------------

def _build_prompt(language_result: LanguageResult | None) -> str:
    """
    Build a dynamic extraction prompt based on detected language.
    If language is unknown, Gemini is instructed to auto-detect.
    """
    lang_hint = (
        f"The image contains text in {language_result.language_name} "
        f"({language_result.iso_code})."
        if language_result and language_result.confidence == "high"
        else "The image may contain text in any language — auto-detect the language."
    )

    return f"""
{lang_hint}

Please do the following:
1. Extract ALL text visible in the image exactly as it appears (preserve original script).
2. Translate the extracted text to English.
3. Identify any structured/tabular data (numbers, labels, categories, dates).
4. Return ONLY a valid JSON object — no explanation, no markdown, no code fences.

JSON schema to follow exactly:
{{
  "extracted_text": "<full original text as single string>",
  "translated_text": "<full English translation as single string>",
  "detected_language": "<ISO 639-1 code>",
  "regions": [
    {{
      "original": "<original text for this region>",
      "translated": "<English translation for this region>"
    }}
  ],
  "structured_data": {{
    "<key>": "<value>"
  }}
}}

For structured_data: extract any key metrics, dates, names, numbers, or table data
that appear in the image. Use snake_case keys. Example for a health report:
{{"date": "2025-05-27", "new_cases": 25, "total_cases": 11629, "new_deaths": 0}}

If a field has no data, use null. Never omit any key from the schema.
""".strip()


# ---------------------------------------------------------------------------
# Gemini Vision API call
# ---------------------------------------------------------------------------

def _call_gemini_vision(
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    api_key: str,
) -> str:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
    )

    if not response.text:
        raise ValueError("Gemini returned empty response")

    logger.debug("Gemini raw response preview: %s", response.text[:200])
    return response.text


# ---------------------------------------------------------------------------
# Response parser (unchanged)
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> dict[str, Any]:
    """
    Parse Gemini's JSON response robustly.
    Strips markdown fences if present despite instructions not to include them.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned non-JSON response: {e}\nRaw: {raw[:500]}")


# ---------------------------------------------------------------------------
# Public API (signature unchanged — ocr_engine.py needs zero changes)
# ---------------------------------------------------------------------------

def extract_and_translate(
    image: np.ndarray | str | Path,
    language_result: LanguageResult | None = None,
    api_key: str = "",
) -> OCRResult:
    """
    Primary entry point. Uses Gemini Vision API (free tier).

    Args:
        image:           OpenCV ndarray, file path str, or Path object
        language_result: Output from language_service.detect_language()
                         Pass None to let Gemini auto-detect the language
        api_key:         Gemini API key — reads GEMINI_API_KEY env var if empty

    Returns:
        OCRResult with extracted_text, translated_text, regions, structured_data
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. "
            "Get a free key at https://aistudio.google.com/app/apikey "
            "and add it to your .env file as GEMINI_API_KEY=your_key_here"
        )

    # 1. Load image bytes (no base64 encoding — Gemini SDK handles it)
    image_bytes, mime_type = _load_image_bytes(image)
    logger.info("Image loaded: mime_type=%s size=%d bytes", mime_type, len(image_bytes))

    # 2. Build language-aware prompt
    prompt = _build_prompt(language_result)

    # 3. Single Gemini Vision call — OCR + translate + structured extract
    try:
        raw_response = _call_gemini_vision(image_bytes, mime_type, prompt, api_key)
    except Exception as e:
        logger.error("Gemini Vision call failed: %s", e)
        raise

    logger.info("Gemini response received (%d chars)", len(raw_response))

    # 4. Parse structured JSON from response
    data = _parse_response(raw_response)

    # 5. Build typed OCRResult
    regions = [
        TextRegion(
            original_text=r.get("original", ""),
            translated_text=r.get("translated", ""),
        )
        for r in (data.get("regions") or [])
    ]

    return OCRResult(
        extracted_text=data.get("extracted_text") or "",
        translated_text=data.get("translated_text") or "",
        language=language_result,
        regions=regions,
        structured_data=data.get("structured_data") or {},
        warning=None,
    )