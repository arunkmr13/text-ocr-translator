# ocr_service.py
# Uses Google Gemini Vision API (free tier) instead of Claude.
# Free tier: 15 requests/min, 1500 requests/day — no credit card needed.
# Get API key at: https://aistudio.google.com/app/apikey
#
# Key change from original:
#   - Prompt now requests normalized bounding boxes (0.0–1.0) per region
#   - TextRegion now always has a BoundingBox (converted from normalized coords)
#   - Everything else unchanged

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

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
]


# ---------------------------------------------------------------------------
# Data models (unchanged public API)
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
    extracted_text: str
    translated_text: str
    language: LanguageResult | None
    regions: list[TextRegion] = field(default_factory=list)
    structured_data: dict[str, Any] = field(default_factory=dict)
    warning: str | None = None


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image_bytes(image: np.ndarray | str | Path) -> tuple[bytes, str, int, int]:
    """
    Load image as raw bytes for Gemini SDK.
    Returns: (image_bytes, mime_type, image_width, image_height)
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
            data = f.read()

        # Read dimensions via OpenCV
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w = (img.shape[:2] if img is not None else (0, 0))
        return data, mime_type, w, h

    elif isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Failed to encode OpenCV image to JPEG")
        return buffer.tobytes(), "image/jpeg", w, h

    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


# ---------------------------------------------------------------------------
# Prompt builder — now requests normalized bounding boxes
# ---------------------------------------------------------------------------

def _build_prompt(language_result: LanguageResult | None) -> str:
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
2. Translate ALL extracted text to English.
3. For each distinct text region (title, label, table cell, paragraph, etc.), provide:
   - The original text
   - The English translation
   - A bounding box as normalized coordinates (values between 0.0 and 1.0):
     * x: left edge / image width
     * y: top edge / image height
     * w: region width / image width
     * h: region height / image height
4. Identify any structured/tabular data (numbers, labels, categories, dates).
5. Return ONLY a valid JSON object — no explanation, no markdown, no code fences.

JSON schema to follow exactly:
{{
  "extracted_text": "<full original text as single string>",
  "translated_text": "<full English translation as single string>",
  "detected_language": "<ISO 639-1 code>",
  "regions": [
    {{
      "original": "<original text for this region>",
      "translated": "<English translation for this region>",
      "bbox": {{
        "x": 0.0,
        "y": 0.0,
        "w": 1.0,
        "h": 0.05
      }}
    }}
  ],
  "structured_data": {{
    "<key>": "<value>"
  }}
}}

Important for bounding boxes:
- Be as accurate as possible — these are used to position translated text directly
  over the original text in the image.
- Each region should correspond to a visually distinct text element.
- For tables, provide one region per cell or per row — whichever gives better granularity.
- All coordinate values MUST be between 0.0 and 1.0.

For structured_data: extract key metrics, dates, names, numbers, or table data.
Use snake_case keys. Example: {{"date": "2025-05-27", "new_cases": 25, "total_cases": 11629}}

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
    last_error = None

    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
            )
            if not response.text:
                raise ValueError("Gemini returned empty response")
            logger.info("Gemini call succeeded with model: %s", model)
            logger.debug("Gemini raw response preview: %s", response.text[:200])
            return response.text
        except Exception as e:
            if any(x in str(e) for x in ["503", "404", "UNAVAILABLE", "NOT_FOUND", "quota"]):
                logger.warning("Model %s unavailable, trying next... (%s)", model, e)
                last_error = e
                continue
            raise # Non-503 errors (bad key, invalid request, etc.) raise immediately

    raise RuntimeError(
        f"All Gemini models unavailable. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned non-JSON response: {e}\nRaw: {raw[:500]}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_and_translate(
    image: np.ndarray | str | Path,
    language_result: LanguageResult | None = None,
    api_key: str = "",
) -> OCRResult:
    """
    Primary entry point. Uses Gemini Vision API.

    Args:
        image:           OpenCV ndarray, file path str, or Path object
        language_result: Output from language_service.detect_language()
        api_key:         Gemini API key

    Returns:
        OCRResult with extracted_text, translated_text, regions (with bounding boxes),
        and structured_data
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. "
            "Get a free key at https://aistudio.google.com/app/apikey "
            "and add it to your .env file as GEMINI_API_KEY=your_key_here"
        )

    # 1. Load image bytes + get pixel dimensions for bbox denormalization
    image_bytes, mime_type, img_w, img_h = _load_image_bytes(image)
    logger.info("Image loaded: mime_type=%s size=%d bytes w=%d h=%d",
                mime_type, len(image_bytes), img_w, img_h)

    # 2. Build language-aware prompt (now includes bbox instructions)
    prompt = _build_prompt(language_result)

    # 3. Single Gemini Vision call
    try:
        raw_response = _call_gemini_vision(image_bytes, mime_type, prompt, api_key)
    except Exception as e:
        logger.error("Gemini Vision call failed: %s", e)
        raise

    logger.info("Gemini response received (%d chars)", len(raw_response))

    # 4. Parse JSON
    data = _parse_response(raw_response)

    # 5. Build typed OCRResult — denormalize bounding boxes to pixel coords
    regions: list[TextRegion] = []
    for r in (data.get("regions") or []):
        bbox = None
        raw_bbox = r.get("bbox")
        if raw_bbox and img_w > 0 and img_h > 0:
            try:
                nx = float(raw_bbox.get("x", 0))
                ny = float(raw_bbox.get("y", 0))
                nw = float(raw_bbox.get("w", 1))
                nh = float(raw_bbox.get("h", 0.05))
                # Clamp to [0, 1]
                nx, ny = max(0.0, min(nx, 1.0)), max(0.0, min(ny, 1.0))
                nw, nh = max(0.01, min(nw, 1.0)), max(0.01, min(nh, 1.0))
                bbox = BoundingBox(
                    x=int(nx * img_w),
                    y=int(ny * img_h),
                    width=int(nw * img_w),
                    height=int(nh * img_h),
                )
            except (TypeError, ValueError) as e:
                logger.warning("Could not parse bbox for region '%s': %s",
                               r.get("original", "")[:30], e)

        regions.append(TextRegion(
            original_text=r.get("original", ""),
            translated_text=r.get("translated", ""),
            bounding_box=bbox,
        ))

    logger.info("Parsed %d regions (%d with bounding boxes)",
                len(regions), sum(1 for r in regions if r.bounding_box))

    return OCRResult(
        extracted_text=data.get("extracted_text") or "",
        translated_text=data.get("translated_text") or "",
        language=language_result,
        regions=regions,
        structured_data=data.get("structured_data") or {},
        warning=None,
    )