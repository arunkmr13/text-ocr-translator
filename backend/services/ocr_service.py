# ocr_service.py
# Uses Google Gemini Vision API (free tier) instead of Claude.
# Free tier: 15 requests/min, 1500 requests/day — no credit card needed.
# Get API key at: https://aistudio.google.com/app/apikey

from __future__ import annotations
import json
import logging
import re
import os
import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
from google import genai
from google.genai import types

from backend.services.language_service import LanguageResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FIX #3: Model order corrected for typical free-tier quota exhaustion pattern.
#
# Problem: gemini-2.0-flash-lite hits quota first (lowest RPM on free tier),
# but it was 3rd in the list — so every request burned 2 failed attempts
# + time.sleep(2) each before reaching a working model.
#
# Fix: Put the most reliable free-tier models first.
# gemini-flash-latest  → stable alias, rarely quota-exhausted
# gemini-2.0-flash     → good capacity, primary workhorse
# gemini-2.5-flash     → newest, sometimes rate-limited during peak
# gemini-2.0-flash-lite → lowest quota, last resort
# ---------------------------------------------------------------------------
GEMINI_MODELS = [
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
]

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass
class TextRegion:
    original_text: str
    translated_text: str
    bounding_box: BoundingBox | None = None


@dataclass
class OCRResult:
    extracted_text: str           # Formatted source text (markdown tables)
    translated_text: str          # Formatted English text (markdown tables)
    language: LanguageResult | None
    regions: list[TextRegion] = field(default_factory=list)
    structured_data: dict[str, Any] = field(default_factory=dict)
    warning: str | None = None


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image_bytes(image: np.ndarray | str | Path) -> tuple[bytes, str]:
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


def _load_image_cv2(image: np.ndarray | str | Path) -> tuple[np.ndarray, int, int]:
    """Load image and return (cv2_img, width, height)."""
    if isinstance(image, (str, Path)):
        with open(image, "rb") as f:
            arr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    if img is None:
        return None, 1, 1
    h, w = img.shape[:2]
    return img, w, h


# ---------------------------------------------------------------------------
# Prompt builder
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

Analyze this image carefully and do the following:

1. Extract ALL text visible in the image exactly as it appears (preserve original script).
2. Translate ALL text to English.
3. If the image contains a table, extract it as a proper table structure.
4. Format extracted_text and translated_text as structured markdown:
   - Include title, date, key stats on separate lines
   - Render any tables using markdown pipe format: | col1 | col2 | col3 |
   - Include footer/notes at the end
5. For each text region provide a bounding box as normalized coordinates (0.0-1.0):
   x = left edge / image width, y = top edge / image height
   w = region width / image width, h = region height / image height
6. Return ONLY a valid JSON object — no explanation, no markdown, no code fences.

JSON schema to follow exactly:
{{
  "extracted_text": "<full original text formatted with markdown tables if applicable>",
  "translated_text": "<full English translation formatted with markdown tables if applicable>",
  "detected_language": "<ISO 639-1 code>",
  "regions": [
    {{
      "original": "<original text for this region>",
      "translated": "<English translation for this region>",
      "bbox": {{"x": 0.0, "y": 0.0, "w": 0.1, "h": 0.05}}
    }}
  ],
  "structured_data": {{
    "title": "<document title in English>",
    "date": "<date if present>",
    "summary": {{
      "<metric_name>": "<value>"
    }},
    "tables": [
      {{
        "headers": ["<col1>", "<col2>", "<col3>"],
        "rows": [
          ["<val1>", "<val2>", "<val3>"]
        ]
      }}
    ],
    "footer": "<footer text in English if present>"
  }}
}}

Important formatting rules for extracted_text and translated_text:
- Start with the title on its own line
- Add date on its own line
- Add key statistics as "Label: Value" pairs, one per line
- Render tables using markdown pipe format with header separator row
- Example table format:
  | Province | New Cases | Total Cases | Deaths |
  |----------|-----------|-------------|--------|
  | Name     | 0         | 0           | 0      |
- End with footer/notes if present

Important for bounding boxes:
- Be as accurate as possible — used to position translated text over original
- One region per distinct text element (title, label, table cell, button)
- For tables: one region per cell for best granularity
- All values MUST be between 0.0 and 1.0

If a field has no data, use null. Never omit any key from the schema.
""".strip()


# ---------------------------------------------------------------------------
# FIX #3: Gemini Vision API call — respects retryDelay from 429 responses
# ---------------------------------------------------------------------------

def _parse_retry_delay(error_str: str, default_sleep: float = 2.0) -> float:
    """
    Extract retryDelay from a Gemini 429 error string.
    Google returns e.g. 'retryDelay': '44s' — we parse and honour it,
    capped at 60s so we never block a request worker indefinitely.

    Falls back to default_sleep if no retryDelay found.
    """
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s", error_str)
    if match:
        delay = int(match.group(1))
        capped = min(delay, 60)
        logger.info("Honouring retryDelay=%ss (capped at %ss)", delay, capped)
        return float(capped)
    return default_sleep


def _call_gemini_vision(
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    api_key: str,
) -> str:
    client     = genai.Client(api_key=api_key)
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
            err_str = str(e)
            is_retryable = any(
                x in err_str
                for x in ["503", "429", "404", "UNAVAILABLE", "NOT_FOUND",
                           "quota", "RESOURCE_EXHAUSTED"]
            )
            if is_retryable:
                # FIX #3: Respect Google's retryDelay instead of flat 2s sleep
                delay = _parse_retry_delay(err_str, default_sleep=2.0)
                logger.warning(
                    "Model %s unavailable, sleeping %.0fs before next fallback... (%s)",
                    model, delay, err_str[:120]
                )
                last_error = e
                time.sleep(delay)
                continue
            raise  # Non-retryable error — propagate immediately

    raise RuntimeError(f"All Gemini models unavailable. Last error: {last_error}")


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
# Table formatter
# ---------------------------------------------------------------------------

def _format_as_markdown(data: dict[str, Any], language: str = "en") -> tuple[str, str]:
    sd = data.get("structured_data") or {}

    extracted  = data.get("extracted_text")  or ""
    translated = data.get("translated_text") or ""

    tables = sd.get("tables") or []
    if tables:
        lines = []
        if sd.get("title"):
            lines.append(sd["title"]); lines.append("")
        if sd.get("date"):
            lines.append(f"Date: {sd['date']}"); lines.append("")
        summary = sd.get("summary") or {}
        if summary:
            for k, v in summary.items():
                lines.append(f"{k.replace('_', ' ').title()}: {v}")
            lines.append("")
        for table in tables:
            headers = table.get("headers") or []
            rows    = table.get("rows")    or []
            if headers:
                lines.append("| " + " | ".join(str(h) for h in headers) + " |")
                lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in rows:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
            lines.append("")
        if sd.get("footer"):
            lines.append(sd["footer"])
        if lines:
            translated = "\n".join(lines)

    return extracted, translated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_and_translate(
    image: np.ndarray | str | Path,
    language_result: LanguageResult | None = None,
    api_key: str = "",
) -> OCRResult:
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. "
            "Get a free key at https://aistudio.google.com/app/apikey "
            "and add it to your .env file as GEMINI_API_KEY=your_key_here"
        )

    image_bytes, mime_type = _load_image_bytes(image)
    logger.info("Image loaded: mime_type=%s size=%d bytes", mime_type, len(image_bytes))

    _, img_w, img_h = _load_image_cv2(image)

    prompt = _build_prompt(language_result)

    try:
        raw_response = _call_gemini_vision(image_bytes, mime_type, prompt, api_key)
    except Exception as e:
        logger.error("Gemini Vision call failed: %s", e)
        raise

    logger.info("Gemini response received (%d chars)", len(raw_response))

    data = _parse_response(raw_response)

    extracted_text, translated_text = _format_as_markdown(data)

    regions = []
    for r in (data.get("regions") or []):
        bbox = None
        raw_bbox = r.get("bbox")
        if raw_bbox and img_w > 1 and img_h > 1:
            try:
                nx = max(0.0, min(float(raw_bbox.get("x", 0)), 1.0))
                ny = max(0.0, min(float(raw_bbox.get("y", 0)), 1.0))
                nw = max(0.01, min(float(raw_bbox.get("w", 0.1)), 1.0))
                nh = max(0.01, min(float(raw_bbox.get("h", 0.05)), 1.0))
                bbox = BoundingBox(
                    x=int(nx * img_w),
                    y=int(ny * img_h),
                    width=int(nw * img_w),
                    height=int(nh * img_h),
                )
            except (TypeError, ValueError):
                pass
        regions.append(TextRegion(
            original_text=r.get("original", ""),
            translated_text=r.get("translated", ""),
            bounding_box=bbox,
        ))

    logger.info(
        "Parsed %d regions (%d with bounding boxes)",
        len(regions), sum(1 for r in regions if r.bounding_box)
    )

    return OCRResult(
        extracted_text=extracted_text,
        translated_text=translated_text,
        language=language_result,
        regions=regions,
        structured_data=data.get("structured_data") or {},
        warning=None,
    )