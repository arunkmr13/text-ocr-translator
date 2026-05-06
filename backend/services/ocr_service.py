# ocr_service.py
# Two-pass Gemini strategy:
#   Pass 1 — full OCR + translation + non-table regions (title, date, stats, headers, footers)
#   Pass 2 — dedicated table-row bbox extraction (all rows, compact JSON)
#
# Design principle: NEVER mutate Gemini bboxes.
# The bbox defines the actual text coverage area — expanding or shifting it
# causes visual overlap with adjacent cells.

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
    region_type: str = "other"


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

def _load_image_bytes(image: np.ndarray | str | Path) -> tuple[bytes, str]:
    if isinstance(image, (str, Path)):
        path = Path(image)
        mime_type = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",  ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/jpeg")
        with open(path, "rb") as f:
            return f.read(), mime_type
    elif isinstance(image, np.ndarray):
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Failed to encode image")
        return buffer.tobytes(), "image/jpeg"
    raise TypeError(f"Unsupported type: {type(image)}")


def _load_image_cv2(image: np.ndarray | str | Path) -> tuple[np.ndarray, int, int]:
    if isinstance(image, (str, Path)):
        with open(image, "rb") as f:
            arr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise TypeError(f"Unsupported type: {type(image)}")
    if img is None:
        return None, 1, 1
    h, w = img.shape[:2]
    return img, w, h


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------

def _parse_retry_delay(err: str, default: float = 2.0) -> float:
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s", err)
    return float(min(int(m.group(1)), 60)) if m else default


def _call_gemini(
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
                raise ValueError("Empty response")
            logger.info("Gemini succeeded: %s (%d chars)", model, len(response.text))
            return response.text
        except Exception as e:
            err = str(e)
            if any(x in err for x in ["503","429","404","UNAVAILABLE","NOT_FOUND","quota","RESOURCE_EXHAUSTED"]):
                delay = _parse_retry_delay(err)
                logger.warning("Model %s unavailable — %.0fs delay", model, delay)
                last_error = e
                time.sleep(delay)
                continue
            raise
    raise RuntimeError(f"All models unavailable. Last: {last_error}")


def _parse_json(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Non-JSON response: {e}\nRaw: {raw[:400]}")


# ---------------------------------------------------------------------------
# Pass 1 prompt
# ---------------------------------------------------------------------------

def _prompt_pass1(language_result: LanguageResult | None) -> str:
    lang_hint = (
        f"The image contains text in {language_result.language_name} ({language_result.iso_code})."
        if language_result and language_result.confidence == "high"
        else "Auto-detect the language of text in this image."
    )
    return f"""
{lang_hint}

You are a precise OCR and translation engine. Return a single JSON object.

TASK:
1. Extract ALL text and translate to English.
2. Return bounding boxes for NON-TABLE elements only:
   title, subtitle, date, stat labels, stat values, table headers, footers.
3. Do NOT include table data rows in regions — they are fetched separately.
4. Format extracted_text and translated_text as markdown with pipe tables.

BBOX RULES:
- Normalised: x=left/W, y=top/H, w=width/W, h=height/H (all 0.0-1.0)
- ONE region per element — never merge multiple elements
- TIGHT fit around the text pixels only
- Stat boxes: TWO regions — label bbox + value bbox separately
- Table headers: one region PER header cell

Return ONLY raw JSON:
{{
  "extracted_text": "<markdown>",
  "translated_text": "<markdown>",
  "detected_language": "<ISO 639-1>",
  "regions": [
    {{
      "original": "<text>",
      "translated": "<English>",
      "type": "<title|subtitle|date|stat_label|stat_value|table_header|footer>",
      "bbox": {{"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}}
    }}
  ],
  "structured_data": {{
    "title": "<English>",
    "date": "<date>",
    "summary": {{"<metric>": "<value>"}},
    "tables": [
      {{"headers": ["<col1>","<col2>"], "rows": [["<v1>","<v2>"]]}}
    ],
    "footer": "<English>"
  }}
}}
""".strip()


# ---------------------------------------------------------------------------
# Pass 2 prompt
# ---------------------------------------------------------------------------

_PROMPT_PASS2 = """
This image contains a data table. Return bounding boxes for EVERY data row cell.

CRITICAL: Return ALL rows — do not stop early or truncate.

For each cell in each data row:
- "o": original text
- "t": English translation
- "b": [x, y, w, h] normalised bbox (0.0-1.0), TIGHT around text only

Return ONLY this JSON:
{
  "rows": [
    {
      "cells": [
        {"o": "<orig>", "t": "<English>", "b": [x, y, w, h]}
      ]
    }
  ]
}
""".strip()


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------

def _format_markdown(data: dict[str, Any]) -> tuple[str, str]:
    sd         = data.get("structured_data") or {}
    extracted  = data.get("extracted_text")  or ""
    translated = data.get("translated_text") or ""
    tables     = sd.get("tables") or []

    if tables:
        lines = []
        if sd.get("title"):   lines += [sd["title"], ""]
        if sd.get("date"):    lines += [f"Date: {sd['date']}", ""]
        for k, v in (sd.get("summary") or {}).items():
            lines.append(f"{k.replace('_',' ').title()}: {v}")
        if sd.get("summary"): lines.append("")
        for tbl in tables:
            hdrs = tbl.get("headers") or []
            rows = tbl.get("rows")    or []
            if hdrs:
                lines.append("| " + " | ".join(str(h) for h in hdrs) + " |")
                lines.append("| " + " | ".join("---" for _ in hdrs) + " |")
            for row in rows:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
            lines.append("")
        if sd.get("footer"): lines.append(sd["footer"])
        if lines: translated = "\n".join(lines)

    return extracted, translated


# ---------------------------------------------------------------------------
# BBox builders
# ---------------------------------------------------------------------------

# Types where wide bbox is expected and should not be rejected
_WIDE_OK = {"title", "subtitle", "footer", "date"}


def _make_bbox(raw: dict, img_w: int, img_h: int, rtype: str = "other") -> BoundingBox | None:
    if not raw:
        return None
    try:
        nx = max(0.0, min(float(raw.get("x", 0)), 1.0))
        ny = max(0.0, min(float(raw.get("y", 0)), 1.0))
        nw = max(0.005, min(float(raw.get("w", 0)), 1.0 - nx))
        nh = max(0.005, min(float(raw.get("h", 0)), 1.0 - ny))
        if nw > 0.45 and rtype not in _WIDE_OK:
            logger.debug("Rejected wide bbox nw=%.3f type=%s", nw, rtype)
            return None
        return BoundingBox(
            x=int(nx * img_w), y=int(ny * img_h),
            width=int(nw * img_w), height=int(nh * img_h),
        )
    except (TypeError, ValueError):
        return None


def _make_bbox_list(coords: list, img_w: int, img_h: int) -> BoundingBox | None:
    if not coords or len(coords) < 4:
        return None
    try:
        nx = max(0.0, min(float(coords[0]), 1.0))
        ny = max(0.0, min(float(coords[1]), 1.0))
        nw = max(0.005, min(float(coords[2]), 1.0 - nx))
        nh = max(0.005, min(float(coords[3]), 1.0 - ny))
        return BoundingBox(
            x=int(nx * img_w), y=int(ny * img_h),
            width=int(nw * img_w), height=int(nh * img_h),
        )
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# IoU deduplication helper
# ---------------------------------------------------------------------------

def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ix = max(0, min(a.x+a.width,  b.x+b.width)  - max(a.x, b.x))
    iy = max(0, min(a.y+a.height, b.y+b.height) - max(a.y, b.y))
    inter = ix * iy
    union = a.width*a.height + b.width*b.height - inter
    return inter / union if union > 0 else 0.0


# Types exempt from IoU deduplication — they are Pass 1 only and should
# never be removed by a Pass 2 cell match
_DEDUP_EXEMPT = {
    "stat_label", "stat_value",
    "table_header", "title", "subtitle", "date", "footer",
}


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
        raise EnvironmentError("GEMINI_API_KEY not set.")

    image_bytes, mime_type = _load_image_bytes(image)
    logger.info("Image: mime=%s size=%d bytes", mime_type, len(image_bytes))

    _, img_w, img_h = _load_image_cv2(image)

    # ── Pass 1 ───────────────────────────────────────────────────────────────
    raw1  = _call_gemini(image_bytes, mime_type, _prompt_pass1(language_result), api_key)
    data1 = _parse_json(raw1)

    extracted_text, translated_text = _format_markdown(data1)

    regions: list[TextRegion] = []
    for r in (data1.get("regions") or []):
        rtype = r.get("type", "other")
        bbox  = _make_bbox(r.get("bbox"), img_w, img_h, rtype)
        regions.append(TextRegion(
            original_text=r.get("original", ""),
            translated_text=r.get("translated", ""),
            bounding_box=bbox,
            region_type=rtype,
        ))

    logger.info("Pass 1: %d regions (%d with bbox)",
                len(regions), sum(1 for r in regions if r.bounding_box))

    # ── Pass 2 ───────────────────────────────────────────────────────────────
    sd     = data1.get("structured_data") or {}
    tables = sd.get("tables") or []

    if any(tbl.get("rows") for tbl in tables):
        try:
            raw2  = _call_gemini(image_bytes, mime_type, _PROMPT_PASS2, api_key)
            data2 = _parse_json(raw2)

            row_regions: list[TextRegion] = []
            for row in (data2.get("rows") or []):
                for cell in (row.get("cells") or []):
                    row_regions.append(TextRegion(
                        original_text=cell.get("o", ""),
                        translated_text=cell.get("t", ""),
                        bounding_box=_make_bbox_list(cell.get("b"), img_w, img_h),
                        region_type="table_cell",
                    ))

            logger.info("Pass 2: %d cells (%d with bbox)",
                        len(row_regions), sum(1 for r in row_regions if r.bounding_box))

            # Deduplicate: remove Pass 1 non-exempt regions that overlap Pass 2 cells
            p2_bboxes = [r.bounding_box for r in row_regions if r.bounding_box]
            deduped   = [
                r for r in regions
                if r.region_type in _DEDUP_EXEMPT
                or not r.bounding_box
                or not any(_iou(r.bounding_box, b) > 0.30 for b in p2_bboxes)
            ]
            removed = len(regions) - len(deduped)
            if removed:
                logger.info("Dedup: removed %d Pass 1 regions overlapping Pass 2", removed)

            regions = deduped
            regions.extend(row_regions)

        except Exception as e:
            logger.warning("Pass 2 failed — no table cell overlay: %s", e)

    logger.info("Total: %d regions (%d with bbox)",
                len(regions), sum(1 for r in regions if r.bounding_box))

    return OCRResult(
        extracted_text=extracted_text,
        translated_text=translated_text,
        language=language_result,
        regions=regions,
        structured_data=sd,
        warning=None,
    )