# ocr_engine.py
# Orchestrator: delegates to language_service, ocr_service, overlay_service.

from __future__ import annotations

import logging
import os
import uuid
import pathlib
from typing import Callable

import cv2
import numpy as np

from backend.services.language_service import detect_language, LanguageResult
from backend.services.ocr_service import extract_and_translate, OCRResult
from backend.services.overlay_service import (
    overlay_translations,
    build_overlay_blocks_from_regions,
    OverlayBlock,
)

logger = logging.getLogger(__name__)

BASE_DIR   = pathlib.Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def validate_image_bytes(data: bytes) -> bool:
    if len(data) < 12:
        return False
    if data[:4] == b"%PDF":                            return True
    if data[:3] == b"\xff\xd8\xff":                    return True
    if data[:8] == b"\x89PNG\r\n\x1a\n":               return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":  return True
    return False


def _load_image(image_path: str) -> np.ndarray | None:
    img = cv2.imread(image_path)
    if img is None:
        logger.error("cv2.imread failed for path: %s", image_path)
    return img


def process_image(
    image_path: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """
    Main pipeline entry point.

    Args:
        image_path:        Path to uploaded image file
        progress_callback: Optional callable(done, total) called after each
                           region is rendered. Used for SSE real-time progress.

    Returns dict with keys:
        extracted_text, translated_text, translated_image,
        detected_language, structured_data, region_count, error
    """
    result = {
        "extracted_text":    "",
        "translated_text":   "",
        "translated_image":  "",
        "detected_language": "unknown",
        "structured_data":   {},
        "region_count":      0,
        "error":             None,
    }

    # Stage 1: Load
    img = _load_image(image_path)
    if img is None:
        result["error"] = "Could not read image. Check path and file format."
        return result

    # Stage 2: Language detection (deferred — done after OCR)
    language_result: LanguageResult | None = None

    # Stage 3: OCR + Translation
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        result["error"] = (
            "GEMINI_API_KEY not set. "
            "Add GEMINI_API_KEY=your_key to your .env file."
        )
        return result

    try:
        ocr_result: OCRResult = extract_and_translate(
            image=image_path,
            language_result=language_result,
            api_key=api_key,
        )
    except Exception as e:
        logger.exception("OCR/translation stage failed")
        result["error"] = f"OCR failed: {e}"
        return result

    # Stage 3b: Confirm language from extracted text
    if ocr_result.extracted_text:
        language_result = detect_language(ocr_result.extracted_text)
        logger.info(
            "Language confirmed: %s (%s) | confidence=%s | font=%s | dir=%s",
            language_result.language_name,
            language_result.iso_code,
            language_result.confidence,
            language_result.noto_font,
            language_result.direction,
        )
    else:
        logger.warning("No text extracted — skipping language confirmation")

    result["extracted_text"]    = ocr_result.extracted_text
    result["translated_text"]   = ocr_result.translated_text
    result["detected_language"] = (
        language_result.iso_code if language_result else "unknown"
    )
    result["structured_data"]   = ocr_result.structured_data

    if not ocr_result.extracted_text.strip():
        result["error"] = "No readable text found in image."
        return result

    # Stage 4: Overlay rendering
    try:
        if ocr_result.regions:
            region_dicts = [
                {
                    "x":          r.bounding_box.x      if r.bounding_box else 0,
                    "y":          r.bounding_box.y      if r.bounding_box else 0,
                    "width":      r.bounding_box.width  if r.bounding_box else img.shape[1],
                    "height":     r.bounding_box.height if r.bounding_box else 30,
                    "translated": r.translated_text,
                }
                for r in ocr_result.regions
                if r.translated_text.strip()
            ]
            blocks = build_overlay_blocks_from_regions(region_dicts)
        else:
            h, w = img.shape[:2]
            blocks = [
                OverlayBlock(
                    x=0,
                    y=max(0, h - max(80, h // 5)),
                    width=w,
                    height=max(80, h // 5),
                    translated_text=ocr_result.translated_text,
                )
            ]

        result["region_count"] = len(blocks)

        rendered = overlay_translations(
            image=img,
            blocks=blocks,
            language_result=language_result,
            fill_alpha=0.88,
            progress_callback=progress_callback,  # ← real-time progress
        )

        out_path = OUTPUT_DIR / f"{uuid.uuid4()}.jpg"
        cv2.imwrite(str(out_path), rendered)
        result["translated_image"] = str(out_path)

    except Exception as e:
        logger.exception("Overlay rendering failed")
        result["error"] = f"Render failed: {e}"

    return result