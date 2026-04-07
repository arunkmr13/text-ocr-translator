# ocr_engine.py
# Orchestrator: replaces the 700-line Tesseract monolith.
# Delegates entirely to language_service, ocr_service, overlay_service.
# No Tesseract. No googletrans. No PSM tuning. No score multipliers.

from __future__ import annotations

import logging
import os
import uuid
import pathlib
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR   = pathlib.Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Image validation (retained from original — format checks are still valid)
# ---------------------------------------------------------------------------

def validate_image_bytes(data: bytes) -> bool:
    """Quick magic-byte validation before passing to OpenCV."""
    if len(data) < 12:
        return False
    if data[:4] == b"%PDF":                            return True
    if data[:3] == b"\xff\xd8\xff":                    return True
    if data[:8] == b"\x89PNG\r\n\x1a\n":               return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":  return True
    return False


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image(image_path: str) -> np.ndarray | None:
    img = cv2.imread(image_path)
    if img is None:
        logger.error("cv2.imread failed for path: %s", image_path)
    return img


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_image(image_path: str) -> dict:
    """
    Main pipeline entry point. Replaces the original 700-line process_image.

    Pipeline:
      1. Load image
      2. Detect language (Unicode block analysis)
      3. Extract + translate via Claude Vision API (single call)
      4. Render translated overlay via PIL + Noto fonts
      5. Return structured result dict

    Returns dict with keys:
      extracted_text    : original language text from image
      translated_text   : English translation
      translated_image  : path to overlay-rendered output image
      detected_language : ISO 639-1 code e.g. "lo", "ar", "zh"
      structured_data   : key metrics extracted by Claude (if any)
      error             : non-None string if a stage failed
    """
    result = {
        "extracted_text":    "",
        "translated_text":   "",
        "translated_image":  "",
        "detected_language": "unknown",
        "structured_data":   {},
        "error":             None,
    }

    # ── Stage 1: Load ────────────────────────────────────────────────
    img = _load_image(image_path)
    if img is None:
        result["error"] = "Could not read image. Check path and file format."
        return result

    # ── Stage 2: Language detection ──────────────────────────────────
    # We do a lightweight OCR-free Unicode probe first.
    # Pass a small center crop of the image as PIL to Claude for initial
    # character sampling — or simply let Claude Vision auto-detect.
    # For now: pass None to let ocr_service prompt Claude to auto-detect,
    # then confirm with Unicode analysis on the returned extracted_text.
    language_result: LanguageResult | None = None

    # ── Stage 3: OCR + Translation ───────────────────
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
            language_result=language_result,   # None → Claude auto-detects
            api_key=api_key,
        )
    except Exception as e:
        logger.exception("OCR/translation stage failed")
        result["error"] = f"OCR failed: {e}"
        return result

    # ── Stage 3b: Confirm language from extracted text ───────────────
    # Now that we have extracted text, run Unicode block analysis on it.
    # This gives us the correct Noto font + RTL flag for overlay rendering.
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

    # ── Stage 4: Overlay rendering ────────────────────────────────────
    try:
        # Build overlay blocks from OCR regions
        # If Claude returned per-region bounding boxes, use them.
        # Otherwise fall back to a single full-image banner block.
        if ocr_result.regions:
            region_dicts = [
                {
                    "x":           r.bounding_box.x      if r.bounding_box else 0,
                    "y":           r.bounding_box.y      if r.bounding_box else 0,
                    "width":       r.bounding_box.width  if r.bounding_box else img.shape[1],
                    "height":      r.bounding_box.height if r.bounding_box else 30,
                    "translated":  r.translated_text,
                }
                for r in ocr_result.regions
                if r.translated_text.strip()
            ]
            blocks = build_overlay_blocks_from_regions(region_dicts)
        else:
            # No bounding boxes — single banner at bottom of image
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

        rendered = overlay_translations(
            image=img,
            blocks=blocks,
            language_result=language_result,
            fill_alpha=0.88,
        )

        # Save output
        out_path = OUTPUT_DIR / f"{uuid.uuid4()}.jpg"
        cv2.imwrite(str(out_path), rendered)
        result["translated_image"] = str(out_path)

    except Exception as e:
        logger.exception("Overlay rendering failed")
        result["error"] = f"Render failed: {e}"
        # Still return text results even if render fails

    return result