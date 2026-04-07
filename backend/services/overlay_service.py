# overlay_service.py
# Replaces OpenCV putText with PIL for full Unicode support.
# Dynamically selects Noto fonts per script, handles RTL, wraps text to fit boxes.

from __future__ import annotations
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.services.language_service import LanguageResult

# ---------------------------------------------------------------------------
# Font registry
# Noto font files should be placed in: assets/fonts/
# Download from: https://fonts.google.com/noto
# ---------------------------------------------------------------------------

FONT_DIR = Path(__file__).parent / "assets" / "fonts"

# Maps Noto font name → TTF filename on disk
FONT_FILE_MAP: dict[str, str] = {
    "NotoSans":             "NotoSans-Regular.ttf",
    "NotoSansLao":          "NotoSansLao-Regular.ttf",
    "NotoSansThai":         "NotoSansThai-Regular.ttf",
    "NotoSansArabic":       "NotoSansArabic-Regular.ttf",
    "NotoSansDevanagari":   "NotoSansDevanagari-Regular.ttf",
    "NotoSansBengali":      "NotoSansBengali-Regular.ttf",
    "NotoSansGujarati":     "NotoSansGujarati-Regular.ttf",
    "NotoSansTamil":        "NotoSansTamil-Regular.ttf",
    "NotoSansTelugu":       "NotoSansTelugu-Regular.ttf",
    "NotoSansKannada":      "NotoSansKannada-Regular.ttf",
    "NotoSansMalayalam":    "NotoSansMalayalam-Regular.ttf",
    "NotoSansCJK":          "NotoSansCJK-Regular.ttc",
    "NotoSansMyanmar":      "NotoSansMyanmar-Regular.ttf",
    "NotoSansKhmer":        "NotoSansKhmer-Regular.ttf",
    "NotoSansHebrew":       "NotoSansHebrew-Regular.ttf",
    "NotoSansArabic":       "NotoSansArabic-Regular.ttf",
    "NotoSerifTibetan":     "NotoSerifTibetan-Regular.ttf",
}

_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _load_font(noto_font_name: str, size: int) -> ImageFont.FreeTypeFont:
    """
    Load and cache a Noto font by name and size.
    Falls back to NotoSans (Latin) if the specific font file isn't found,
    and then to PIL's built-in default as last resort.
    """
    cache_key = (noto_font_name, size)
    if cache_key in _font_cache:
        return _font_cache[cache_key]

    font_file = FONT_FILE_MAP.get(noto_font_name, "NotoSans-Regular.ttf")
    font_path = FONT_DIR / font_file

    try:
        font = ImageFont.truetype(str(font_path), size=size)
    except (OSError, IOError):
        # Try Latin fallback
        fallback_path = FONT_DIR / "NotoSans-Regular.ttf"
        try:
            font = ImageFont.truetype(str(fallback_path), size=size)
        except (OSError, IOError):
            # PIL built-in last resort — no Unicode but won't crash
            font = ImageFont.load_default()

    _font_cache[cache_key] = font
    return font


# ---------------------------------------------------------------------------
# Text fitting
# ---------------------------------------------------------------------------

def _fit_text_to_box(
    text: str,
    box_width: int,
    box_height: int,
    font: ImageFont.FreeTypeFont,
    draw: ImageDraw.ImageDraw,
) -> tuple[list[str], int]:
    """
    Wrap and scale text to fit within a bounding box.
    Returns (wrapped_lines, final_font_size).

    Strategy:
      - Start at box_height // 2 font size
      - Wrap text to fit width
      - Scale down until all lines fit height
      - Hard floor at font size 8 to stay readable
    """
    font_size = max(box_height // 2, 10)
    min_size = 8

    while font_size >= min_size:
        # Estimate chars per line from box width
        # Use a representative char width from the font
        try:
            avg_char_width = draw.textlength("A", font=font) or 8
        except Exception:
            avg_char_width = font_size * 0.6

        chars_per_line = max(int(box_width / avg_char_width), 1)
        wrapped = textwrap.wrap(text, width=chars_per_line) or [text]

        # Check if all lines fit vertically
        line_height = font_size + 2
        total_height = len(wrapped) * line_height

        if total_height <= box_height:
            return wrapped, font_size

        font_size -= 1

    # At minimum size, just truncate
    wrapped = textwrap.wrap(text, width=max(int(box_width / (min_size * 0.6)), 1)) or [text]
    return wrapped[:max(box_height // (min_size + 2), 1)], min_size


# ---------------------------------------------------------------------------
# RTL handling
# ---------------------------------------------------------------------------

def _apply_rtl(text: str) -> str:
    """
    Apply Unicode BiDi algorithm for RTL scripts (Arabic, Hebrew).
    Uses the `python-bidi` library if available, else reverses words as approximation.
    """
    try:
        from bidi.algorithm import get_display
        return get_display(text)
    except ImportError:
        # Rough approximation: reverse word order
        return " ".join(text.split()[::-1])


# ---------------------------------------------------------------------------
# Core overlay
# ---------------------------------------------------------------------------

@dataclass
class OverlayBlock:
    """A text region to overlay on the image."""
    x: int
    y: int
    width: int
    height: int
    translated_text: str


def overlay_translations(
    image: np.ndarray,
    blocks: list[OverlayBlock],
    language_result: LanguageResult | None = None,
    fill_alpha: float = 0.85,
) -> np.ndarray:
    """
    Overlay translated text onto the image using PIL + Noto fonts.

    Args:
        image:           OpenCV BGR numpy array
        blocks:          List of OverlayBlock (position + translated text)
        language_result: From language_service — determines font + RTL
        fill_alpha:      Opacity of the white fill box (0.0–1.0)
                         0.85 = mostly opaque, preserves slight background hint

    Returns:
        OpenCV BGR numpy array with translations overlaid
    """
    if not blocks:
        return image

    # Determine font and direction from language metadata
    noto_font_name = language_result.noto_font if language_result else "NotoSans"
    is_rtl = (language_result.direction == "rtl") if language_result else False

    # Convert OpenCV BGR → PIL RGB
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay_layer = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_layer)

    for block in blocks:
        x, y, w, h = block.x, block.y, block.width, block.height
        text = block.translated_text.strip()

        if not text:
            continue

        # Apply RTL reordering if needed
        if is_rtl:
            text = _apply_rtl(text)

        # Initial font load at estimated size for fitting
        estimated_size = max(h // 2, 10)
        font = _load_font(noto_font_name, estimated_size)

        # Fit text to bounding box
        wrapped_lines, final_size = _fit_text_to_box(text, w, h, font, draw)
        font = _load_font(noto_font_name, final_size)

        # Draw semi-transparent white fill over original region
        fill_alpha_int = int(fill_alpha * 255)
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            fill=(255, 255, 255, fill_alpha_int)
        )

        # Draw each wrapped line
        line_height = final_size + 2
        for i, line in enumerate(wrapped_lines):
            text_y = y + (i * line_height)

            # RTL: right-align text within box
            if is_rtl:
                try:
                    text_width = draw.textlength(line, font=font)
                except Exception:
                    text_width = len(line) * final_size * 0.6
                text_x = x + w - int(text_width) - 2
            else:
                text_x = x + 2

            # Thin shadow for readability on light backgrounds
            draw.text((text_x + 1, text_y + 1), line, font=font, fill=(180, 180, 180, 200))
            # Main text
            draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0, 255))

    # Composite overlay onto original image
    composited = Image.alpha_composite(pil_image, overlay_layer).convert("RGB")

    # Convert back to OpenCV BGR
    return cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Convenience: build OverlayBlocks from OCRResult regions
# ---------------------------------------------------------------------------

def build_overlay_blocks_from_regions(regions: list[dict]) -> list[OverlayBlock]:
    """
    Convert raw region dicts (from OCR result) to typed OverlayBlock list.
    Expected dict keys: x, y, w (or width), h (or height), translated
    """
    blocks = []
    for r in regions:
        blocks.append(OverlayBlock(
            x=r.get("x", 0),
            y=r.get("y", 0),
            width=r.get("w") or r.get("width", 100),
            height=r.get("h") or r.get("height", 30),
            translated_text=r.get("translated", "") or r.get("translated_text", ""),
        ))
    return blocks