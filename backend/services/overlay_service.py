# overlay_service.py
# Renders translated text IN-PLACE over original text regions.
# Each OverlayBlock's bounding box is used to:
#   1. Sample the background colour of that region
#   2. Paint a filled rectangle to erase the original text
#   3. Render the translated English text at the same position and scale
#
# Falls back to a clean bottom-panel if no bounding boxes are available.
# Supports all scripts via Noto font family. Handles RTL (Arabic, Hebrew).

from __future__ import annotations
import textwrap
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.services.language_service import LanguageResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Font registry
# ---------------------------------------------------------------------------

FONT_DIR = Path(__file__).parent.parent / "assets" / "fonts"

FONT_FILE_MAP: dict[str, str] = {
    "NotoSans":             "NotoSans-Regular.ttf",
    "NotoSansLao":          "NotoSansLao-Regular.ttf",
    "NotoSansThai":         "NotoSansThai-Regular.ttf",
    "NotoSansArabic":       "NotoSansArabic-Regular.ttf",
    "NotoSansDevanagari":   "NotoSansDevanagari-Regular.ttf",
    "NotoSansBengali":      "NotoSansBengali-Regular.ttf",
    "NotoSansTamil":        "NotoSansTamil-Regular.ttf",
    "NotoSansCJK":          "NotoSansCJK-Regular.ttc",
    "NotoSansMyanmar":      "NotoSansMyanmar-Regular.ttf",
    "NotoSansKhmer":        "NotoSansKhmer-Regular.ttf",
    "NotoSansHebrew":       "NotoSansHebrew-Regular.ttf",
}

_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    key = (name, size)
    if key in _font_cache:
        return _font_cache[key]
    for font_name in [name, "NotoSans"]:
        path = FONT_DIR / FONT_FILE_MAP.get(font_name, "NotoSans-Regular.ttf")
        try:
            f = ImageFont.truetype(str(path), size=size)
            _font_cache[key] = f
            return f
        except (OSError, IOError):
            continue
    f = ImageFont.load_default()
    _font_cache[key] = f
    return f


# ---------------------------------------------------------------------------
# RTL support
# ---------------------------------------------------------------------------

def _apply_rtl(text: str) -> str:
    try:
        from bidi.algorithm import get_display
        from arabic_reshaper import reshape
        return get_display(reshape(text))
    except ImportError:
        return " ".join(text.split()[::-1])


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class OverlayBlock:
    x: int
    y: int
    width: int
    height: int
    translated_text: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_bg_color(image: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[int, int, int]:
    """
    Sample the dominant background colour of a region by looking at its border pixels.
    Returns an (R, G, B) tuple suitable for PIL.
    """
    ih, iw = image.shape[:2]
    x  = max(0, min(x,  iw - 1))
    y  = max(0, min(y,  ih - 1))
    x2 = max(0, min(x + w, iw))
    y2 = max(0, min(y + h, ih))

    if x2 <= x or y2 <= y:
        return (30, 30, 30)

    region = image[y:y2, x:x2]   # BGR

    # Use the border pixels (top row + bottom row + left col + right col)
    border_pixels = np.concatenate([
        region[0, :],
        region[-1, :],
        region[:, 0],
        region[:, -1],
    ], axis=0)

    # Median colour is robust to text pixels in the border sample
    median_bgr = np.median(border_pixels, axis=0).astype(int)
    b, g, r = int(median_bgr[0]), int(median_bgr[1]), int(median_bgr[2])
    return (r, g, b)


def _fit_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
    font_name: str,
    min_size: int = 8,
    max_size: int = 72,
) -> tuple[ImageFont.FreeTypeFont, int]:
    """
    Binary-search for the largest font size where text fits in (max_width × max_height).
    Returns (font, size).
    """
    lo, hi = min_size, max_size
    best_size = min_size
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(font_name, mid)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = mid * len(text) * 0.6, mid * 1.2
        if tw <= max_width and th <= max_height:
            best_size = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return _load_font(font_name, best_size), best_size


def _choose_text_color(bg_rgb: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """Return black or white text colour depending on background luminance."""
    r, g, b = bg_rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (10, 10, 10, 255) if luminance > 140 else (245, 245, 245, 255)


# ---------------------------------------------------------------------------
# In-place block renderer
# ---------------------------------------------------------------------------

def _render_block_inplace(
    draw: ImageDraw.ImageDraw,
    image_bgr: np.ndarray,
    block: OverlayBlock,
    font_name: str,
    is_rtl: bool,
    padding: int = 3,
) -> None:
    x, y, w, h = block.x, block.y, block.width, block.height
    text = block.translated_text.strip()
    if not text:
        return
    
    # Skip blocks covering more than 40% of image width OR height — likely a bad bbox
    ih, iw = image_bgr.shape[:2]
    if w > iw * 0.4 or h > ih * 0.4:
        logger.warning("Skipping oversized block (%dx%d) for text: %s", w, h, text[:30])
        return

    if is_rtl:
        text = _apply_rtl(text)

    """
    For a single OverlayBlock:
      1. Sample background colour from the original image
      2. Paint a filled rectangle to cover the original text
      3. Render the translated text, auto-sized to fit
    """
    x, y, w, h = block.x, block.y, block.width, block.height
    text = block.translated_text.strip()
    if not text:
        return

    if is_rtl:
        text = _apply_rtl(text)

    # 1. Sample background
    bg_rgb = _sample_bg_color(image_bgr, x, y, w, h)

    # 2. Fill rectangle (erase original text)
    # Slight inset so we don't bleed over adjacent elements
    inset = 2
    draw.rectangle(
        [(x + inset, y + inset), (x + w - inset, y + h - inset)],
        fill=(*bg_rgb, 210),  # less opaque so background shows through slightly
)

    # 3. Fit font
    inner_w = max(w - padding * 2, 10)
    inner_h = max(h - padding * 2, 8)

    # Try to fit on one line first; if too long, wrap and shrink
    font, size = _fit_font_size(draw, text, inner_w, inner_h, font_name)

    text_color = _choose_text_color(bg_rgb)

    # Attempt single-line; fall back to wrapped multi-line
    try:
        tb = draw.textbbox((0, 0), text, font=font)
        text_w = tb[2] - tb[0]
    except Exception:
        text_w = size * len(text)

    if text_w <= inner_w:
        # Single line — centre vertically and horizontally
        try:
            tb = draw.textbbox((0, 0), text, font=font)
            th = tb[3] - tb[1]
            tw = tb[2] - tb[0]
        except Exception:
            tw, th = text_w, size
        tx = x + padding + max(0, (inner_w - tw) // 2)
        ty = y + padding + max(0, (inner_h - th) // 2)
        # Shadow
        draw.text((tx + 1, ty + 1), text, font=font, fill=(0, 0, 0, 80))
        draw.text((tx, ty), text, font=font, fill=text_color)
    else:
        # Multi-line wrap
        try:
            avg_cw = draw.textlength("A", font=font)
        except Exception:
            avg_cw = size * 0.6
        cpl = max(5, int(inner_w / max(avg_cw, 1)))
        lines = textwrap.wrap(text, width=cpl)

        # Re-fit font to multi-line height
        line_h = size + 2
        total_h = line_h * len(lines)
        if total_h > inner_h and len(lines) > 1:
            new_size = max(7, int(inner_h / len(lines)) - 2)
            font = _load_font(font_name, new_size)
            size = new_size
            line_h = size + 2

        ty = y + padding + max(0, (inner_h - line_h * len(lines)) // 2)
        for line in lines:
            if ty + line_h > y + h:
                break
            draw.text((x + padding + 1, ty + 1), line, font=font, fill=(0, 0, 0, 80))
            draw.text((x + padding, ty), line, font=font, fill=text_color)
            ty += line_h


# ---------------------------------------------------------------------------
# Fallback: bottom panel (used when no bounding boxes available)
# ---------------------------------------------------------------------------

def _render_bottom_panel(
    pil_image: Image.Image,
    overlay: Image.Image,
    draw: ImageDraw.ImageDraw,
    full_text: str,
    language_result: LanguageResult | None,
    font_name: str,
) -> None:
    w, h = pil_image.size
    panel_padding = max(12, w // 60)
    font_size     = max(14, min(22, w // 52))
    line_spacing  = font_size + 6
    header_size   = max(11, font_size - 3)
    header_height = header_size + panel_padding * 2

    font        = _load_font(font_name, font_size)
    header_font = _load_font(font_name, header_size)

    usable_width = w - panel_padding * 2
    try:
        avg_cw = draw.textlength("A", font=font)
    except Exception:
        avg_cw = font_size * 0.55
    cpl = max(20, int(usable_width / max(avg_cw, 1)))
    wrapped = textwrap.wrap(full_text, width=cpl)

    max_lines = min(len(wrapped), max(6, int(h * 0.38 / line_spacing)))
    panel_h   = min(header_height + max_lines * line_spacing + panel_padding * 2, int(h * 0.45))
    panel_y   = h - panel_h

    draw.rectangle([(0, panel_y), (w, h)], fill=(15, 15, 30, 224))
    draw.line([(0, panel_y), (w, panel_y)], fill=(80, 160, 255, 200), width=2)

    header = (
        f"TRANSLATED  ({language_result.language_name.upper()} → ENGLISH)"
        if language_result else "ENGLISH TRANSLATION"
    )
    draw.text((panel_padding, panel_y + panel_padding), header,
              font=header_font, fill=(80, 160, 255, 220))

    sep_y = panel_y + header_height
    draw.line([(panel_padding, sep_y), (w - panel_padding, sep_y)],
              fill=(80, 160, 255, 80), width=1)

    ty = sep_y + panel_padding
    drawn = 0
    for line in wrapped:
        if ty + line_spacing > h - panel_padding:
            break
        draw.text((panel_padding + 1, ty + 1), line, font=font, fill=(0, 0, 0, 180))
        draw.text((panel_padding, ty), line, font=font, fill=(240, 240, 240, 255))
        ty += line_spacing
        drawn += 1

    remaining = len(wrapped) - drawn
    if remaining > 0:
        clip_font = _load_font(font_name, max(10, header_size - 1))
        draw.text((panel_padding, ty),
                  f"... +{remaining} more line{'s' if remaining > 1 else ''}",
                  font=clip_font, fill=(80, 160, 255, 160))


# ---------------------------------------------------------------------------
# Core public renderer
# ---------------------------------------------------------------------------

def overlay_translations(
    image: np.ndarray,
    blocks: list[OverlayBlock],
    language_result: LanguageResult | None = None,
    fill_alpha: float = 0.88,
) -> np.ndarray:
    """
    Render translated text over the image.

    If blocks have valid bounding boxes (x/y/width/height > 0), renders each
    translation IN-PLACE by:
      - sampling the background colour
      - painting over the original text
      - drawing the English translation at the correct position and scale

    Falls back to a semi-transparent bottom panel when no bounding boxes
    are available (e.g. Gemini didn't return coordinates).

    Args:
        image:           OpenCV BGR numpy array
        blocks:          List of OverlayBlock with positions + translated text
        language_result: Used for font selection and RTL flag
        fill_alpha:      Opacity for fallback panel (ignored for in-place mode)
    """
    if not blocks:
        return image

    is_rtl    = (language_result.direction == "rtl") if language_result else False
    font_name = "NotoSans"  # Translation is always English (LTR)

    h, w = image.shape[:2]

    # Decide mode: in-place if ANY block has a real bounding box
    has_boxes = any(
        b.width > 0 and b.height > 0 and (b.x > 0 or b.y > 0)
        for b in blocks
    )

    # Convert OpenCV BGR → PIL RGBA
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay   = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    draw      = ImageDraw.Draw(overlay)

    if has_boxes:
        logger.info("In-place overlay mode: rendering %d blocks with bounding boxes", len(blocks))
        for block in blocks:
            if not block.translated_text.strip():
                continue
            _render_block_inplace(
                draw=draw,
                image_bgr=image,
                block=block,
                font_name=font_name,
                is_rtl=is_rtl,
            )
    else:
        logger.info("Fallback panel mode: no bounding boxes, rendering bottom panel")
        full_text = " ".join(
            b.translated_text.strip() for b in blocks if b.translated_text.strip()
        )
        if is_rtl:
            full_text = _apply_rtl(full_text)
        _render_bottom_panel(pil_image, overlay, draw, full_text, language_result, font_name)

    composited = Image.alpha_composite(pil_image, overlay).convert("RGB")
    return cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Convenience builder (unchanged signature — ocr_engine.py uses this)
# ---------------------------------------------------------------------------

def build_overlay_blocks_from_regions(regions: list[dict]) -> list[OverlayBlock]:
    """Convert raw region dicts from OCR result to typed OverlayBlock list."""
    blocks = []
    for r in regions:
        blocks.append(OverlayBlock(
            x=r.get("x", 0),
            y=r.get("y", 0),
            width=r.get("w") or r.get("width", 0),
            height=r.get("h") or r.get("height", 0),
            translated_text=r.get("translated", "") or r.get("translated_text", ""),
        ))
    return blocks