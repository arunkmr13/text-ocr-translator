# overlay_service.py
# Renders translated text IN-PLACE over original text regions.
# Supports a progress_callback(done, total) for real-time rendering progress.

from __future__ import annotations
import textwrap
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

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
    "NotoSans":           "NotoSans-Regular.ttf",
    "NotoSansLao":        "NotoSansLao-Regular.ttf",
    "NotoSansThai":       "NotoSansThai-Regular.ttf",
    "NotoSansArabic":     "NotoSansArabic-Regular.ttf",
    "NotoSansDevanagari": "NotoSansDevanagari-Regular.ttf",
    "NotoSansBengali":    "NotoSansBengali-Regular.ttf",
    "NotoSansTamil":      "NotoSansTamil-Regular.ttf",
    "NotoSansCJK":        "NotoSansCJK-Regular.ttc",
    "NotoSansMyanmar":    "NotoSansMyanmar-Regular.ttf",
    "NotoSansKhmer":      "NotoSansKhmer-Regular.ttf",
    "NotoSansHebrew":     "NotoSansHebrew-Regular.ttf",
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
    ih, iw = image.shape[:2]
    x  = max(0, min(x, iw - 1))
    y  = max(0, min(y, ih - 1))
    x2 = max(0, min(x + w, iw))
    y2 = max(0, min(y + h, ih))
    if x2 <= x or y2 <= y:
        return (30, 30, 30)
    region = image[y:y2, x:x2]
    border_pixels = np.concatenate([
        region[0, :], region[-1, :], region[:, 0], region[:, -1],
    ], axis=0)
    median_bgr = np.median(border_pixels, axis=0).astype(int)
    return (int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0]))


def _fit_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
    font_name: str,
    min_size: int = 6,
    max_size: int = 72,
) -> tuple[ImageFont.FreeTypeFont, int]:
    lo, hi = min_size, max_size
    best_size = min_size
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(font_name, mid)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = mid * len(text) * 0.6, mid * 1.2
        if tw <= max_width and th <= max_height:
            best_size = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return _load_font(font_name, best_size), best_size


def _choose_text_color(bg_rgb: tuple[int, int, int]) -> tuple[int, int, int, int]:
    r, g, b = bg_rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (15, 15, 15, 255) if luminance > 140 else (245, 245, 245, 255)


# ---------------------------------------------------------------------------
# In-place block renderer
# ---------------------------------------------------------------------------

def _render_block_inplace(
    draw: ImageDraw.ImageDraw,
    image_bgr: np.ndarray,
    block: OverlayBlock,
    font_name: str,
    is_rtl: bool,
    padding: int = 2,
) -> None:
    x, y, w, h = block.x, block.y, block.width, block.height
    text = block.translated_text.strip()
    if not text or len(text) <= 1:
        return

    ih, iw = image_bgr.shape[:2]
    if w > iw * 0.60 or h > ih * 0.06 or h < 15:
        return

    x  = max(0, min(x, iw - 1))
    y  = max(0, min(y, ih - 1))
    w  = min(w, iw - x)
    h  = min(h, ih - y)
    if w < 5 or h < 5:
        return

    if is_rtl:
        text = _apply_rtl(text)

    bg_rgb = _sample_bg_color(image_bgr, x, y, w, h)
    draw.rectangle([(x, y), (x + w, y + h)], fill=(*bg_rgb, 230))

    inner_w = max(w - padding * 2, 8)
    inner_h = max(h - padding * 2, 6)
    font, size = _fit_font_size(draw, text, inner_w, inner_h, font_name)
    text_color = _choose_text_color(bg_rgb)

    try:
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
    except Exception:
        tw, th = size * len(text) * 0.6, size

    if tw <= inner_w:
        tx = x + padding + max(0, (inner_w - tw) // 2)
        ty = y + padding + max(0, (inner_h - th) // 2)
        draw.text((tx + 1, ty + 1), text, font=font, fill=(0, 0, 0, 80))
        draw.text((tx, ty), text, font=font, fill=text_color)
    else:
        try:
            avg_cw = draw.textlength("A", font=font)
        except Exception:
            avg_cw = size * 0.6
        cpl = max(4, int(inner_w / max(avg_cw, 1)))
        lines = textwrap.wrap(text, width=cpl)
        line_h = size + 2
        if line_h * len(lines) > inner_h and len(lines) > 1:
            new_size = max(6, int(inner_h / len(lines)) - 2)
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
# Region grouping
# ---------------------------------------------------------------------------

def _group_nearby_blocks(
    blocks: list[OverlayBlock],
    y_threshold: int = 10,
    x_gap_threshold: int = 30,
) -> list[OverlayBlock]:
    if not blocks:
        return blocks
    sorted_blocks = sorted(blocks, key=lambda b: (b.y, b.x))
    grouped = []
    used = set()
    for i, b in enumerate(sorted_blocks):
        if i in used:
            continue
        group = [b]
        used.add(i)
        for j, b2 in enumerate(sorted_blocks):
            if j in used:
                continue
            if abs(b2.y - b.y) <= y_threshold:
                b_right = b.x + b.width
                if abs(b2.x - b_right) <= x_gap_threshold or abs(b.x - (b2.x + b2.width)) <= x_gap_threshold:
                    group.append(b2)
                    used.add(j)
        if len(group) == 1:
            grouped.append(b)
        else:
            min_x = min(g.x for g in group)
            min_y = min(g.y for g in group)
            max_x = max(g.x + g.width for g in group)
            max_y = max(g.y + g.height for g in group)
            group_sorted = sorted(group, key=lambda g: g.x)
            merged_text = " ".join(g.translated_text.strip() for g in group_sorted if g.translated_text.strip())
            grouped.append(OverlayBlock(
                x=min_x, y=min_y,
                width=max_x - min_x,
                height=max_y - min_y,
                translated_text=merged_text,
            ))
    return grouped


# ---------------------------------------------------------------------------
# Fallback: bottom panel
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
    cpl     = max(20, int(usable_width / max(avg_cw, 1)))
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
# Core public renderer — with real progress callback
# ---------------------------------------------------------------------------

def overlay_translations(
    image: np.ndarray,
    blocks: list[OverlayBlock],
    language_result: LanguageResult | None = None,
    fill_alpha: float = 0.88,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """
    Render translated text over the image in-place.

    Args:
        image:             OpenCV BGR numpy array
        blocks:            List of OverlayBlock with positions + translated text
        language_result:   Used for font selection and RTL flag
        fill_alpha:        Opacity for fallback panel
        progress_callback: Optional callable(done, total) called after each block renders
                           Use this for real-time SSE progress reporting.
    """
    if not blocks:
        return image

    is_rtl    = (language_result.direction == "rtl") if language_result else False
    font_name = "NotoSans"

    has_boxes = any(
        b.width > 0 and b.height > 0 and (b.x > 0 or b.y > 0)
        for b in blocks
    )

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay   = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    draw      = ImageDraw.Draw(overlay)

    if has_boxes:
        blocks = _group_nearby_blocks(blocks)
        total  = len(blocks)
        logger.info("In-place mode: rendering %d blocks", total)

        for i, block in enumerate(blocks):
            if not block.translated_text.strip():
                if progress_callback:
                    progress_callback(i + 1, total)
                continue
            _render_block_inplace(
                draw=draw,
                image_bgr=image,
                block=block,
                font_name=font_name,
                is_rtl=is_rtl,
            )
            # ← Real progress: called after EACH block is rendered
            if progress_callback:
                progress_callback(i + 1, total)

    else:
        logger.info("Fallback panel mode: no bounding boxes")
        full_text = " ".join(
            b.translated_text.strip() for b in blocks if b.translated_text.strip()
        )
        if is_rtl:
            full_text = _apply_rtl(full_text)
        if progress_callback:
            progress_callback(0, 1)
        _render_bottom_panel(pil_image, overlay, draw, full_text, language_result, font_name)
        if progress_callback:
            progress_callback(1, 1)

    composited = Image.alpha_composite(pil_image, overlay).convert("RGB")
    return cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_overlay_blocks_from_regions(regions: list[dict]) -> list[OverlayBlock]:
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