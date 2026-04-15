# overlay_service.py
# Semi-transparent box overlay — readability-first.
# Font size is HEIGHT-DRIVEN: derived from bbox height to match the visual
# scale of the original text, not binary-searched down to minimum fit.

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.services.language_service import LanguageResult

logger = logging.getLogger(__name__)

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
    size = max(8, size)
    key  = (name, size)
    if key in _font_cache:
        return _font_cache[key]
    for fn in [name, "NotoSans"]:
        path = FONT_DIR / FONT_FILE_MAP.get(fn, "NotoSans-Regular.ttf")
        try:
            f = ImageFont.truetype(str(path), size=size)
            _font_cache[key] = f
            return f
        except (OSError, IOError):
            continue
    f = ImageFont.load_default()
    _font_cache[key] = f
    return f


def _apply_rtl(text: str) -> str:
    try:
        from bidi.algorithm import get_display
        from arabic_reshaper import reshape
        return get_display(reshape(text))
    except ImportError:
        return " ".join(text.split()[::-1])


@dataclass
class OverlayBlock:
    x: int
    y: int
    width: int
    height: int
    translated_text: str


# ── Colour helpers ────────────────────────────────────────────────────────────

def _sample_bg(img: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[int,int,int]:
    """Median colour of the centre 40% of the region (avoids grid-line colours)."""
    ih, iw = img.shape[:2]
    x  = max(0, min(x,  iw-1));  y  = max(0, min(y,  ih-1))
    x2 = min(x+w, iw);           y2 = min(y+h, ih)
    if x2 <= x or y2 <= y:
        return (30, 30, 30)
    cx1 = x  + int((x2-x)*0.30);  cy1 = y  + int((y2-y)*0.30)
    cx2 = x2 - int((x2-x)*0.30);  cy2 = y2 - int((y2-y)*0.30)
    if cx2 <= cx1 or cy2 <= cy1:
        cx1, cy1, cx2, cy2 = x, y, x2, y2
    med = np.median(img[cy1:cy2, cx1:cx2].reshape(-1,3), axis=0).astype(int)
    return (int(med[2]), int(med[1]), int(med[0]))   # BGR→RGB


def _text_color(bg: tuple[int,int,int]) -> tuple[int,int,int,int]:
    lum = 0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]
    return (15,15,15,255) if lum > 140 else (245,245,245,255)


def _fill_color(bg: tuple[int,int,int], f: float = 0.82) -> tuple[int,int,int]:
    r,g,b = bg
    lum   = 0.299*r + 0.587*g + 0.114*b
    if lum > 140:
        return (int(r*f), int(g*f), int(b*f))
    inv = 1.0 + (1.0 - f)
    return (min(255,int(r*inv)), min(255,int(g*inv)), min(255,int(b*inv)))


# ── Font sizing — HEIGHT-DRIVEN ───────────────────────────────────────────────

def _size_from_height(box_h: int, padding: int = 3) -> int:
    """
    Derive font point size from bbox height.
    font_size = (inner_height) × 0.78
    This matches the visual scale of the original text because the bbox height
    is determined by the original glyph height, so English text at this size
    occupies the same vertical space.
    """
    inner = max(box_h - padding * 2, 8)
    return max(8, int(inner * 0.78))


def _wrap_and_reduce(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_name: str,
    font_size: int,
    inner_w: int,
    inner_h: int,
) -> tuple[ImageFont.FreeTypeFont, list[str], int]:
    """
    Start at font_size (height-derived). Wrap text to fit inner_w.
    If wrapped block exceeds inner_h, step down 1pt at a time until it fits.
    Returns (font, lines, final_size).
    """
    def line_height(fnt):
        try:
            bb = draw.textbbox((0,0), "Ag", font=fnt)
            return bb[3] - bb[1] + 2
        except Exception:
            return fnt.size + 2

    def wrap(fnt, txt):
        try:
            if draw.textlength(txt, font=fnt) <= inner_w:
                return [txt]
            avg = max(draw.textlength("A", font=fnt), 1)
        except Exception:
            avg = fnt.size * 0.55
        cpl = max(3, int(inner_w / avg))
        return textwrap.wrap(txt, width=cpl) or [txt]

    size  = font_size
    fnt   = _load_font(font_name, size)
    lines = wrap(fnt, text)
    lh    = line_height(fnt)

    while lh * len(lines) > inner_h and size > 8:
        size  = max(8, size - 1)
        fnt   = _load_font(font_name, size)
        lines = wrap(fnt, text)
        lh    = line_height(fnt)

    return fnt, lines, size


# ── Core renderer ─────────────────────────────────────────────────────────────

def _render_block(
    draw: ImageDraw.ImageDraw,
    image_bgr: np.ndarray,
    block: OverlayBlock,
    font_name: str,
    is_rtl: bool,
    box_alpha: int = 215,
    padding: int = 3,
) -> None:
    x, y, w, h = block.x, block.y, block.width, block.height
    text = block.translated_text.strip()
    if not text:
        return

    ih, iw = image_bgr.shape[:2]
    x = max(0, min(x, iw-1));  y = max(0, min(y, ih-1))
    w = max(1, min(w, iw-x));  h = max(1, min(h, ih-y))
    if w < 8 or h < 8:
        return

    if is_rtl:
        text = _apply_rtl(text)

    bg   = _sample_bg(image_bgr, x, y, w, h)
    fill = _fill_color(bg)
    tc   = _text_color(fill)

    # Filled semi-transparent box — fully covers original text
    draw.rectangle([(x,y),(x+w,y+h)], fill=(*fill, box_alpha))

    # Font size from height — matches original text scale
    font_size = _size_from_height(h, padding)
    inner_w   = max(w - padding*2, 4)
    inner_h   = max(h - padding*2, 4)

    font, lines, size = _wrap_and_reduce(
        draw, text, font_name, font_size, inner_w, inner_h
    )
    if not lines:
        return

    try:
        bb     = draw.textbbox((0,0), lines[0], font=font)
        line_h = bb[3] - bb[1] + 2
    except Exception:
        line_h = size + 2

    total_h = line_h * len(lines)
    ty = y + padding + max(0, (h - padding*2 - total_h) // 2)

    for line in lines:
        if ty + line_h > y + h:
            break
        try:    lw = draw.textlength(line, font=font)
        except: lw = size * len(line) * 0.55
        tx = x + padding + max(0, int((w - padding*2 - lw) / 2))
        draw.text((tx+1, ty+1), line, font=font, fill=(0,0,0,110))   # shadow
        draw.text((tx,   ty  ), line, font=font, fill=tc)             # text
        ty += line_h


# ── Block grouping ────────────────────────────────────────────────────────────

def _group_nearby_blocks(
    blocks: list[OverlayBlock],
    y_threshold: int = 6,
    x_gap_threshold: int = 15,
) -> list[OverlayBlock]:
    """Merge only directly adjacent same-row fragments — tight thresholds
    prevent separate table columns being merged into one block."""
    if not blocks:
        return blocks
    sorted_b  = sorted(blocks, key=lambda b: (b.y, b.x))
    grouped: list[OverlayBlock] = []
    used: set[int] = set()

    for i, b in enumerate(sorted_b):
        if i in used: continue
        group = [b]; used.add(i)
        for j, b2 in enumerate(sorted_b):
            if j in used: continue
            if abs(b2.y - b.y) <= y_threshold and \
               abs(b2.x - (b.x + b.width)) <= x_gap_threshold:
                group.append(b2); used.add(j)

        if len(group) == 1:
            grouped.append(b)
        else:
            min_x  = min(g.x            for g in group)
            min_y  = min(g.y            for g in group)
            max_x  = max(g.x + g.width  for g in group)
            max_y  = max(g.y + g.height for g in group)
            merged = " ".join(
                g.translated_text.strip()
                for g in sorted(group, key=lambda g: g.x)
                if g.translated_text.strip()
            )
            grouped.append(OverlayBlock(
                x=min_x, y=min_y,
                width=max_x-min_x, height=max_y-min_y,
                translated_text=merged,
            ))
    return grouped


# ── Fallback bottom panel ─────────────────────────────────────────────────────

def _render_bottom_panel(
    draw, img_w, img_h, full_text, language_result, font_name
) -> None:
    pad   = max(12, img_w//60);  fsize = max(14, min(22, img_w//52))
    lsp   = fsize + 6;           hsize = max(11, fsize-3);  hh = hsize + pad*2
    font  = _load_font(font_name, fsize)
    hfont = _load_font(font_name, hsize)
    try:    avg = draw.textlength("A", font=font)
    except: avg = fsize * 0.55
    cpl     = max(20, int((img_w-pad*2)/max(avg,1)))
    wrapped = textwrap.wrap(full_text, width=cpl)
    max_l   = min(len(wrapped), max(6, int(img_h*0.38/lsp)))
    ph      = min(hh + max_l*lsp + pad*2, int(img_h*0.45))
    py      = img_h - ph
    draw.rectangle([(0,py),(img_w,img_h)], fill=(15,15,30,224))
    draw.line([(0,py),(img_w,py)], fill=(80,160,255,200), width=2)
    hdr = (f"TRANSLATED  ({language_result.language_name.upper()} → ENGLISH)"
           if language_result else "ENGLISH TRANSLATION")
    draw.text((pad, py+pad), hdr, font=hfont, fill=(80,160,255,220))
    draw.line([(pad,py+hh),(img_w-pad,py+hh)], fill=(80,160,255,80), width=1)
    ty = py+hh+pad; drawn = 0
    for line in wrapped:
        if ty+lsp > img_h-pad: break
        draw.text((pad+1,ty+1), line, font=font, fill=(0,0,0,180))
        draw.text((pad,  ty  ), line, font=font, fill=(240,240,240,255))
        ty += lsp; drawn += 1
    rem = len(wrapped)-drawn
    if rem > 0:
        cf = _load_font(font_name, max(10,hsize-1))
        draw.text((pad,ty), f"... +{rem} more line{'s' if rem>1 else ''}",
                  font=cf, fill=(80,160,255,160))


# ── Public API ────────────────────────────────────────────────────────────────

def overlay_translations(
    image: np.ndarray,
    blocks: list[OverlayBlock],
    language_result: LanguageResult | None = None,
    fill_alpha: float = 0.88,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    if not blocks:
        return image

    is_rtl    = (language_result.direction == "rtl") if language_result else False
    font_name = "NotoSans"
    has_boxes = any(b.width >= 8 and b.height >= 8 for b in blocks)

    pil  = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov   = Image.new("RGBA", pil.size, (0,0,0,0))
    draw = ImageDraw.Draw(ov)

    if has_boxes:
        blocks = _group_nearby_blocks(blocks)
        total  = len(blocks)
        logger.info("Overlay: %d blocks (height-driven font mode)", total)
        for i, block in enumerate(blocks):
            if block.translated_text.strip():
                _render_block(draw, image, block, font_name, is_rtl,
                              box_alpha=215, padding=3)
            if progress_callback:
                progress_callback(i+1, total)
    else:
        logger.info("Overlay: no bboxes — bottom panel")
        full = " ".join(b.translated_text.strip() for b in blocks if b.translated_text.strip())
        if is_rtl: full = _apply_rtl(full)
        if progress_callback: progress_callback(0,1)
        _render_bottom_panel(draw, pil.width, pil.height, full, language_result, font_name)
        if progress_callback: progress_callback(1,1)

    out = Image.alpha_composite(pil, ov).convert("RGB")
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


def build_overlay_blocks_from_regions(regions: list[dict]) -> list[OverlayBlock]:
    return [
        OverlayBlock(
            x=r.get("x",0), y=r.get("y",0),
            width =r.get("w") or r.get("width", 0),
            height=r.get("h") or r.get("height",0),
            translated_text=r.get("translated","") or r.get("translated_text",""),
        )
        for r in regions
    ]