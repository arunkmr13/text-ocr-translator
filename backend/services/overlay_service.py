# overlay_service.py
# Semi-transparent box overlay — readability-first.
# Font size is HEIGHT-DRIVEN: derived from bbox height to match the visual
# scale of the original text.
#
# Design principle: NEVER mutate bboxes. The bbox defines the coverage area.
# Only control how text renders INSIDE the given bbox.

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
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
    region_type: str = "other"


# ── Colour helpers ────────────────────────────────────────────────────────────

def _sample_bg(img: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[int,int,int]:
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
    return (int(med[2]), int(med[1]), int(med[0]))


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


# ── Font sizing ───────────────────────────────────────────────────────────────

def _size_from_height(box_h: int, padding: int = 3) -> int:
    """
    Font size derived from bbox height with multiplier 1.25.
    Calibrated so English cap-height matches original Lao glyph visual size.
    _wrap_and_reduce will scale down if text overflows horizontally.
    """
    inner = max(box_h - padding * 2, 8)
    return max(8, int(inner * 1.75))


def _wrap_and_reduce(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_name: str,
    font_size: int,
    inner_w: int,
    inner_h: int,
    single_line_only: bool = False,
) -> tuple[ImageFont.FreeTypeFont, list[str], int]:
    """
    Start at font_size. Wrap text to fit inner_w.
    Reduce font size until wrapped block fits in inner_h.

    single_line_only: if True, never wrap — fit on one line only,
    truncating with ellipsis if needed. Used for stat labels and
    table headers where wrapping to tiny multi-line is worse than
    single-line truncation.
    """
    def line_height(fnt):
        try:
            bb = draw.textbbox((0,0), "Ag", font=fnt)
            return bb[3] - bb[1] + 2
        except Exception:
            return fnt.size + 2

    def text_width(fnt, txt):
        try:
            return draw.textlength(txt, font=fnt)
        except Exception:
            return fnt.size * len(txt) * 0.55

    def wrap_lines(fnt, txt):
        if text_width(fnt, txt) <= inner_w:
            return [txt]
        try:
            avg = max(draw.textlength("A", font=fnt), 1)
        except Exception:
            avg = fnt.size * 0.55
        cpl = max(3, int(inner_w / avg))
        return textwrap.wrap(txt, width=cpl) or [txt]

    def fit_single_line(fnt, txt):
        """Fit on one line, truncate with ellipsis if needed."""
        if text_width(fnt, txt) <= inner_w:
            return [txt]
        try:
            avg = max(draw.textlength("A", font=fnt), 1)
        except Exception:
            avg = fnt.size * 0.55
        chars = max(2, int(inner_w / avg) - 1)
        return [txt[:chars] + "…"]

    size = font_size
    fnt  = _load_font(font_name, size)

    if single_line_only:
        # Reduce font until single line fits vertically, then truncate width
        lh = line_height(fnt)
        while lh > inner_h and size > 8:
            size = max(8, size - 1)
            fnt  = _load_font(font_name, size)
            lh   = line_height(fnt)
        return fnt, fit_single_line(fnt, text), size

    # Normal: wrap then reduce until fits
    lines = wrap_lines(fnt, text)
    lh    = line_height(fnt)
    while lh * len(lines) > inner_h and size > 8:
        size  = max(8, size - 1)
        fnt   = _load_font(font_name, size)
        lines = wrap_lines(fnt, text)
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
    text  = block.translated_text.strip()
    rtype = block.region_type

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

    draw.rectangle([(x,y),(x+w,y+h)], fill=(*fill, box_alpha))

    font_size = _size_from_height(h, padding)
    inner_w   = max(w - padding*2, 4)
    inner_h   = max(h - padding*2, 4)

    # Rendering mode strategy — two discriminators combined:
    #
    # PARAGRAPH mode (wrap at readable size):
    #   Condition: word_count > 6 AND aspect_ratio < 3.0
    #   OR rtype in WRAP_ALWAYS (title, subtitle, footer)
    #   Used for: body text blocks in documents (Italian letter paragraphs etc.)
    #   Gemini returns one bbox covering multiple text lines → height-driven
    #   font would be enormous. Use fixed readable size (14pt) + word wrap.
    #
    # SINGLE-LINE mode (height-driven font, truncate):
    #   Used for: table cells, headers, numbers, stat labels, province names.
    #   Font size = (bbox_height - padding) * 1.25 → matches original scale.
    #   Truncates with ellipsis if text overflows width.
    #
    # The word_count + ratio combination correctly handles edge cases:
    #   - Single digit '3' (ratio=0.8, words=1) → single-line ✓
    #   - 'New Infections' (ratio=3.8, words=2) → single-line ✓
    #   - Long paragraph (ratio=2.0, words=12) → paragraph ✓

    word_count   = len(text.split())
    aspect_ratio = w / max(h, 1)
    WRAP_ALWAYS  = {"title", "subtitle", "footer"}

    is_paragraph = (word_count > 6 and aspect_ratio < 3.0) or rtype in WRAP_ALWAYS

    if is_paragraph:
        single_line = False
        # Readable paragraph font: 14pt base, slightly wider boxes get larger
        font_size = max(11, min(16, int(inner_w / 28)))
    else:
        single_line = True
        # font_size already set by _size_from_height above

    font, lines, size = _wrap_and_reduce(
        draw, text, font_name, font_size, inner_w, inner_h,
        single_line_only=single_line,
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
        draw.text((tx+1, ty+1), line, font=font, fill=(0,0,0,110))
        draw.text((tx,   ty  ), line, font=font, fill=tc)
        ty += line_h


# ── Block grouping ────────────────────────────────────────────────────────────

def _group_nearby_blocks(
    blocks: list[OverlayBlock],
    y_threshold: int = 6,
    x_gap_threshold: int = 15,
) -> list[OverlayBlock]:
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
            min_x  = min(g.x for g in group)
            min_y  = min(g.y for g in group)
            max_x  = max(g.x + g.width  for g in group)
            max_y  = max(g.y + g.height for g in group)
            merged = " ".join(
                g.translated_text.strip()
                for g in sorted(group, key=lambda g: g.x)
                if g.translated_text.strip()
            )
            # Keep the region_type of the first block in the group
            grouped.append(OverlayBlock(
                x=min_x, y=min_y,
                width=max_x-min_x, height=max_y-min_y,
                translated_text=merged,
                region_type=b.region_type,
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
        logger.info("Overlay: %d blocks", total)
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