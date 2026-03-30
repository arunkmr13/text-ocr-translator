"""
ocr_engine.py  —  OCR + translation pipeline
Base: doc-1 (recent version, English path retained as-is)

Fixes applied (non-English paths only):
  [FIX-1]  ocr_string: per-script confidence cutoff — Arabic/Devanagari/SEA
            use conf > 20, Latin/CJK use conf > 40. is_valid_text now
            script-aware so Arabic/Devanagari chars are not counted as "weird".
  [FIX-2]  get_lines: per-pack PSM order and merge thresholds.
            Arabic gets RTL-aware PSM list. SEA gets tighter merges.
            CJK retains vertical PSM 5.
  [FIX-3]  render_overlay: RTL rendering for Arabic/Urdu/Persian via
            python-bidi + arabic_reshaper when available, graceful fallback.
            Overlap guard from original (doc-2) restored.
            Per-script font sizing and box padding.
            Box width capped at 90% of image width.
  [FIX-4]  probe scoring: per-family score multipliers unified and raised
            for Arabic/Devanagari to compete fairly with Latin probes.
  [FIX-5]  _check_latin_early_exit: mixed-script guard — if raw CJK/SEA/
            Arabic chars appear in the eng scan the strong Latin exit is
            suppressed even when latin_ratio >= 0.85.
  [FIX-6]  langdetect cross-check: min-token guard lowered to 3 for
            non-Latin scripts (short Arabic/CJK texts were not overriding).
"""

import cv2
import logging
import math
import threading
import unicodedata
import re
import os
import uuid
import pathlib
import numpy as np
import pytesseract

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory, LangDetectException
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

DetectorFactory.seed = 0

# ── Critical pack list ──────────────────────────────────────────────
_CRITICAL_PACKS = {
    "jpn":      "Japanese",
    "kor":      "Korean",
    "chi_sim":  "Chinese (Simplified)",
    "chi_tra":  "Chinese (Traditional)",
    "tha":      "Thai",
    "lao":      "Lao",
    "khm":      "Khmer",
    "ara":      "Arabic",
    "hin":      "Hindi",
    "urd":      "Urdu",
}

_HOMEBREW_TESSERACT_CANDIDATES = [
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
]
_HOMEBREW_TESSDATA_CANDIDATES = [
    "/opt/homebrew/share/tessdata",
    "/usr/local/share/tessdata",
    "/opt/homebrew/Cellar/tesseract-lang/4.1.0/share/tessdata",
    "/usr/local/Cellar/tesseract-lang/4.1.0/share/tessdata",
]

# ── Module-level CLAHE (reused across calls) ───────────────────────
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ── Thread-safety lock for langs cache ────────────────────────────
_cache_lock = threading.Lock()

# ── Module-level thread pool ───────────────────────────────────────
_OCR_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ocr")

# ── RTL pack set (used by rendering and get_lines) ─────────────────
_RTL_PACKS = {"ara", "urd", "fas", "heb"}

# ── [FIX-3] Optional RTL rendering libs ───────────────────────────
try:
    from bidi.algorithm import get_display as bidi_display
    from arabic_reshaper import reshape as arabic_reshape
    _HAS_BIDI = True
except ImportError:
    _HAS_BIDI = False
    logger.debug(
        "python-bidi / arabic_reshaper not installed — RTL text "
        "will render without reshaping. "
        "pip install python-bidi arabic-reshaper"
    )

import atexit
atexit.register(_OCR_EXECUTOR.shutdown, wait=False)


# ──────────────────────── Tesseract setup ─────────────────────────

def _configure_tesseract():
    import shutil
    cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if cmd and os.path.isfile(cmd):
        pytesseract.pytesseract.tesseract_cmd = cmd
        logger.info("tesseract binary (env): %s", cmd)
    else:
        for candidate in _HOMEBREW_TESSERACT_CANDIDATES:
            if os.path.isfile(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                logger.info("tesseract binary (homebrew): %s", candidate)
                break
        else:
            found = shutil.which("tesseract")
            if found:
                pytesseract.pytesseract.tesseract_cmd = found
                logger.info("tesseract binary (PATH): %s", found)

    tdata = os.environ.get("TESSDATA_PREFIX", "").strip()
    if tdata and os.path.isdir(tdata):
        os.environ["TESSDATA_PREFIX"] = tdata
        logger.info("tessdata (env): %s", tdata)
    else:
        for candidate in _HOMEBREW_TESSDATA_CANDIDATES:
            if os.path.isdir(candidate):
                files = os.listdir(candidate)
                packs = [f[:-13] for f in files if f.endswith(".traineddata")]
                if len(packs) > 5:
                    os.environ["TESSDATA_PREFIX"] = candidate
                    logger.info("tessdata (homebrew, %d packs): %s",
                                len(packs), candidate)
                    break
        else:
            logger.debug("tessdata: using system default")


def _check_packs_on_startup() -> set:
    _configure_tesseract()
    try:
        installed = set(pytesseract.get_languages(config=""))
        missing = {p: lang for p, lang in _CRITICAL_PACKS.items()
                   if p not in installed}
        if missing:
            logger.warning(
                "Missing Tesseract language packs: %s\n"
                "  macOS:  brew install tesseract-lang\n"
                "  Ubuntu: sudo apt-get install tesseract-ocr-all",
                ", ".join(f"{p} ({lang})" for p, lang in missing.items()),
            )
        else:
            logger.info("All critical Tesseract packs installed (%d total).",
                        len(installed))
        return installed
    except Exception as e:
        logger.error("Failed to query Tesseract languages: %s", e)
        return {"eng"}


_AVAILABLE_LANGS_CACHE: set | None = None


def get_available_langs() -> set:
    """Thread-safe lazy initialisation of the langs cache."""
    global _AVAILABLE_LANGS_CACHE
    if _AVAILABLE_LANGS_CACHE is None:
        with _cache_lock:
            if _AVAILABLE_LANGS_CACHE is None:
                _AVAILABLE_LANGS_CACHE = _check_packs_on_startup()
    return _AVAILABLE_LANGS_CACHE


# ──────────────────────────── Paths ───────────────────────────────

BASE_DIR   = str(pathlib.Path(__file__).parent.parent)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANG_TO_PACK = {
    "af":"afr","ar":"ara","hy":"hye","az":"aze","eu":"eus","be":"bel",
    "bn":"ben","bg":"bul","ca":"cat","zh-cn":"chi_sim","zh-tw":"chi_tra",
    "hr":"hrv","cs":"ces","da":"dan","nl":"nld","en":"eng","et":"est",
    "fi":"fin","fr":"fra","gl":"glg","ka":"kat","de":"deu","el":"ell",
    "gu":"guj","he":"heb","hi":"hin","hu":"hun","id":"ind","ga":"gle",
    "it":"ita","ja":"jpn","kn":"kan","ko":"kor","lo":"lao","lv":"lav",
    "lt":"lit","mk":"mkd","ms":"msa","ml":"mal","mt":"mlt","mr":"mar",
    "ne":"nep","fa":"fas","pl":"pol","pt":"por","pa":"pan","ro":"ron",
    "ru":"rus","sr":"srp","sk":"slk","sl":"slv","es":"spa","sw":"swa",
    "sv":"swe","tl":"fil","ta":"tam","te":"tel","th":"tha","tr":"tur",
    "uk":"ukr","ur":"urd","vi":"vie","cy":"cym","si":"sin","km":"khm",
    "mn":"mon","my":"mya","am":"amh","ti":"tir",
}
PACK_TO_LANG = {v: k for k, v in LANG_TO_PACK.items()}

SCRIPT_TO_PACKS = {
    "Lao":["lao"],"Thai":["tha"],"Khmer":["khm"],"Myanmar":["mya"],
    "Japanese":["jpn"],"HanS":["chi_sim"],"HanT":["chi_tra"],
    "Hangul":["kor"],
    "Arabic":["ara","fas","urd"],"Devanagari":["hin","mar","nep"],
    "Bengali":["ben"],"Tamil":["tam"],"Telugu":["tel"],"Kannada":["kan"],
    "Malayalam":["mal"],"Gujarati":["guj"],"Gurmukhi":["pan"],
    "Cyrillic":["rus","ukr","bul"],"Greek":["ell"],"Armenian":["hye"],
    "Hebrew":["heb"],"Georgian":["kat"],"Ethiopic":["amh"],
    "Latin":["eng","fra","deu","spa","por","ita","nld","pol","vie",
             "ind","tur"],
}
SEA_PACKS   = ["lao","tha","khm","mya"]
CJK_PACKS   = ["jpn","chi_sim","chi_tra"]
KOREAN_PACK = "kor"


# ──────────────────────── Font resolution ─────────────────────────

_FONT_SEARCH_PATHS = [
    os.path.join(BASE_DIR, "fonts", "NotoSans-Regular.ttf"),
    os.path.join(BASE_DIR, "fonts", "DejaVuSans.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

_resolved_font_path: str | None = None


def _resolve_font_path() -> str | None:
    global _resolved_font_path
    if _resolved_font_path is not None:
        return _resolved_font_path
    for fp in _FONT_SEARCH_PATHS:
        if os.path.exists(fp):
            _resolved_font_path = fp
            return fp
    return None


# Font cache: ImageFont.truetype() is expensive (~1ms per call).
# fit_text() calls it up to 300+ times per render (20 boxes × 15 binary
# search steps). Caching by size reduces this to one disk read per size.
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}

def load_font(size: int) -> ImageFont.FreeTypeFont:
    size = max(8, int(size))
    if size in _font_cache:
        return _font_cache[size]
    fp = _resolve_font_path()
    if fp:
        try:
            font = ImageFont.truetype(fp, size)
            _font_cache[size] = font
            return font
        except Exception:
            pass
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


# ──────────────────────── Image validation ────────────────────────

def validate_image_bytes(data: bytes) -> bool:
    if len(data) < 12:
        return False
    if data[:4] == b"%PDF":                           return True
    if data[:3] == b"\xff\xd8\xff":                   return True
    if data[:8] == b"\x89PNG\r\n\x1a\n":              return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP": return True
    return False


# ──────────────────────────── Helpers ─────────────────────────────

def _latin_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if ord(c) < 0x0250) / len(alpha)


def _is_latin_dominant(text: str, threshold: float = 0.70) -> bool:
    return _latin_ratio(text) >= threshold


def quality_score(text: str) -> float:
    if not text or not text.strip():
        return 0
    tokens = text.split()
    if not tokens:
        return 0
    length_penalty = min(1.0, len(tokens) / 8)
    sc = sum(
        1 for c in text
        if ord(c) > 127 and not c.isdigit()
        and unicodedata.category(c) not in
            ("Po","Ps","Pe","Pi","Pf","Pd","Zs","Cc","Cf")
    )
    rw = sum(1 for t in tokens
             if len(t) >= 2 and re.search(r'[a-zA-Z\u0080-\uFFFF]', t))
    sr = sum(1 for t in tokens if len(t) == 1) / len(tokens)
    return (sc * 2 + rw) * max(0.2, 1.0 - sr * 1.5) * length_penalty


# [FIX-1] Script-aware validity check.
# Only penalise genuine junk (control chars, box-drawing symbols).
# Arabic/Devanagari/SEA codepoints are NOT counted as weird.
def is_valid_text(text: str, pack: str = "eng") -> bool:
    if not text or len(text.strip()) < 3:
        return False
    junk = sum(
        1 for c in text
        if unicodedata.category(c) in ("Cc", "Cf", "Cs", "Co", "Cn")
        or (0x2500 <= ord(c) <= 0x257F)   # box-drawing — common OCR noise
    )
    ratio = junk / max(1, len(text))
    return ratio < 0.35


# ── Script-specific character counters ────────────────────────────

def cjk_char_count(text):
    return sum(1 for c in text if (
        0x3040 <= ord(c) <= 0x30FF or
        0x4E00 <= ord(c) <= 0x9FFF or
        0xAC00 <= ord(c) <= 0xD7AF or
        0x3400 <= ord(c) <= 0x4DBF
    ))

def ideograph_char_count(text):
    return sum(1 for c in text if (
        0x4E00 <= ord(c) <= 0x9FFF or
        0x3400 <= ord(c) <= 0x4DBF
    ))

def hangul_char_count(text):
    return sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7AF)

def kana_char_count(text):
    return sum(1 for c in text if 0x3040 <= ord(c) <= 0x30FF)

def sea_char_count(text, script):
    ranges = {
        "Thai":    (0x0E00, 0x0E7F),
        "Lao":     (0x0E80, 0x0EFF),
        "Khmer":   (0x1780, 0x17FF),
        "Myanmar": (0x1000, 0x109F),
    }
    lo, hi = ranges.get(script, (0, 0))
    return sum(1 for c in text if lo <= ord(c) <= hi)

def arabic_char_count(text):
    """Arabic + extended Arabic-Indic (covers Urdu/Persian glyphs too)."""
    return sum(1 for c in text if
               0x0600 <= ord(c) <= 0x06FF or   # Arabic
               0x0750 <= ord(c) <= 0x077F or   # Arabic Supplement
               0xFB50 <= ord(c) <= 0xFDFF or   # Arabic Presentation Forms-A
               0xFE70 <= ord(c) <= 0xFEFF)      # Arabic Presentation Forms-B

def devanagari_char_count(text):
    return sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)


# ── Handwriting detection ──────────────────────────────────────────

def is_likely_handwritten(gray: np.ndarray) -> bool:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var()) > 800


def preprocess_for_handwriting(gray: np.ndarray) -> np.ndarray:
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.dilate(binary, kernel, iterations=1)


def fast_preprocess(gray: np.ndarray) -> np.ndarray:
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# [FIX-1] Per-script confidence cutoff.
# Arabic/Devanagari/SEA Tesseract models produce systematically lower
# confidence scores. conf > 20 retains valid tokens; conf > 40 keeps
# Latin/CJK output clean. Do NOT lower these further — conf > 8 admits
# pure noise and degrades extraction quality for all scripts.
_LOW_CONF_PACKS = {
    "ara","urd","fas",          # Arabic-script
    "hin","mar","nep",          # Devanagari
    "ben","tam","tel","kan",    # Other Indic
    "mal","guj","pan",
    "tha","lao","khm","mya",    # SEA
    "heb","kat","amh","tir",    # Other non-Latin
}


def ocr_string(gray, pack, psm=6):
    """
    Run Tesseract OCR and return filtered text.
    conf > 20 for scripts in _LOW_CONF_PACKS, conf > 40 for all others.
    """
    conf_threshold = 20 if pack in _LOW_CONF_PACKS else 40
    try:
        d = pytesseract.image_to_data(
            gray,
            lang=pack,
            output_type=pytesseract.Output.DICT,
            config=f"--oem 3 --psm {psm}",
        )
        texts = [
            d["text"][i]
            for i in range(len(d["text"]))
            if int(d["conf"][i]) > conf_threshold
            and d["text"][i].strip()
        ]
        filtered = [t for t in texts if is_valid_text(t, pack)]
        return " ".join(filtered)
    except Exception as e:
        logger.debug("ocr_string pack=%s psm=%d failed: %s", pack, psm, e)
        return ""


# ── Translation helpers ────────────────────────────────────────────

def _translator() -> GoogleTranslator:
    return GoogleTranslator(source="auto", target="en")


@lru_cache(maxsize=512)
def translate_cached(text: str) -> str:
    try:
        t = _translator().translate(text)
        return t or text
    except Exception:
        return text


def translate_one(src: str) -> str:
    return translate_cached(src)


def translate_full(text: str) -> str:
    if not text.strip():
        return text
    try:
        chunks, chunk = [], ""
        for w in text.split():
            if len(chunk) + len(w) + 1 > 4500:
                chunks.append(chunk.strip())
                chunk = w
            else:
                chunk += " " + w
        if chunk:
            chunks.append(chunk.strip())
        return " ".join(translate_cached(c) for c in chunks if c)
    except Exception:
        return text


def batch_translate(texts: list) -> list:
    if not texts:
        return []
    SEP = "\n||||\n"
    try:
        joined = SEP.join(t.strip() for t in texts)
        if len(joined) <= 4500:
            translated = _translator().translate(joined) or joined
            parts = [p.strip("\n ") for p in translated.split("||||")]
            if len(parts) == len(texts):
                return parts
        return [translate_one(t) for t in texts]
    except Exception:
        return [translate_one(t) for t in texts]


# ─────────────────────────── Detection ───────────────────────────

def try_osd(gray):
    h, w = gray.shape
    small = cv2.resize(gray, (min(1400, w), min(1000, h)))
    _, bw = cv2.threshold(small, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    found_latin = False
    for img in [bw, 255 - bw, small]:
        try:
            osd = pytesseract.image_to_osd(
                img, config="--psm 0 --oem 1",
                output_type=pytesseract.Output.DICT,
            )
            s = osd.get("script", "")
            if s and s != "Latin":
                return s
            if s == "Latin":
                found_latin = True
        except Exception:
            continue
    return "Latin" if found_latin else None


def probe(gray, packs, available):
    results = {}
    for p in packs:
        if p not in available:
            continue
        t6   = ocr_string(gray, p, 6)
        t3   = ocr_string(gray, p, 3)
        text = t6 if quality_score(t6) >= quality_score(t3) else t3
        results[p] = (quality_score(text), text)
    return results


def probe_parallel(gray, packs, available) -> dict:
    valid_packs = [p for p in packs if p in available]
    if not valid_packs:
        return {}
    results = {}
    futures = {_OCR_EXECUTOR.submit(ocr_string, gray, p, 6): p
               for p in valid_packs}
    for fut in as_completed(futures):
        p    = futures[fut]
        text = fut.result() or ""
        results[p] = (quality_score(text), text)
    return results


def probe_korean(gray, available) -> tuple:
    if KOREAN_PACK not in available:
        return 0, ""
    t  = ocr_string(gray, KOREAN_PACK, 6) or ocr_string(gray, KOREAN_PACK, 3)
    hc = hangul_char_count(t)
    if hc < 3:
        logger.debug("kor probe: only %d Hangul chars, skipping", hc)
        return 0, ""
    score = quality_score(t) + hc * 5
    logger.debug("kor probe: hangul=%d score=%.1f", hc, score)
    return score, t


def _ocr_japanese(gray, available) -> tuple:
    if "jpn" not in available:
        return "", 0
    t_h = ocr_string(gray, "jpn", 6) or ocr_string(gray, "jpn", 3)
    t_v = ocr_string(gray, "jpn", 5)

    def _jpn_score(t):
        kc = kana_char_count(t)
        cc = cjk_char_count(t)
        return quality_score(t) + kc * 6 + cc * 3, kc, cc

    sh, kh, ch = _jpn_score(t_h)
    sv, kv, cv = _jpn_score(t_v)
    if sv > sh:
        logger.debug("jpn: vertical PSM5 wins (score=%.1f kana=%d cjk=%d)",
                     sv, kv, cv)
        return t_v, sv
    logger.debug("jpn: horizontal wins (score=%.1f kana=%d cjk=%d)",
                 sh, kh, ch)
    return t_h, sh


def probe_cjk(gray, available) -> dict:
    results = {}
    for p in [x for x in CJK_PACKS if x in available]:
        if p == "jpn":
            t, score = _ocr_japanese(gray, available)
            kc = kana_char_count(t)
            cc = cjk_char_count(t)
            if kc < 3 and cc < 8:
                logger.debug("jpn probe: kana=%d cjk=%d — below threshold",
                             kc, cc)
                continue
            if _is_latin_dominant(t, threshold=0.60):
                logger.info("jpn probe: Latin-dominant (ratio=%.2f) — discard",
                            _latin_ratio(t))
                continue
        else:
            t  = ocr_string(gray, p, 6) or ocr_string(gray, p, 3)
            cc = ideograph_char_count(t)
            hc = hangul_char_count(t)
            if hc >= max(1, cc * 0.3) or cc < 3:
                logger.debug("chi probe %s: ideograph=%d hangul=%d — skip",
                             p, cc, hc)
                continue
            score = quality_score(t) + cc * 4
        results[p] = (score, t)
        logger.debug("cjk probe pack=%s kana=%d cjk=%d score=%.1f",
                     p, kana_char_count(t), cjk_char_count(t), score)
    return results


# [FIX-4] Dedicated Arabic/Indic probe with calibrated score multipliers.
# ara/urd: ×6 per Arabic char. hin/mar/nep: ×5 per Devanagari char.
# This brings parity with kana ×6 and Hangul ×5 already used for CJK/Korean.
def probe_arabic_indic(gray, available) -> dict:
    results = {}

    for pack in ("ara", "urd", "fas"):
        if pack not in available:
            continue
        t  = ocr_string(gray, pack, 6) or ocr_string(gray, pack, 3)
        ac = arabic_char_count(t)
        if ac < 8:
            logger.debug("ara/urd probe %s: only %d Arabic chars — skip",
                         pack, ac)
            continue
        score = quality_score(t) + ac * 6
        results[pack] = (score, t)
        logger.debug("ara probe pack=%s arabic_chars=%d score=%.1f",
                     pack, ac, score)

    for pack in ("hin", "mar", "nep"):
        if pack not in available:
            continue
        t  = ocr_string(gray, pack, 6) or ocr_string(gray, pack, 3)
        dc = devanagari_char_count(t)
        if dc < 6:
            logger.debug("devan probe %s: only %d Devanagari chars — skip",
                         pack, dc)
            continue
        score = quality_score(t) + dc * 5
        results[pack] = (score, t)
        logger.debug("devan probe pack=%s devan_chars=%d score=%.1f",
                     pack, dc, score)

    return results


def probe_sea_scripted(gray, script, available) -> dict:
    MIN_CHARS_MAP = {"Thai": 3, "Lao": 1, "Khmer": 2, "Myanmar": 2}
    pack_map      = {"Thai":"tha","Lao":"lao","Khmer":"khm","Myanmar":"mya"}
    score_mult    = {"Thai": 5, "Lao": 12, "Khmer": 6, "Myanmar": 6}
    pack = pack_map.get(script)
    if not pack or pack not in available:
        logger.debug("SEA pack for %s not available", script)
        return {}
    t  = ocr_string(gray, pack, 6) or ocr_string(gray, pack, 3)
    sc = sea_char_count(t, script)
    if sc < MIN_CHARS_MAP.get(script, 3):
        logger.debug("SEA %s probe: only %d script chars — not trusted",
                     script, sc)
        return {}
    total_chars = sum(1 for c in t if not c.isspace())
    ratio       = sc / total_chars if total_chars > 0 else 0
    min_ratio   = 0.08 if script == "Lao" else 0.15
    if ratio < min_ratio:
        logger.debug("SEA %s probe: ratio=%.2f < %.2f — likely noise",
                     script, ratio, min_ratio)
        return {}
    score = quality_score(t) + sc * score_mult.get(script, 5)
    logger.debug("SEA %s probe: chars=%d ratio=%.2f score=%.1f",
                 script, sc, ratio, score)
    return {pack: (score, t)}


def _disambiguate_lao_thai(gray, sea_results: dict, available: set) -> dict:
    tha_result = sea_results.get("tha")
    if not tha_result:
        return sea_results
    tha_text     = tha_result[1]
    lao_in_thai  = sea_char_count(tha_text, "Lao")
    thai_in_thai = sea_char_count(tha_text, "Thai")
    logger.debug("lao/thai disambiguate: lao=%d thai=%d",
                 lao_in_thai, thai_in_thai)
    if lao_in_thai >= 3 and lao_in_thai >= thai_in_thai * 0.4:
        lao_result = probe_sea_scripted(gray, "Lao", available)
        if lao_result:
            logger.info("Lao/Thai disambiguation: switching to Lao")
            out = dict(sea_results)
            out.update(lao_result)
            return out
    return sea_results


def _fallback_pack_priority(available: set) -> list:
    entries = [
        ("kor",     hangul_char_count,       3),
        ("jpn",     kana_char_count,         3),
        ("chi_sim", ideograph_char_count,    3),
        ("chi_tra", ideograph_char_count,    3),
        ("tha",     lambda t: sea_char_count(t, "Thai"),    3),
        ("lao",     lambda t: sea_char_count(t, "Lao"),     1),
        ("khm",     lambda t: sea_char_count(t, "Khmer"),   2),
        ("mya",     lambda t: sea_char_count(t, "Myanmar"), 2),
        ("ara",     arabic_char_count,       8),
        ("urd",     arabic_char_count,       8),
        ("hin",     devanagari_char_count,   6),
    ]
    return [(p, v, m) for p, v, m in entries if p in available]


def _langdetect_fallback(gray, seed_text: str, available: set,
                         skip_scripts: tuple = ()) -> dict:
    logger.warning("_langdetect_fallback triggered — ideal pack likely missing")
    for pack, validator, min_chars in _fallback_pack_priority(available):
        if pack in skip_scripts:
            continue
        t  = ocr_string(gray, pack, 6) or ocr_string(gray, pack, 3)
        sc = validator(t)
        logger.debug("  fallback pack=%s script_chars=%d min=%d",
                     pack, sc, min_chars)
        if sc >= min_chars:
            score = quality_score(t) + sc * 5
            logger.info("_langdetect_fallback winner: pack=%s chars=%d "
                        "score=%.1f", pack, sc, score)
            return {pack: (score, t)}
    try:
        ld = detect(seed_text) if seed_text.strip() else "en"
    except LangDetectException:
        ld = "en"
    lp = LANG_TO_PACK.get(ld, "eng")
    logger.warning("_langdetect_fallback last resort: detected=%s pack=%s",
                   ld, lp)
    if lp in available:
        t = ocr_string(gray, lp, 6) or ocr_string(gray, lp, 3)
        if t.strip():
            return {lp: (quality_score(t), t)}
    t = ocr_string(gray, "eng", 6)
    return {"eng": (quality_score(t), t)} if t.strip() else {}


# ── Latin early-exit ───────────────────────────────────────────────
_LATIN_SCRIPT_LANGS = {
    "en","fr","de","es","it","pt","nl","pl","sv","da","fi","et",
    "lv","lt","ro","hr","cs","sk","sl","hu","tr","id","ms","vi",
    "af","sw","tl","cy","gl","eu","ca","is","ga","mt",
}


def _check_latin_early_exit(gray, available: set) -> tuple:
    """
    [FIX-5] Mixed-script guard.
    Suppresses the Latin early-exit when the eng scan contains non-Latin
    chars, preventing English words on a non-Latin sign from bypassing
    the full probe cascade.
    """
    eng_text = ocr_string(gray, "eng", 6) or ocr_string(gray, "eng", 3)
    lat_r    = _latin_ratio(eng_text)

    _raw_cjk = sum(1 for c in eng_text if (
        0x3040 <= ord(c) <= 0x30FF or
        0x4E00 <= ord(c) <= 0x9FFF or
        0xAC00 <= ord(c) <= 0xD7AF
    ))
    _raw_ara = arabic_char_count(eng_text)
    _raw_sea = sum(1 for c in eng_text if (
        0x0E00 <= ord(c) <= 0x0EFF or
        0x1780 <= ord(c) <= 0x17FF or
        0x1000 <= ord(c) <= 0x109F
    ))
    _raw_dev = devanagari_char_count(eng_text)
    _has_non_latin = (_raw_cjk + _raw_ara + _raw_sea + _raw_dev) >= 2

    ld_hint: str | None = None
    if eng_text.strip():
        try:
            ld_hint = detect(eng_text)
        except LangDetectException:
            pass

    if lat_r >= 0.85 and not _has_non_latin:
        logger.debug("Latin early-exit STRONG: ratio=%.2f", lat_r)
        return True, eng_text, lat_r

    if lat_r >= 0.50 and ld_hint in _LATIN_SCRIPT_LANGS and not _has_non_latin:
        logger.debug("Latin early-exit WEAK: ratio=%.2f ld=%s", lat_r, ld_hint)
        return True, eng_text, lat_r

    logger.debug("Latin early-exit MISS: ratio=%.2f ld=%s non_latin=%s",
                 lat_r, ld_hint, _has_non_latin)
    return False, eng_text, lat_r


# ─────────────────────── Main detection ───────────────────────────

def detect_lang(gray, available):
    """
    Main language detection pipeline.

    English path: unchanged from doc-1.
    Non-Latin paths: probe_arabic_indic() replaces generic probe() for
    Arabic/Devanagari. langdetect cross-check min-token guard: 3 for
    non-Latin, 6 for Latin.
    """
    script = try_osd(gray)

    # Fast path retained from doc-1
    if script == "Latin":
        eng_text    = ocr_string(gray, "eng", 6)
        latin_ratio = _latin_ratio(eng_text)
        if (latin_ratio > 0.85
                and len(eng_text.split()) > 3
                and is_valid_text(eng_text)):
            return "eng", "en", eng_text

    logger.debug("OSD script=%s  available=%s", script, sorted(available))
    results: dict = {}

    if script and script not in ("Latin", None):
        # ── Non-Latin script branch ────────────────────────────────
        if script in ("Lao", "Thai", "Khmer", "Myanmar"):
            results = probe_sea_scripted(gray, script, available)
            if not results:
                logger.warning("SEA pack for %s missing — trying all SEA",
                               script)
                for s in ("Thai", "Lao", "Khmer", "Myanmar"):
                    results.update(probe_sea_scripted(gray, s, available))
            results = _disambiguate_lao_thai(gray, results, available)
            if not results:
                seed = (ocr_string(gray, "eng", 6)
                        or ocr_string(gray, "eng", 3))
                results = _langdetect_fallback(gray, seed, available,
                                               skip_scripts=("ara","urd"))

        elif script == "Hangul":
            kor_score, kor_text = probe_korean(gray, available)
            if kor_score > 0:
                results = {KOREAN_PACK: (kor_score, kor_text)}

        elif script in ("Japanese", "HanS", "HanT"):
            kor_score, kor_text = probe_korean(gray, available)
            if kor_score > 0:
                results[KOREAN_PACK] = (kor_score, kor_text)
            cjk_results = probe_cjk(gray, available)
            results.update(cjk_results)
            if not results:
                fallback_text = (ocr_string(gray, "eng", 6)
                                 or ocr_string(gray, "eng", 3))
                results = _langdetect_fallback(gray, fallback_text, available,
                                               skip_scripts=("ara","urd"))

        elif script == "Arabic":
            results = probe_arabic_indic(gray, available)
            if not results:
                seed = (ocr_string(gray, "eng", 6)
                        or ocr_string(gray, "eng", 3))
                results = _langdetect_fallback(
                    gray, seed, available,
                    skip_scripts=("jpn","chi_sim","chi_tra","kor"))

        elif script == "Devanagari":
            results = probe_arabic_indic(gray, available)
            if not results:
                seed = (ocr_string(gray, "eng", 6)
                        or ocr_string(gray, "eng", 3))
                results = _langdetect_fallback(gray, seed, available)

        else:
            packs = [p for p in SCRIPT_TO_PACKS.get(script, ["eng"])
                     if p in available]
            if not packs:
                packs = ["eng"]
            results = probe(gray, packs[:4], available)

        # Kannada guard — all non-Latin OSD paths
        if "kan" in available:
            kn_text  = (ocr_string(gray, "kan", 6)
                        or ocr_string(gray, "kan", 3))
            kn_chars = sum(1 for c in kn_text
                           if 0x0C80 <= ord(c) <= 0x0CFF)
            if kn_chars >= 5:
                score = quality_score(kn_text) + kn_chars * 5
                results["kan"] = (score, kn_text)

        if not results:
            logger.warning("Script=%s yielded no results — universal fallback",
                           script)
            seed = ocr_string(gray, "eng", 6) or ocr_string(gray, "eng", 3)
            results = _langdetect_fallback(gray, seed, available)
        if not results:
            results = probe(gray, ["eng"], available)

    else:
        # ── OSD returned Latin or failed ───────────────────────────
        is_latin, eng_text, lat_r = _check_latin_early_exit(gray, available)

        if is_latin:
            logger.info("Latin early-exit fired (ratio=%.2f) — skipping "
                        "CJK/SEA/Arabic", lat_r)
            lat_results = probe_parallel(
                gray,
                ["eng","fra","deu","spa","ita","por",
                 "nld","pol","rus","vie"],
                available,
            )
            if not lat_results:
                lat_results = {"eng": (quality_score(eng_text), eng_text)}
            results = lat_results

        else:
            # Full cascade — not clearly Latin

            # 1. Korean
            kor_score, kor_text = probe_korean(gray, available)
            if kor_score > 0:
                results[KOREAN_PACK] = (kor_score, kor_text)

            # 2. Kannada guard
            if "kan" in available:
                kn_text  = (ocr_string(gray, "kan", 6)
                            or ocr_string(gray, "kan", 3))
                kn_chars = sum(1 for c in kn_text
                               if 0x0C80 <= ord(c) <= 0x0CFF)
                if kn_chars >= 5:
                    score = quality_score(kn_text) + kn_chars * 5
                    results["kan"] = (score, kn_text)

            # 3. CJK
            cjk_results = probe_cjk(gray, available)
            results.update(cjk_results)
            logger.debug("CJK probe: %s",
                         {k: round(v[0], 1) for k, v in cjk_results.items()})

            # 4. SEA
            sea_results: dict = {}
            for s in ("Thai", "Lao", "Khmer", "Myanmar"):
                sea_results.update(probe_sea_scripted(gray, s, available))
            sea_results = _disambiguate_lao_thai(gray, sea_results, available)
            results.update(sea_results)
            sea_best = max((v[0] for v in sea_results.values()), default=0)

            specialised_best = max((v[0] for v in results.values()), default=0)

            _raw_cjk  = sum(1 for c in eng_text if (
                0x3040 <= ord(c) <= 0x30FF or
                0x4E00 <= ord(c) <= 0x9FFF or
                0xAC00 <= ord(c) <= 0xD7AF
            ))
            _raw_lao  = sum(1 for c in eng_text
                            if 0x0E80 <= ord(c) <= 0x0EFF)
            _raw_thai = sum(1 for c in eng_text
                            if 0x0E00 <= ord(c) <= 0x0E7F)
            _has_cjk_presence = _raw_cjk >= 2
            _has_sea_presence = (_raw_lao + _raw_thai) >= 2
            logger.debug("raw scan: cjk=%d lao=%d thai=%d",
                         _raw_cjk, _raw_lao, _raw_thai)

            if (_has_cjk_presence and sea_best > 0
                    and specialised_best == sea_best):
                logger.info("CJK chars in raw scan but SEA won — "
                            "overriding with CJK probe")
                for sea_pack in ("tha","lao","khm","mya"):
                    results.pop(sea_pack, None)
                specialised_best = max(
                    (v[0] for v in results.values()), default=0)

            if (_has_cjk_presence and specialised_best == 0
                    and "jpn" in available):
                t_retry, s_retry = _ocr_japanese(gray, available)
                kc_r = kana_char_count(t_retry)
                cc_r = cjk_char_count(t_retry)
                if ((kc_r >= 1 or cc_r >= 3)
                        and not _is_latin_dominant(t_retry, 0.60)):
                    score_r = (quality_score(t_retry)
                               + kc_r * 6 + cc_r * 3)
                    results["jpn"] = (score_r, t_retry)
                    specialised_best = score_r
                    logger.info("jpn retry: kana=%d cjk=%d score=%.1f",
                                kc_r, cc_r, score_r)

            # 5. Arabic/Indic — dedicated probe
            _ara_results: dict = {}
            _skip_ara = (specialised_best > 0
                         or _has_cjk_presence
                         or _has_sea_presence)
            if not _skip_ara:
                _ara_results = probe_arabic_indic(gray, available)

            # 6. Latin (parallel)
            lat_results = probe_parallel(
                gray,
                ["eng","fra","deu","spa","rus","vie"],
                available,
            )
            lat_results.update(_ara_results)
            lat_best = max((v[0] for v in lat_results.values()), default=0)
            results.update(lat_results)

            logger.debug("Latin best=%.1f  specialised best=%.1f",
                         lat_best, specialised_best)

            # 7. Universal fallback
            if specialised_best == 0 and lat_best < 5:
                logger.warning("All probes weak — universal fallback")
                fallback = _langdetect_fallback(gray, eng_text, available)
                results.update(fallback)

    if not results:
        return "eng", "en", ocr_string(gray, "eng", 3)

    bp     = max(results, key=lambda p: results[p][0])
    bs, bt = results[bp]
    lc     = PACK_TO_LANG.get(bp, "en")
    logger.debug("winner=%s lang=%s score=%.1f", bp, lc, bs)

    # [FIX-6] langdetect cross-check.
    # min_tokens: 3 for non-Latin (dense scripts), 6 for Latin.
    is_non_latin_pack = bp not in {
        "eng","fra","deu","spa","por","ita","nld","pol","rus","vie",
        "ind","tur","afr","ces","dan","fin","hun","ron","slk","slv",
        "swe","ukr"
    }
    min_tokens = 3 if is_non_latin_pack else 6
    if bt.strip() and len(bt.split()) >= min_tokens:
        try:
            ld = detect(bt)
            lp = LANG_TO_PACK.get(ld, "eng")
            alt_score, alt_text = results.get(lp, (0, ""))
            if alt_score >= bs * 0.85 and alt_text.strip():
                lc = ld
                bt = alt_text
                logger.debug("langdetect override -> lang=%s score=%.1f",
                             lc, alt_score)
            else:
                logger.debug("langdetect=%s rejected (alt=%.1f < %.1f)",
                             ld, alt_score, bs * 0.85)
        except LangDetectException:
            pass

    # Final SEA sanity check
    SEA_PACKS_SET = {"tha","lao","khm","mya"}
    if bp in SEA_PACKS_SET:
        win_ratio = _latin_ratio(bt)
        if win_ratio >= 0.60:
            latin_packs = ["eng","fra","deu","spa","por","ita",
                           "nld","pol","vie","ind"]
            best_lat_pack  = None
            best_lat_score = 0
            for lp in latin_packs:
                if lp in results and results[lp][0] > best_lat_score:
                    best_lat_score = results[lp][0]
                    best_lat_pack  = lp
            if best_lat_pack:
                logger.info("SEA winner %s had Latin text (ratio=%.2f) "
                            "-> override %s", bp, win_ratio, best_lat_pack)
                bp = best_lat_pack
                bs, bt = results[bp]
                lc = PACK_TO_LANG.get(bp, "en")

    return bp, lc, bt


# ──────────────────────── Line extraction ─────────────────────────

# [FIX-2] Per-pack PSM order and merge thresholds.
_PSM_ORDER = {
    "jpn":     (5, 6, 3),
    "chi_sim": (6, 3),
    "chi_tra": (6, 3),
    "kor":     (6, 3),
    "ara":     (6, 3, 4),
    "urd":     (6, 3, 4),
    "fas":     (6, 3, 4),
    "tha":     (6, 3),
    "lao":     (6, 3),
    "khm":     (6, 3),
    "mya":     (6, 3),
    "hin":     (6, 3),
    "mar":     (6, 3),
    "nep":     (6, 3),
}
_DEFAULT_PSM_ORDER = (6, 11)


def get_lines(gray_up, pack, scale_x, scale_y, W, H):
    """
    Extract line bounding boxes and map back to original coordinates.
    Per-pack PSM order and merge thresholds applied.
    """
    psm_order = _PSM_ORDER.get(pack, _DEFAULT_PSM_ORDER)

    if pack in _RTL_PACKS:
        section_mult, block_mult = 0.5, 0.25
    elif pack in ("tha","lao","khm","mya"):
        section_mult, block_mult = 0.5, 0.2
    elif pack in ("jpn","chi_sim","chi_tra","kor"):
        section_mult, block_mult = 0.6, 0.3
    else:
        section_mult, block_mult = 0.6, 0.3   # Latin — same as doc-1

    for psm in psm_order:
        try:
            data = pytesseract.image_to_data(
                gray_up, lang=pack,
                output_type=pytesseract.Output.DICT,
                config=f"--psm {psm} --oem 1",
            )
        except Exception as e:
            logger.debug("get_lines psm=%d failed: %s", psm, e)
            continue

        n        = len(data["text"])
        line_map = {}
        for i in range(n):
            try:
                conf = int(data["conf"][i])
            except Exception:
                conf = -1
            word = str(data["text"][i]).strip()
            if conf < 25 or not word:
                continue
            key = (data["block_num"][i], data["par_num"][i],
                   data["line_num"][i])
            line_map.setdefault(key, []).append(i)

        if not line_map:
            continue

        raw = []
        for idxs in line_map.values():
            x1u = min(data["left"][i]                     for i in idxs)
            y1u = min(data["top"][i]                      for i in idxs)
            x2u = max(data["left"][i] + data["width"][i]  for i in idxs)
            y2u = max(data["top"][i]  + data["height"][i] for i in idxs)
            src = " ".join(str(data["text"][i]) for i in idxs).strip()
            x1  = max(0, int(x1u / scale_x))
            y1  = max(0, int(y1u / scale_y))
            x2  = min(W, int(x2u / scale_x))
            y2  = min(H, int(y2u / scale_y))
            if x2 - x1 < 6 or y2 - y1 < 4 or not src:
                continue
            raw.append(dict(
                x1=x1, y1=y1, x2=x2, y2=y2, src=src,
                bk=data["block_num"][idxs[0]],
                par=data["par_num"][idxs[0]],
                ln=data["line_num"][idxs[0]],
            ))

        if not raw:
            continue

        raw.sort(key=lambda l: (l["bk"], l["par"], l["y1"]))

        merged: list = []
        cur = None
        for ln in raw:
            if cur is None:
                cur = dict(**ln)
                cur["lines"] = [ln["src"]]
                continue
            same_block   = (ln["bk"] == cur["bk"])
            same_section = same_block and (ln["par"] == cur["par"])
            gap   = ln["y1"] - cur["y2"]
            avg_h = (cur["y2"] - cur["y1"] + ln["y2"] - ln["y1"]) / 2
            should_merge = (
                (same_section and -2 <= gap <= avg_h * section_mult) or
                (same_block   and  0 <= gap <= avg_h * block_mult)
            )
            if should_merge:
                cur["x1"] = min(cur["x1"], ln["x1"])
                cur["x2"] = max(cur["x2"], ln["x2"])
                cur["y2"] = max(cur["y2"], ln["y2"])
                cur["lines"].append(ln["src"])
            else:
                cur["src"] = " ".join(cur["lines"])
                merged.append(cur)
                cur = dict(**ln)
                cur["lines"] = [ln["src"]]
        if cur:
            cur["src"] = " ".join(cur["lines"])
            merged.append(cur)

        logger.debug("bbox psm %d -> %d line-blocks", psm, len(merged))
        return merged

    logger.debug("bbox: all psm modes returned 0 lines")
    return []


# ──────────────────────────── Overlay ─────────────────────────────

def fit_text(draw, text, box_w, box_h, min_fs=8, max_fs=None):
    if max_fs is None:
        max_fs = max(min_fs, int(box_h * 0.82))
    max_fs = max(min_fs, max_fs)
    best   = None
    lo, hi = min_fs, max_fs
    while lo <= hi:
        fs    = (lo + hi) // 2
        font  = load_font(fs)
        lh    = draw.textbbox((0, 0), "Ag", font=font)[3] + 2
        words = text.split()
        lines: list = []
        line:  list = []
        for w in words:
            test = " ".join(line + [w])
            if draw.textbbox((0, 0), test, font=font)[2] > box_w - 6:
                if line:
                    lines.append(" ".join(line))
                    line = [w]
                else:
                    lines.append(w)
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        total_h = len(lines) * lh
        if total_h <= box_h:
            best = (font, lines, lh, total_h)
            lo   = fs + 1
        else:
            hi = fs - 1
    if best:
        return best
    font = load_font(min_fs)
    lh   = draw.textbbox((0, 0), "Ag", font=font)[3] + 2
    return font, [text], lh, lh


def _prepare_rtl_text(text: str) -> str:
    """
    [FIX-3] Reshape + BiDi for Arabic/Urdu so Pillow renders correctly.
    Graceful no-op if python-bidi / arabic-reshaper not installed.
    """
    if not _HAS_BIDI:
        return text
    try:
        return bidi_display(arabic_reshape(text))
    except Exception:
        return text


def render_overlay(orig_path, gray_layout, scale_x, scale_y,
                   pack, lang_code, final_text,
                   precomputed_lines=None, precomputed_translations=None):
    """
    [PERF] Two optimisations vs previous version:

    1. precomputed_lines: caller can pass the get_lines result from a
       previous call (e.g. detect_lang already ran image_to_data).
       Eliminates a redundant full Tesseract inference pass.

    2. precomputed_translations: caller can pass already-translated strings
       (from translate_full in process_image) so batch_translate is not
       called a second time for the same content. Eliminates a duplicate
       Google Translate HTTP round-trip.

    Both parameters are optional — render_overlay falls back gracefully
    when not provided (e.g. called directly in tests or PDF path).
    """
    orig = cv2.imread(orig_path)
    pil  = Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)).convert("RGBA")
    W, H = pil.size

    # [PERF] Reuse precomputed lines when available — avoids a full
    # image_to_data Tesseract pass that duplicates work from detect_lang.
    lines = (precomputed_lines
             if precomputed_lines is not None
             else get_lines(gray_layout, pack, scale_x, scale_y, W, H))

    # [PERF] Reuse precomputed per-line translations when available.
    # process_image passes these to avoid a duplicate Google Translate call.
    if precomputed_translations is not None:
        translated_lines = precomputed_translations
    else:
        try:
            translated_lines = batch_translate([l["src"] for l in lines])
        except Exception:
            translated_lines = [l["src"] for l in lines]

    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    is_rtl = pack in _RTL_PACKS

    # Per-script minimum readable font size
    if pack in (_RTL_PACKS | {"hin","mar","nep","ben","tam","tel","kan","mal"}):
        min_readable_fs = max(14, H // 55)
    elif pack in ("jpn","chi_sim","chi_tra","kor"):
        min_readable_fs = max(13, H // 58)
    else:
        min_readable_fs = max(13, H // 60)   # Latin

    # Global font cap (from doc-2)
    global_max_fs = max(16, min(56, W // 36))

    # Sort top-to-bottom; overlap guard
    lines_sorted = sorted(
        [(i, ln) for i, ln in enumerate(lines)],
        key=lambda x: x[1]["y1"]
    )
    last_drawn_y2 = -1

    for i, ln in lines_sorted:
        x1, y1, x2, y2 = ln["x1"], ln["y1"], ln["x2"], ln["y2"]
        # FIX: cap box width at 90% of image width (from doc-5, good addition)
        bw_px = min(x2 - x1, int(W * 0.9))
        bh_px = y2 - y1

        raw_tr = (translated_lines[i]
                  if i < len(translated_lines) else ln["src"])
        txt = (raw_tr or "").strip()
        if not txt:
            continue

        if bw_px < 20 or bh_px < 8:
            continue

        # Overlap guard
        y1 = max(y1, last_drawn_y2 + 2)
        if y1 >= H:
            continue

        max_fs = min(global_max_fs,
                     max(min_readable_fs, int(bh_px * 0.85)))
        min_fs = min_readable_fs

        display_txt = _prepare_rtl_text(txt) if is_rtl else txt

        font, wrapped, lh, total_h = fit_text(
            draw, display_txt, bw_px, bh_px,
            min_fs=min_fs, max_fs=max_fs,
        )

        box_h     = max(bh_px, total_h + 8)
        actual_y2 = min(H, y1 + box_h)

        draw.rectangle([(x1, y1), (x2, actual_y2)], fill=(10, 10, 20, 215))

        ty = y1 + max(4, (actual_y2 - y1 - total_h) // 2)
        for line_text in wrapped:
            if ty + lh > actual_y2:
                break
            for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
                draw.text((x1 + 5 + dx, ty + dy), line_text,
                          font=font, fill=(0, 0, 0, 200))
            draw.text((x1 + 5, ty), line_text,
                      font=font, fill=(255, 255, 255, 255))
            ty += lh

        last_drawn_y2 = actual_y2

    out  = Image.alpha_composite(pil, overlay).convert("RGB")
    path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.jpg")
    out.save(path, quality=93)
    return path


def _banner(pil, W, H, raw, lang_code):
    """Fallback full-image banner when no bounding boxes are found."""
    tr   = raw if lang_code == "en" else translate_full(raw)
    ov   = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    fs   = max(13, min(20, W // 38))
    lh   = fs + 7
    font = load_font(fs)
    sm   = load_font(max(10, fs - 3))

    is_rtl = lang_code in {PACK_TO_LANG.get(p, "") for p in _RTL_PACKS}
    display_tr = _prepare_rtl_text(tr) if is_rtl else tr

    words = display_tr.split()
    lines: list = []
    line:  list = []
    for w in words:
        t = " ".join(line + [w])
        if draw.textbbox((0, 0), t, font=font)[2] > W - 32:
            if line:
                lines.append(" ".join(line))
                line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))

    ml = min(len(lines), max(3, int(H * 0.40 / lh)))
    bh = 30 + ml * lh + 14
    bt = H - bh
    draw.rectangle([(0, bt), (W, H)], fill=(8, 8, 24, 225))
    draw.line([(0, bt), (W, bt)], fill=(100, 200, 255, 160), width=2)
    draw.text((16, bt + 8), "TRANSLATED (EN)", font=sm,
              fill=(100, 200, 255, 220))
    y = bt + 28
    for ln in lines[:ml]:
        draw.text((16, y), ln, font=font, fill=(255, 255, 255, 255))
        y += lh

    clipped = len(lines) - ml
    if clipped > 0:
        clip_note = f"... (+{clipped} more line{'s' if clipped != 1 else ''})"
        draw.text((16, y - lh + fs), clip_note,
                  font=sm, fill=(100, 200, 255, 180))

    path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.jpg")
    Image.alpha_composite(pil, ov).convert("RGB").save(path, quality=93)
    return path


# ─────────────────────────────── Main ─────────────────────────────

def process_image(image_path: str) -> dict:
    """Main entry point. English path fully retained from doc-1."""
    result = {
        "extracted_text":    "",
        "translated_text":   "",
        "translated_image":  "",
        "detected_language": "unknown",
        "error":             None,
    }
    img = cv2.imread(image_path)
    if img is None:
        result["error"] = "Could not read image."
        return result

    h0, w0 = img.shape[:2]
    TARGET  = 1400

    if max(h0, w0) < TARGET:
        sc     = TARGET / max(h0, w0)
        img_up = cv2.resize(img, None, fx=sc, fy=sc,
                            interpolation=cv2.INTER_CUBIC)
    else:
        sc     = 1.0
        img_up = img.copy()

    sx       = img_up.shape[1] / w0
    sy       = img_up.shape[0] / h0
    gray_raw = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)

    gray_clahe = _CLAHE.apply(gray_raw)

    if is_likely_handwritten(gray_raw):
        logger.info("Handwriting detected — adaptive threshold for layout")
        gray_for_layout = preprocess_for_handwriting(gray_raw)
    else:
        gray_for_layout = gray_clahe

    available = get_available_langs()

    pack, lc, extracted = detect_lang(gray_clahe, available)
    result["detected_language"] = lc

    if not extracted.strip():
        result["error"] = "No readable text found."
        return result

    result["extracted_text"]  = extracted

    # [PERF] Run get_lines ONCE here so render_overlay can reuse the result
    # instead of calling image_to_data a second time from scratch.
    # W/H are read from the upscaled image dimensions.
    H_up, W_up = gray_for_layout.shape[:2]
    lines = get_lines(gray_for_layout, pack, sx, sy, W_up, H_up)

    # [PERF] Translate once — per-line for the overlay, joined for the
    # translated_text field. This replaces the previous pattern of:
    #   translate_full(extracted)      ← 1st HTTP call
    #   batch_translate(line sources)  ← 2nd HTTP call (same content)
    # Now we make a single batch_translate call and join for the text field.
    if lc == "en" and pack == "eng":
        translated_lines     = [l["src"] for l in lines]
        result["translated_text"] = extracted
    else:
        try:
            translated_lines = batch_translate([l["src"] for l in lines])
        except Exception:
            translated_lines = [l["src"] for l in lines]
        # Join translated lines as the full translated text.
        # Falls back to translate_full only when no lines were extracted
        # (e.g. get_lines returned empty — banner fallback path).
        if translated_lines and any(t.strip() for t in translated_lines):
            result["translated_text"] = " ".join(
                t for t in translated_lines if t.strip()
            )
        else:
            result["translated_text"] = translate_full(extracted)

    try:
        result["translated_image"] = render_overlay(
            image_path, gray_for_layout, sx, sy, pack, lc,
            result["translated_text"],
            precomputed_lines=lines,
            precomputed_translations=translated_lines,
        )
    except Exception as e:
        logger.exception("Render failed")
        result["error"] = f"Render failed: {e}"

    return result


# ─────────────────────── PDF processing ───────────────────────────

def _extract_pdf_text_direct(pdf_path: str, max_pages: int) -> list | None:
    try:
        import fitz
    except ImportError:
        return None
    try:
        doc   = fitz.open(pdf_path)
        pages = min(len(doc), max_pages)
        has_text = any(
            len(doc[i].get_text("text").strip()) >= 20
            for i in range(pages)
        )
        if not has_text:
            doc.close()
            return None

        results = []
        for i in range(pages):
            text     = doc[i].get_text("text").strip()
            mat      = fitz.Matrix(2.0, 2.0)
            pix      = doc[i].get_pixmap(matrix=mat)
            base     = os.path.splitext(pdf_path)[0]
            img_path = f"{base}_page{i}.png"
            pix.save(img_path)

            if not text:
                results.append({
                    "page": i + 1,
                    "extracted_text":    "",
                    "translated_text":   "",
                    "translated_image":  "",
                    "detected_language": "unknown",
                    "error":             "No text on this page.",
                    "_saved_image_path": img_path,
                })
                continue

            try:
                ld = detect(text)
            except LangDetectException:
                ld = "en"

            translated = text if ld == "en" else translate_full(text)
            results.append({
                "page": i + 1,
                "extracted_text":    text,
                "translated_text":   translated,
                "translated_image":  "",
                "detected_language": ld,
                "error":             None,
                "_saved_image_path": img_path,
            })

        doc.close()
        return results

    except Exception as e:
        logger.warning("Direct PDF text extraction failed: %s", e)
        return None


def process_pdf(pdf_path: str, max_pages: int = 10) -> list:
    try:
        import fitz  # noqa: F401
    except ImportError:
        logger.error("PyMuPDF not installed — pip install pymupdf")
        return [{
            "page": 1,
            "error": "PyMuPDF not installed.",
            "extracted_text": "",
            "translated_text": "",
            "translated_image": "",
            "detected_language": "unknown",
            "_saved_image_path": "",
        }]

    direct = _extract_pdf_text_direct(pdf_path, max_pages)
    if direct is not None:
        logger.info("PDF: direct text extraction succeeded (%d pages)",
                    len(direct))
        return direct

    logger.info("PDF: no embedded text — falling back to OCR per page")
    results = []
    try:
        import fitz
        doc   = fitz.open(pdf_path)
        pages = min(len(doc), max_pages)
        base  = os.path.splitext(pdf_path)[0]

        for i in range(pages):
            page     = doc[i]
            mat      = fitz.Matrix(2.0, 2.0)
            pix      = page.get_pixmap(matrix=mat)
            img_path = f"{base}_page{i}.png"
            pix.save(img_path)

            res = process_image(img_path)
            res["_saved_image_path"] = img_path
            results.append({"page": i + 1, **res})

        doc.close()
    except Exception as e:
        logger.exception("PDF processing failed")
        results.append({
            "page": 1,
            "error": str(e),
            "extracted_text": "",
            "translated_text": "",
            "translated_image": "",
            "detected_language": "unknown",
            "_saved_image_path": "",
        })
    return results