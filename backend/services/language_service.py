# language_service.py
# Language detection using Unicode block analysis as primary strategy,
# with langdetect as fallback for ambiguous Latin-script languages.
# Input: raw image (numpy array or PIL) OR extracted text string
# Output: LanguageResult dataclass with ISO code, script, direction, font hint

from __future__ import annotations
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LanguageResult:
    iso_code: str                          # ISO 639-1 e.g. "lo", "th", "ar"
    language_name: str                     # Human-readable e.g. "Lao"
    script: str                            # Unicode script family e.g. "Lao", "Arabic"
    direction: Literal["ltr", "rtl"]       # Text direction
    noto_font: str                         # Corresponding Noto font name
    confidence: Literal["high", "low"]     # Detection confidence


# ---------------------------------------------------------------------------
# Unicode block → language metadata mapping
# Each tuple: (range_start, range_end, iso_code, language_name, script, direction, noto_font)
# ---------------------------------------------------------------------------

UNICODE_BLOCK_MAP: list[tuple] = [
    (0x0E80, 0x0EFF, "lo", "Lao",                  "Lao",        "ltr", "NotoSansLao"),
    (0x0E00, 0x0E7F, "th", "Thai",                 "Thai",       "ltr", "NotoSansThai"),
    (0x0600, 0x06FF, "ar", "Arabic",               "Arabic",     "rtl", "NotoSansArabic"),
    (0x0900, 0x097F, "hi", "Hindi",                "Devanagari", "ltr", "NotoSansDevanagari"),
    (0x0980, 0x09FF, "bn", "Bengali",              "Bengali",    "ltr", "NotoSansBengali"),
    (0x0A80, 0x0AFF, "gu", "Gujarati",             "Gujarati",   "ltr", "NotoSansGujarati"),
    (0x0B00, 0x0B7F, "or", "Odia",                 "Odia",       "ltr", "NotoSansOriya"),
    (0x0B80, 0x0BFF, "ta", "Tamil",                "Tamil",      "ltr", "NotoSansTamil"),
    (0x0C00, 0x0C7F, "te", "Telugu",               "Telugu",     "ltr", "NotoSansTelugu"),
    (0x0C80, 0x0CFF, "kn", "Kannada",              "Kannada",    "ltr", "NotoSansKannada"),
    (0x0D00, 0x0D7F, "ml", "Malayalam",            "Malayalam",  "ltr", "NotoSansMalayalam"),
    (0x0400, 0x04FF, "ru", "Russian",              "Cyrillic",   "ltr", "NotoSans"),
    (0x0500, 0x052F, "ru", "Russian (Supplement)", "Cyrillic",   "ltr", "NotoSans"),
    (0x4E00, 0x9FFF, "zh", "Chinese",              "CJK",        "ltr", "NotoSansCJK"),
    (0x3040, 0x309F, "ja", "Japanese (Hiragana)",  "Hiragana",   "ltr", "NotoSansCJK"),
    (0x30A0, 0x30FF, "ja", "Japanese (Katakana)",  "Katakana",   "ltr", "NotoSansCJK"),
    (0xAC00, 0xD7AF, "ko", "Korean",               "Hangul",     "ltr", "NotoSansCJK"),
    (0x0370, 0x03FF, "el", "Greek",                "Greek",      "ltr", "NotoSans"),
    (0x0590, 0x05FF, "he", "Hebrew",               "Hebrew",     "rtl", "NotoSansHebrew"),
    (0x0700, 0x074F, "syc","Syriac",               "Syriac",     "rtl", "NotoSansSyriac"),
    (0x1000, 0x109F, "my", "Burmese",              "Myanmar",    "ltr", "NotoSansMyanmar"),
    (0x0F00, 0x0FFF, "bo", "Tibetan",              "Tibetan",    "ltr", "NotoSerifTibetan"),
    (0x1780, 0x17FF, "km", "Khmer",                "Khmer",      "ltr", "NotoSansKhmer"),
    (0xFB50, 0xFDFF, "ar", "Arabic Extended",      "Arabic",     "rtl", "NotoSansArabic"),
    (0x0020, 0x007F, "en", "English/Latin",        "Latin",      "ltr", "NotoSans"),  # ASCII fallback
]

# Default fallback
_DEFAULT = LanguageResult(
    iso_code="en",
    language_name="English/Latin",
    script="Latin",
    direction="ltr",
    noto_font="NotoSans",
    confidence="low"
)


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------

def _detect_by_unicode_blocks(text: str) -> LanguageResult | None:
    """
    Analyse the Unicode block distribution of characters in the text.
    Returns a LanguageResult if a non-Latin block dominates (>30% of chars),
    otherwise returns None to signal fallback to langdetect.
    """
    if not text or not text.strip():
        return None

    block_votes: Counter = Counter()

    for char in text:
        cp = ord(char)
        for (start, end, iso, name, script, direction, font) in UNICODE_BLOCK_MAP:
            if start <= cp <= end:
                block_votes[iso] += 1
                break

    if not block_votes:
        return None

    total_chars = sum(block_votes.values())
    dominant_iso, dominant_count = block_votes.most_common(1)[0]
    dominance_ratio = dominant_count / total_chars

    # High-confidence: non-Latin script dominates
    if dominant_iso != "en" and dominance_ratio > 0.30:
        for (start, end, iso, name, script, direction, font) in UNICODE_BLOCK_MAP:
            if iso == dominant_iso:
                return LanguageResult(
                    iso_code=iso,
                    language_name=name,
                    script=script,
                    direction=direction,
                    noto_font=font,
                    confidence="high"
                )

    return None  # Ambiguous — likely Latin-based, use langdetect


def _detect_by_langdetect(text: str) -> LanguageResult:
    """
    Fallback for Latin-script languages (en, fr, de, es, pt, etc.)
    where Unicode block analysis can't disambiguate.
    """
    try:
        from langdetect import detect as ld_detect
        iso = ld_detect(text)
    except Exception:
        return _DEFAULT

    # Map common Latin-script ISO codes to minimal metadata
    _LATIN_MAP: dict[str, tuple[str, str]] = {
        "en": ("English",    "NotoSans"),
        "fr": ("French",     "NotoSans"),
        "de": ("German",     "NotoSans"),
        "es": ("Spanish",    "NotoSans"),
        "pt": ("Portuguese", "NotoSans"),
        "it": ("Italian",    "NotoSans"),
        "nl": ("Dutch",      "NotoSans"),
        "pl": ("Polish",     "NotoSans"),
        "vi": ("Vietnamese", "NotoSans"),  # Latin-based but tonal
        "id": ("Indonesian", "NotoSans"),
    }

    name, font = _LATIN_MAP.get(iso, ("Unknown Latin", "NotoSans"))
    return LanguageResult(
        iso_code=iso,
        language_name=name,
        script="Latin",
        direction="ltr",
        noto_font=font,
        confidence="low"   # langdetect is probabilistic
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_language(text: str) -> LanguageResult:
    """
    Primary entry point.
    
    Strategy:
      1. Unicode block analysis  → high confidence for non-Latin scripts
      2. langdetect fallback     → probabilistic, for Latin-script disambiguation
      3. Hard default to English → if all else fails, never crashes
    
    Args:
        text: Raw string — can be OCR output, partial text, or even garbled.
              For image-first detection, pass a sample of extracted characters.
    
    Returns:
        LanguageResult with iso_code, script, direction, noto_font, confidence
    """
    result = _detect_by_unicode_blocks(text)
    if result is not None:
        return result

    return _detect_by_langdetect(text)