import cv2
import logging
import pytesseract
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory, LangDetectException
from PIL import Image, ImageDraw, ImageFont
import os, uuid, pathlib, unicodedata, re, numpy as np

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

# Homebrew tesseract binary candidates (Apple Silicon first, then Intel)
_HOMEBREW_TESSERACT_CANDIDATES = [
    "/opt/homebrew/bin/tesseract",   # macOS Apple Silicon
    "/usr/local/bin/tesseract",      # macOS Intel
]
# Homebrew tessdata directory candidates
_HOMEBREW_TESSDATA_CANDIDATES = [
    "/opt/homebrew/share/tessdata",
    "/usr/local/share/tessdata",
    "/opt/homebrew/Cellar/tesseract-lang/4.1.0/share/tessdata",
    "/usr/local/Cellar/tesseract-lang/4.1.0/share/tessdata",
]


def _configure_tesseract():
    """
    Auto-detect the best tesseract binary and tessdata directory.

    On macOS the system /usr/bin/tesseract only ships eng+osd.
    Homebrew installs a full binary + tesseract-lang packs at a separate
    prefix. We prefer the Homebrew binary if it exists.

    Environment variables always take precedence:
      TESSERACT_CMD     — full path to tesseract binary
      TESSDATA_PREFIX   — path to tessdata directory
    """
    import shutil

    # 1. Binary — env var beats everything
    cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if cmd and os.path.isfile(cmd):
        pytesseract.pytesseract.tesseract_cmd = cmd
        logger.info("tesseract binary (env): %s", cmd)
    else:
        # Try Homebrew candidates
        for candidate in _HOMEBREW_TESSERACT_CANDIDATES:
            if os.path.isfile(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                logger.info("tesseract binary (homebrew): %s", candidate)
                break
        else:
            # Fall back to PATH resolution
            found = shutil.which("tesseract")
            if found:
                pytesseract.pytesseract.tesseract_cmd = found
                logger.info("tesseract binary (PATH): %s", found)

    # 2. Tessdata dir — env var beats everything
    tdata = os.environ.get("TESSDATA_PREFIX", "").strip()
    if tdata and os.path.isdir(tdata):
        os.environ["TESSDATA_PREFIX"] = tdata
        logger.info("tessdata (env): %s", tdata)
    else:
        for candidate in _HOMEBREW_TESSDATA_CANDIDATES:
            if os.path.isdir(candidate):
                # Only use it if it has more than eng+osd
                files = os.listdir(candidate)
                packs = [f[:-13] for f in files if f.endswith(".traineddata")]
                if len(packs) > 5:
                    os.environ["TESSDATA_PREFIX"] = candidate
                    logger.info("tessdata (homebrew, %d packs): %s", len(packs), candidate)
                    break
        else:
            logger.debug("tessdata: using system default")


def _check_packs_on_startup() -> set:
    _configure_tesseract()
    try:
        installed = set(pytesseract.get_languages(config=""))
        missing   = {p: lang for p, lang in _CRITICAL_PACKS.items() if p not in installed}
        if missing:
            logger.warning(
                "Missing Tesseract language packs: %s\n"
                "  macOS:  brew install tesseract-lang\n"
                "  Ubuntu: sudo apt-get install tesseract-ocr-all\n"
                "  Then restart the server.",
                ", ".join(f"{p} ({lang})" for p, lang in missing.items()),
            )
        else:
            logger.info("All critical Tesseract packs are installed (%d total).", len(installed))
        return installed
    except Exception as e:
        logger.error("Failed to query Tesseract languages: %s", e)
        return {"eng"}


_AVAILABLE_LANGS_CACHE: set | None = None

def get_available_langs() -> set:
    global _AVAILABLE_LANGS_CACHE
    if _AVAILABLE_LANGS_CACHE is None:
        _AVAILABLE_LANGS_CACHE = _check_packs_on_startup()
    return _AVAILABLE_LANGS_CACHE

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
    "Japanese":["jpn"],"HanS":["chi_sim"],"HanT":["chi_tra"],"Hangul":["kor"],
    "Arabic":["ara","fas","urd"],"Devanagari":["hin","mar","nep"],
    "Bengali":["ben"],"Tamil":["tam"],"Telugu":["tel"],"Kannada":["kan"],
    "Malayalam":["mal"],"Gujarati":["guj"],"Gurmukhi":["pan"],
    "Cyrillic":["rus","ukr","bul"],"Greek":["ell"],"Armenian":["hye"],
    "Hebrew":["heb"],"Georgian":["kat"],"Ethiopic":["amh"],
    "Latin":["eng","fra","deu","spa","por","ita","nld","pol","vie","ind","tur"],
}
SEA_PACKS = ["lao","tha","khm","mya"]
# Korean kept separate from CJK — it needs its own scoring path
CJK_PACKS     = ["jpn","chi_sim","chi_tra"]
KOREAN_PACK   = "kor"

# ── Font resolution (cross-platform) ──────────────────────────────
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

def load_font(size: int) -> ImageFont.FreeTypeFont:
    size = max(8, int(size))
    for fp in _FONT_SEARCH_PATHS:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ─────────────────────── helpers ──────────────────────────────────
def quality_score(text):
    if not text or not text.strip(): return 0
    tokens = text.split()
    if not tokens: return 0
    sc = sum(1 for c in text if ord(c)>127 and not c.isdigit()
             and unicodedata.category(c) not in
             ("Po","Ps","Pe","Pi","Pf","Pd","Zs","Cc","Cf"))
    rw = sum(1 for t in tokens if len(t)>=2 and re.search(r'[a-zA-Z\u0080-\uFFFF]',t))
    sr = sum(1 for t in tokens if len(t)==1)/len(tokens)
    return (sc*2+rw)*max(0.2,1.0-sr*1.5)

# ── Script-specific character counters ────────────────────────────
def cjk_char_count(text):
    """Count all CJK-family characters (generic presence check, includes Hangul)."""
    return sum(1 for c in text if (
        0x3040<=ord(c)<=0x30FF or  # Hiragana + Katakana
        0x4E00<=ord(c)<=0x9FFF or  # CJK Unified Ideographs
        0xAC00<=ord(c)<=0xD7AF or  # Hangul Syllables
        0x3400<=ord(c)<=0x4DBF     # CJK Extension A
    ))

def ideograph_char_count(text):
    """Count CJK ideographs ONLY — excludes Hangul and Kana.
    Used for Chinese scoring so Korean text does not inflate chi_sim scores."""
    return sum(1 for c in text if (
        0x4E00<=ord(c)<=0x9FFF or  # CJK Unified Ideographs
        0x3400<=ord(c)<=0x4DBF     # CJK Extension A
    ))

def hangul_char_count(text):
    """Count Korean Hangul syllable blocks exclusively (0xAC00–0xD7AF)."""
    return sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7AF)

def kana_char_count(text):
    """Count Japanese Hiragana + Katakana exclusively (0x3040–0x30FF)."""
    return sum(1 for c in text if 0x3040 <= ord(c) <= 0x30FF)

def thai_char_count(text):
    """Count Thai Unicode characters (0x0E00–0x0E7F)."""
    return sum(1 for c in text if 0x0E00 <= ord(c) <= 0x0E7F)

def sea_char_count(text, script):
    """Count characters for a given SEA script by Unicode range."""
    ranges = {
        "Thai":    (0x0E00, 0x0E7F),
        "Lao":     (0x0E80, 0x0EFF),
        "Khmer":   (0x1780, 0x17FF),
        "Myanmar": (0x1000, 0x109F),
    }
    lo, hi = ranges.get(script, (0, 0))
    return sum(1 for c in text if lo <= ord(c) <= hi)


def ocr_string(gray, pack, psm=6):
    try:
        d = pytesseract.image_to_data(gray, lang=pack,
            output_type=pytesseract.Output.DICT,
            config=f"--psm {psm} --oem 1")
        return " ".join(d["text"][i] for i in range(len(d["text"]))
                        if int(d["conf"][i])>20 and d["text"][i].strip())
    except: return ""

# ── Translation helpers ────────────────────────────────────────────
def _translator() -> GoogleTranslator:
    return GoogleTranslator(source="auto", target="en")

def translate_one(src: str) -> str:
    try:
        t = _translator().translate(src)
        return t or src
    except Exception:
        return src

def translate_full(text: str) -> str:
    if not text.strip(): return text
    try:
        chunks, chunk = [], ""
        for w in text.split():
            if len(chunk)+len(w)+1 > 4500:
                chunks.append(chunk.strip()); chunk = w
            else:
                chunk += " "+w
        if chunk: chunks.append(chunk.strip())
        return " ".join(_translator().translate(c) for c in chunks if c)
    except Exception:
        return text

def batch_translate(texts: list) -> list:
    if not texts: return []
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

# ── Image validation ───────────────────────────────────────────────
def validate_image_bytes(data: bytes) -> bool:
    if data[:3] == b"\xff\xd8\xff":           return True  # JPEG
    if data[:8] == b"\x89PNG\r\n\x1a\n":      return True  # PNG
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP": return True  # WebP
    return False


# ───────────────────────── detection ──────────────────────────────
def try_osd(gray):
    """
    Ask Tesseract OSD what script the image uses.
    Returns a script name string, or None if detection fails.
    Tries multiple preprocessings to maximise success rate.
    """
    h, w = gray.shape
    small = cv2.resize(gray, (min(1400,w), min(1000,h)))
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    found_latin = False
    for img in [bw, 255-bw, small]:
        try:
            osd = pytesseract.image_to_osd(img, config="--psm 0 --oem 1",
                                            output_type=pytesseract.Output.DICT)
            s = osd.get("script","")
            if s and s != "Latin": return s
            if s == "Latin": found_latin = True
        except: continue
    return "Latin" if found_latin else None


def probe(gray, packs, available):
    """OCR with each pack and return {pack: (score, text)}."""
    results = {}
    for p in packs:
        if p not in available: continue
        t6   = ocr_string(gray, p, 6)
        t3   = ocr_string(gray, p, 3)
        text = t6 if quality_score(t6) >= quality_score(t3) else t3
        results[p] = (quality_score(text), text)
    return results


def probe_korean(gray, available) -> tuple:
    """
    FIX C — dedicated Korean probe that scores on Hangul char count.
    Returns (score, text) or (0, "") if kor pack unavailable / no Hangul found.
    """
    if KOREAN_PACK not in available:
        logger.debug("kor pack not available")
        return 0, ""
    t = ocr_string(gray, KOREAN_PACK, 6) or ocr_string(gray, KOREAN_PACK, 3)
    hc = hangul_char_count(t)
    if hc < 3:
        logger.debug("kor probe: only %d Hangul chars, skipping", hc)
        return 0, ""
    score = quality_score(t) + hc * 5   # Hangul chars are strong signal
    logger.debug("kor probe: hangul=%d score=%.1f", hc, score)
    return score, t


def _ocr_japanese(gray, available) -> tuple:
    """
    Try Japanese OCR with both horizontal (PSM 6) and vertical (PSM 5) modes
    and return the text with the higher kana+CJK score.
    Vertical Japanese text (tategumi) fails entirely with PSM 6/3.
    """
    if "jpn" not in available:
        return "", 0
    # Horizontal modes
    t_h = ocr_string(gray, "jpn", 6) or ocr_string(gray, "jpn", 3)
    # Vertical mode (PSM 5 = assume vertical uniform block of text)
    t_v = ocr_string(gray, "jpn", 5)
    def _jpn_score(t):
        kc = kana_char_count(t)
        cc = cjk_char_count(t)
        return quality_score(t) + kc * 6 + cc * 3, kc, cc
    sh, kh, ch = _jpn_score(t_h)
    sv, kv, cv = _jpn_score(t_v)
    if sv > sh:
        logger.debug("jpn: vertical PSM5 wins (score=%.1f kana=%d cjk=%d)", sv, kv, cv)
        return t_v, sv
    logger.debug("jpn: horizontal wins (score=%.1f kana=%d cjk=%d)", sh, kh, ch)
    return t_h, sh


def probe_cjk(gray, available) -> dict:
    """
    FIX A — CJK probe (Japanese + Chinese only, Korean handled separately).
    Scores on kana count for Japanese, generic CJK count for Chinese.
    Requires at least 3 script-specific chars to register as a candidate,
    preventing low-confidence packs from polluting the fallback path.
    """
    results = {}
    for p in [x for x in CJK_PACKS if x in available]:
        if p == "jpn":
            t, score = _ocr_japanese(gray, available)
            # Japanese: prefer kana over generic CJK to avoid false positives
            kc = kana_char_count(t)
            cc = cjk_char_count(t)
            if kc < 2 and cc < 5:
                logger.debug("jpn probe: kana=%d cjk=%d — below threshold, skip", kc, cc)
                continue
            # score already computed by _ocr_japanese; fall through to results[p]
        else:
            # Chinese (chi_sim / chi_tra) — use ideograph-only count so
            # Korean text OCR'd by chi_sim doesn't inflate the score
            t  = ocr_string(gray, p, 6) or ocr_string(gray, p, 3)
            cc = ideograph_char_count(t)
            hc = hangul_char_count(t)
            # If Hangul chars are significant relative to ideographs, this is
            # Korean being misread by a Chinese pack — reject outright.
            # Use ratio: reject if hangul >= 30% of ideographs or more Hangul than ideographs.
            if hc >= max(1, cc * 0.3) or cc < 3:
                logger.debug("chi probe %s: ideograph=%d hangul=%d — skip", p, cc, hc)
                continue
            score = quality_score(t) + cc * 4
        results[p] = (score, t)
        logger.debug("cjk probe pack=%s kana=%d cjk=%d score=%.1f",
                     p, kana_char_count(t), cjk_char_count(t), score)
    return results


def probe_sea_scripted(gray, script, available) -> dict:
    """
    FIX B — SEA probe that validates output using script-specific Unicode ranges.
    If the designated pack produces fewer than MIN_CHARS of the correct script,
    it is not trusted — preventing Latin/CJK packs from winning on SEA images.
    Lao gets a score multiplier because the lao pack typically produces fewer
    chars than tha even on genuine Lao images (sparser tessdata coverage).
    """
    # Lao tessdata is sparse — it may produce very few Lao codepoints even on
    # genuine Lao images. Use per-script minimums to avoid false negatives.
    MIN_CHARS_MAP = {"Thai":3, "Lao":1, "Khmer":2, "Myanmar":2}
    pack_map = {"Thai":"tha","Lao":"lao","Khmer":"khm","Myanmar":"mya"}
    # Lao char multiplier compensates for weaker tessdata coverage
    score_mult = {"Thai":5, "Lao":12, "Khmer":6, "Myanmar":6}
    pack = pack_map.get(script)
    if not pack or pack not in available:
        logger.debug("SEA pack for %s not available", script)
        return {}
    t  = ocr_string(gray, pack, 6) or ocr_string(gray, pack, 3)
    sc = sea_char_count(t, script)
    min_chars = MIN_CHARS_MAP.get(script, 3)
    if sc < min_chars:
        logger.debug("SEA %s probe: only %d script chars (need %d) — not trusted", script, sc, min_chars)
        return {}
    score = quality_score(t) + sc * score_mult.get(script, 5)
    logger.debug("SEA %s probe: chars=%d score=%.1f", script, sc, score)
    return {pack: (score, t)}


def _disambiguate_lao_thai(gray, sea_results: dict, available: set) -> dict:
    """
    When Thai wins a SEA probe but the image might be Lao, cross-check by
    counting Lao Unicode chars (0x0E80-0x0EFF) in the thai OCR output.
    If the Lao char count is significant, re-run the Lao probe and compare.
    Thai and Lao share some visual similarities so Tesseract can read Lao
    text with the Thai pack but produces many Lao codepoints in the output.
    """
    tha_result = sea_results.get("tha")
    if not tha_result:
        return sea_results
    tha_text = tha_result[1]
    lao_in_thai = sea_char_count(tha_text, "Lao")
    thai_in_thai = sea_char_count(tha_text, "Thai")
    logger.debug("lao/thai disambiguate: lao_chars_in_thai_output=%d thai_chars=%d",
                 lao_in_thai, thai_in_thai)
    # If the Thai pack output contains significant Lao characters, the image is Lao
    if lao_in_thai >= 3 and lao_in_thai >= thai_in_thai * 0.4:
        lao_result = probe_sea_scripted(gray, "Lao", available)
        if lao_result:
            logger.info("Lao/Thai disambiguation: switching to Lao")
            out = dict(sea_results)
            out.update(lao_result)
            return out
    return sea_results


# Priority order of fallback packs when the ideal pack is missing.
# Each entry is (pack, validator_fn, min_chars) — validator_fn counts
# script-specific characters in the OCR output to confirm the pack is
# reading the right script.
def _fallback_pack_priority(available: set) -> list:
    """Return ordered list of (pack, validator, min_chars) for available packs."""
    entries = [
        ("kor",      hangul_char_count,   3),
        ("jpn",      kana_char_count,     2),
        ("chi_sim",  ideograph_char_count, 3),
        ("chi_tra",  ideograph_char_count, 3),
        ("tha",      lambda t: sea_char_count(t, "Thai"),    3),
        ("lao",      lambda t: sea_char_count(t, "Lao"),     3),
        ("khm",      lambda t: sea_char_count(t, "Khmer"),   3),
        ("mya",      lambda t: sea_char_count(t, "Myanmar"), 3),
        ("ara",      lambda t: sum(1 for c in t if 0x0600<=ord(c)<=0x06FF), 8),
        ("hin",      lambda t: sum(1 for c in t if 0x0900<=ord(c)<=0x097F), 6),
        ("urd",      lambda t: sum(1 for c in t if 0x0600<=ord(c)<=0x06FF), 8),
    ]
    return [(p, v, m) for p, v, m in entries if p in available]


def _langdetect_fallback(gray, seed_text: str, available: set,
                         skip_scripts: tuple = ()) -> dict:
    """
    When the ideal script pack is not installed, try all available packs in
    priority order and validate output with Unicode char counting.
    If a pack produces enough script-specific characters, trust it.
    Falls back to eng as last resort.

    skip_scripts: packs to exclude (e.g. ("ara","urd") when OSD found CJK/SEA).
    """
    logger.warning("_langdetect_fallback triggered — ideal pack likely missing")

    # Try each available pack and validate with script-specific char counts
    for pack, validator, min_chars in _fallback_pack_priority(available):
        if pack in skip_scripts:
            logger.debug("  fallback: skipping %s (excluded by caller)", pack)
            continue
        t = ocr_string(gray, pack, 6) or ocr_string(gray, pack, 3)
        sc = validator(t)
        logger.debug("  fallback pack=%s script_chars=%d min=%d", pack, sc, min_chars)
        if sc >= min_chars:
            score = quality_score(t) + sc * 5
            logger.info("_langdetect_fallback winner: pack=%s chars=%d score=%.1f",
                        pack, sc, score)
            return {pack: (score, t)}

    # Nothing validated — use langdetect on seed_text as a hint
    try:
        ld = detect(seed_text) if seed_text.strip() else "en"
    except LangDetectException:
        ld = "en"
    lp = LANG_TO_PACK.get(ld, "eng")
    logger.warning("_langdetect_fallback last resort: detected=%s pack=%s", ld, lp)
    if lp in available:
        t = ocr_string(gray, lp, 6) or ocr_string(gray, lp, 3)
        if t.strip():
            return {lp: (quality_score(t), t)}

    # Absolute last resort — eng
    t = ocr_string(gray, "eng", 6)
    return {"eng": (quality_score(t), t)} if t.strip() else {}


def detect_lang(gray, available):
    """
    Main language detection pipeline.

    Strategy (in priority order):
    1. OSD identifies a non-Latin script → use dedicated pack for that script.
       - SEA scripts (Thai, Lao, Khmer, Myanmar): validated with Unicode range check.
       - Hangul → dedicated Korean probe.
       - Japanese/CJK → CJK probe with kana/ideograph thresholds.
       - Other non-Latin scripts → standard probe.
    2. OSD returns Latin or fails → run Korean probe first, then CJK probe,
       then SEA probe, then Latin probe. Pick the highest scorer.
    3. langdetect cross-check on the winning text (conservative 85% threshold).
    """
    script = try_osd(gray)
    logger.debug("OSD script=%s  available=%s", script, sorted(available))

    results = {}

    if script and script not in ("Latin", None):
        # ── Non-Latin script identified by OSD ────────────────────
        if script in ("Lao","Thai","Khmer","Myanmar"):
            # FIX B: use validated SEA probe
            results = probe_sea_scripted(gray, script, available)
            if not results:
                logger.warning("SEA pack for %s missing or produced no output — "
                               "falling back to all SEA packs", script)
                for s in ("Thai","Lao","Khmer","Myanmar"):
                    r = probe_sea_scripted(gray, s, available)
                    results.update(r)
            # Always disambiguate Lao vs Thai — OSD is weak on Lao
            results = _disambiguate_lao_thai(gray, results, available)
            # If still nothing, use fallback but never allow Arabic/Urdu —
            # OSD confirmed this is a SEA script.
            if not results:
                seed = ocr_string(gray, "eng", 6) or ocr_string(gray, "eng", 3)
                results = _langdetect_fallback(gray, seed, available,
                                               skip_scripts=("ara","urd"))

        elif script == "Hangul":
            # FIX C: dedicated Korean path
            kor_score, kor_text = probe_korean(gray, available)
            if kor_score > 0:
                results = {KOREAN_PACK: (kor_score, kor_text)}

        elif script in ("Japanese","HanS","HanT"):
            # Korean images frequently OSD as HanS/HanT because Hangul syllables
            # share CJK codepoints. Always probe Korean first and let scores decide.
            kor_score, kor_text = probe_korean(gray, available)
            if kor_score > 0:
                results[KOREAN_PACK] = (kor_score, kor_text)
            cjk_results = probe_cjk(gray, available)
            results.update(cjk_results)
            if not results:
                # No CJK/Korean packs installed — use langdetect fallback.
                # Skip Arabic/Urdu — OSD confirmed this is a CJK script.
                fallback_text = ocr_string(gray, "eng", 6) or ocr_string(gray, "eng", 3)
                results = _langdetect_fallback(gray, fallback_text, available,
                                               skip_scripts=("ara","urd"))

        else:
            packs = [p for p in SCRIPT_TO_PACKS.get(script, ["eng"]) if p in available]
            if not packs: packs = ["eng"]
            results = probe(gray, packs[:4], available)

        if not results:
            logger.warning("Script=%s yielded no results — trying universal fallback", script)
            seed = ocr_string(gray, "eng", 6) or ocr_string(gray, "eng", 3)
            results = _langdetect_fallback(gray, seed, available)
        if not results:
            results = probe(gray, ["eng"], available)

    else:
        # ── OSD returned Latin or failed entirely ─────────────────
        # Run all specialised probes and let scores decide.

        # 1. Korean (Hangul is common even without OSD detection)
        kor_score, kor_text = probe_korean(gray, available)
        if kor_score > 0:
            results[KOREAN_PACK] = (kor_score, kor_text)

        # 2. CJK (Japanese / Chinese)
        cjk_results = probe_cjk(gray, available)
        results.update(cjk_results)
        logger.debug("CJK probe results: %s", {k:round(v[0],1) for k,v in cjk_results.items()})

        # 3. SEA scripts (Thai is easy to confuse with other scripts)
        sea_results = {}
        for s in ("Thai","Lao","Khmer","Myanmar"):
            sea_results.update(probe_sea_scripted(gray, s, available))
        # Disambiguate Lao vs Thai when the Thai pack wins (they look similar)
        sea_results = _disambiguate_lao_thai(gray, sea_results, available)
        results.update(sea_results)
        sea_best = max((v[0] for v in sea_results.values()), default=0)
        logger.debug("SEA best score=%.1f", sea_best)

        # 4. Compute specialised_best BEFORE deciding whether to probe Arabic/Urdu.
        #    Arabic/Urdu packs fire on CJK/SEA noise so we skip them entirely when
        #    any specialised probe (Korean, CJK, SEA) already produced a result.
        specialised_best = max((v[0] for v in results.values()), default=0)

        # Raw CJK/kana presence check using eng output — catches images where
        # stylised kanji confuse Tesseract into producing 0 kana (below probe_cjk
        # threshold) but CJK chars are clearly visible.  If ANY CJK/kana chars
        # appear in the eng pass, treat this as a CJK image and skip Arabic/Urdu.
        _eng_scan = ocr_string(gray, "eng", 6) or ocr_string(gray, "eng", 3)
        _raw_cjk  = sum(1 for c in _eng_scan if (
            0x3040 <= ord(c) <= 0x30FF or   # Hiragana + Katakana
            0x4E00 <= ord(c) <= 0x9FFF or   # CJK Unified Ideographs
            0xAC00 <= ord(c) <= 0xD7AF      # Hangul
        ))
        _raw_lao  = sum(1 for c in _eng_scan if 0x0E80 <= ord(c) <= 0x0EFF)
        _raw_thai = sum(1 for c in _eng_scan if 0x0E00 <= ord(c) <= 0x0E7F)
        _has_cjk_presence  = _raw_cjk >= 2
        _has_sea_presence  = (_raw_lao + _raw_thai) >= 2
        logger.debug("raw scan: cjk=%d lao=%d thai=%d", _raw_cjk, _raw_lao, _raw_thai)

        # If CJK/SEA chars found in raw scan but no specialised probe won,
        # lower thresholds and retry Japanese probe before opening Arabic/Urdu gate.
        if _has_cjk_presence and specialised_best == 0 and "jpn" in available:
            t_retry, s_retry = _ocr_japanese(gray, available)
            kc_r = kana_char_count(t_retry)
            cc_r = cjk_char_count(t_retry)
            if kc_r >= 1 or cc_r >= 3:
                score_r = quality_score(t_retry) + kc_r * 6 + cc_r * 3
                results["jpn"] = (score_r, t_retry)
                specialised_best = score_r
                logger.info("jpn retry succeeded: kana=%d cjk=%d score=%.1f",
                            kc_r, cc_r, score_r)

        _ara_urd_results = {}
        # Skip Arabic/Urdu entirely if CJK or SEA chars detected in raw scan
        _skip_ara_urd = (specialised_best > 0 or _has_cjk_presence or _has_sea_presence)
        if not _skip_ara_urd:
            # No specialised script found — safe to probe Arabic/Urdu/Hindi,
            # but apply a strict quality+char-count gate to avoid noise wins.
            _ara_urd_results = probe(gray, ["ara","urd","hin"], available)
            for _p, (_sc, _tx) in list(_ara_urd_results.items()):
                _arabic_chars = sum(1 for c in _tx if 0x0600 <= ord(c) <= 0x06FF)
                _devan_chars  = sum(1 for c in _tx if 0x0900 <= ord(c) <= 0x097F)
                _script_chars = _arabic_chars if _p in ("ara","urd") else _devan_chars
                if _sc < 15 or _script_chars < 15:
                    logger.debug("Dropping %s: quality=%.1f script_chars=%d (insufficient)",
                                 _p, _sc, _script_chars)
                    del _ara_urd_results[_p]
        lat_results = probe(gray, ["eng","fra","deu","spa","rus","vie"], available)
        lat_results.update(_ara_urd_results)
        lat_best = max((v[0] for v in lat_results.values()), default=0)

        # Always add Latin results — scores naturally determine the winner.
        # Specialised probes already carry heavy per-character bonuses
        # (e.g. Hangul ×5, kana ×6) so they win on their own content without
        # needing to gate Latin out entirely.
        results.update(lat_results)

        logger.debug("Latin best=%.1f  specialised best=%.1f", lat_best, specialised_best)

        # If no specialised probe fired AND Latin probes are weak, the image
        # may use a script whose pack is not installed — run universal fallback
        if specialised_best == 0 and lat_best < 5:
            logger.warning("All probes weak — running universal fallback")
            seed = ocr_string(gray, "eng", 6) or ocr_string(gray, "eng", 3)
            fallback = _langdetect_fallback(gray, seed, available)
            results.update(fallback)

    if not results:
        return "eng", "en", ocr_string(gray, "eng", 3)

    bp = max(results, key=lambda p: results[p][0])
    bs, bt = results[bp]
    lc = PACK_TO_LANG.get(bp, "en")
    logger.debug("winner=%s lang=%s score=%.1f", bp, lc, bs)

    # ── langdetect cross-check (conservative: 85% threshold, non-empty only) ──
    if bt.strip():
        try:
            ld = detect(bt)
            lp = LANG_TO_PACK.get(ld, "eng")
            alt_score, alt_text = results.get(lp, (0, ""))
            if alt_score >= bs * 0.85 and alt_text.strip():
                lc = ld
                bt = alt_text
                logger.debug("langdetect override → lang=%s score=%.1f", lc, alt_score)
            else:
                logger.debug("langdetect=%s rejected (alt=%.1f < %.1f)", ld, alt_score, bs*0.85)
        except LangDetectException:
            pass

    return bp, lc, bt


# ──────────────── line extraction ─────────────────────────────────
def get_lines(gray_up, pack, scale_x, scale_y, W, H):
    # For Japanese, try vertical PSM 5 first — tategumi text fails with horizontal modes
    psm_order = (5, 6, 3, 4, 11) if pack == "jpn" else (6, 3, 4, 11)
    for psm in psm_order:
        try:
            data = pytesseract.image_to_data(gray_up, lang=pack,
                output_type=pytesseract.Output.DICT,
                config=f"--psm {psm} --oem 1")
        except: continue

        n = len(data["text"])
        line_map = {}
        for i in range(n):
            try: conf = int(data["conf"][i])
            except: conf = -1
            word = str(data["text"][i]).strip()
            if conf < 25 or not word: continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            line_map.setdefault(key, []).append(i)

        if not line_map: continue

        raw = []
        for idxs in line_map.values():
            x1u = min(data["left"][i]                  for i in idxs)
            y1u = min(data["top"][i]                   for i in idxs)
            x2u = max(data["left"][i]+data["width"][i] for i in idxs)
            y2u = max(data["top"][i]+data["height"][i] for i in idxs)
            src = " ".join(str(data["text"][i]) for i in idxs).strip()
            x1 = max(0, int(x1u/scale_x)); y1 = max(0, int(y1u/scale_y))
            x2 = min(W, int(x2u/scale_x)); y2 = min(H, int(y2u/scale_y))
            if x2-x1 < 6 or y2-y1 < 4 or not src: continue
            raw.append(dict(x1=x1,y1=y1,x2=x2,y2=y2,src=src,
                            bk=data["block_num"][idxs[0]],
                            par=data["par_num"][idxs[0]],
                            ln=data["line_num"][idxs[0]]))

        if not raw: continue
        raw.sort(key=lambda l: (l["bk"], l["par"], l["y1"]))

        merged = []; cur = None
        for ln in raw:
            if cur is None:
                cur = dict(**ln); cur["lines"] = [ln["src"]]; continue
            same_block    = (ln["bk"]==cur["bk"])
            same_section  = same_block and (ln["par"]==cur["par"])
            gap   = ln["y1"] - cur["y2"]
            avg_h = (cur["y2"]-cur["y1"] + ln["y2"]-ln["y1"]) / 2
            # Merge lines that belong to the same paragraph (generous gap)
            # OR that are in the same block and very close (catches tight layouts).
            should_merge = (
                (same_section and -2 <= gap <= avg_h * 1.5) or
                (same_block and 0 <= gap <= avg_h * 0.6)
            )
            if should_merge:
                cur["x1"] = min(cur["x1"], ln["x1"])
                cur["x2"] = max(cur["x2"], ln["x2"])
                cur["y2"] = max(cur["y2"], ln["y2"])
                cur["lines"].append(ln["src"])
            else:
                cur["src"] = " ".join(cur["lines"])
                merged.append(cur)
                cur = dict(**ln); cur["lines"] = [ln["src"]]
        if cur:
            cur["src"] = " ".join(cur["lines"]); merged.append(cur)

        logger.debug("bbox psm %d → %d line-blocks", psm, len(merged))
        return merged

    logger.debug("bbox: all psm modes returned 0 lines")
    return []


# ──────────────────────────── overlay ─────────────────────────────
def fit_text(draw, text, box_w, box_h, min_fs=8, max_fs=None):
    if max_fs is None: max_fs = max(min_fs, int(box_h * 0.82))
    max_fs = max(min_fs, max_fs)
    best = None; lo, hi = min_fs, max_fs
    while lo <= hi:
        fs = (lo+hi)//2; font = load_font(fs)
        lh = draw.textbbox((0,0),"Ag",font=font)[3]+2
        words = text.split(); lines = []; line = []
        for w in words:
            test = " ".join(line+[w])
            if draw.textbbox((0,0),test,font=font)[2] > box_w-6:
                if line: lines.append(" ".join(line)); line=[w]
                else:    lines.append(w)
            else: line.append(w)
        if line: lines.append(" ".join(line))
        total_h = len(lines)*lh
        if total_h <= box_h: best=(font,lines,lh,total_h); lo=fs+1
        else: hi=fs-1
    if best: return best
    font=load_font(min_fs); lh=draw.textbbox((0,0),"Ag",font=font)[3]+2
    return font,[text],lh,lh

def render_overlay(orig_path, gray_up, scale_x, scale_y, pack, lang_code):
    orig = cv2.imread(orig_path)
    pil  = Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)).convert("RGBA")
    W, H = pil.size
    is_en = (lang_code=="en" and pack=="eng")

    blocks = get_lines(gray_up, pack, scale_x, scale_y, W, H)
    if not blocks:
        return _banner(pil, W, H, ocr_string(gray_up, pack, 3), lang_code)

    if is_en:
        for blk in blocks: blk["tr"] = blk["src"]
    else:
        translations = batch_translate([blk["src"] for blk in blocks])
        for blk, tr in zip(blocks, translations): blk["tr"] = tr

    ov   = Image.new("RGBA", pil.size, (0,0,0,0))
    draw = ImageDraw.Draw(ov)

    # Minimum readable font size: 1/60th of image height, floor 13px
    min_readable_fs = max(13, H // 60)
    # Global cap: ~1 char per 36px width, range 16-56px
    global_max_fs   = max(16, min(56, W // 36))

    # Sort top-to-bottom to track occupied vertical space and prevent overlaps
    blocks.sort(key=lambda b: b["y1"])
    last_drawn_y2 = -1

    for blk in blocks:
        x1, y1, x2, y2 = blk["x1"], blk["y1"], blk["x2"], blk["y2"]
        bw_px = x2 - x1
        bh_px = y2 - y1
        tr = (blk["tr"] or "").strip()
        if not tr: continue
        # Skip tiny numeric-only slivers
        if bw_px < 45 and re.fullmatch(r"[\d\s,.\-٠-٩]+", tr): continue
        # Skip degenerate boxes
        if bw_px < 20 or bh_px < 8: continue

        # Push box down if it overlaps the previous drawn box
        y1 = max(y1, last_drawn_y2 + 2)
        if y1 >= H: continue

        # Font sizing: fill block height but enforce minimum readability
        max_fs = min(global_max_fs, max(min_readable_fs, int(bh_px * 0.85)))
        min_fs = min_readable_fs
        font, lines, lh, total_h = fit_text(draw, tr, bw_px, bh_px,
                                             min_fs=min_fs, max_fs=max_fs)

        # Box height: enough for text plus 8px padding, clamped to image
        box_h     = max(bh_px, total_h + 8)
        actual_y2 = min(H, y1 + box_h)

        # Semi-transparent dark background
        draw.rectangle([(x1, y1), (x2, actual_y2)], fill=(10, 10, 20, 220))

        # Vertically centre text in the box
        ty = y1 + max(4, (actual_y2 - y1 - total_h) // 2)
        for ln in lines:
            if ty + lh > actual_y2: break  # do not draw outside the box
            # Thin black outline for contrast
            for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
                draw.text((x1+5+dx, ty+dy), ln, font=font, fill=(0,0,0,200))
            # White text
            draw.text((x1+5, ty), ln, font=font, fill=(255,255,255,255))
            ty += lh

        last_drawn_y2 = actual_y2

    out  = Image.alpha_composite(pil, ov).convert("RGB")
    path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.jpg")
    out.save(path, quality=93)
    return path


def _banner(pil,W,H,raw,lang_code):
    tr=raw if lang_code=="en" else translate_full(raw)
    ov=Image.new("RGBA",pil.size,(0,0,0,0)); draw=ImageDraw.Draw(ov)
    fs=max(13,min(20,W//38)); lh=fs+7
    font=load_font(fs); sm=load_font(max(10,fs-3))
    words=tr.split(); lines=[]; line=[]
    for w in words:
        t=" ".join(line+[w])
        if draw.textbbox((0,0),t,font=font)[2]>W-32:
            if line: lines.append(" ".join(line)); line=[w]
        else: line.append(w)
    if line: lines.append(" ".join(line))
    ml=min(len(lines),max(3,int(H*0.40/lh)))
    bh=30+ml*lh+14; bt=H-bh
    draw.rectangle([(0,bt),(W,H)],fill=(8,8,24,225))
    draw.line([(0,bt),(W,bt)],fill=(100,200,255,160),width=2)
    draw.text((16,bt+8),"TRANSLATED (EN)",font=sm,fill=(100,200,255,220))
    y=bt+28
    for ln in lines[:ml]:
        draw.text((16,y),ln,font=font,fill=(255,255,255,255)); y+=lh
    path=os.path.join(OUTPUT_DIR,f"{uuid.uuid4()}.jpg")
    Image.alpha_composite(pil,ov).convert("RGB").save(path,quality=93)
    return path


# ─────────────────────────────── main ─────────────────────────────
def process_image(image_path: str) -> dict:
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

    h0, w0 = img.shape[:2]; TARGET = 2000
    if max(h0, w0) < TARGET:
        sc     = TARGET / max(h0, w0)
        img_up = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    else:
        sc = 1.0; img_up = img.copy()

    sx   = img_up.shape[1] / w0
    sy   = img_up.shape[0] / h0
    gray = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.convertScaleAbs(gray, alpha=1.35, beta=10)

    available           = get_available_langs()
    pack, lc, extracted = detect_lang(gray, available)
    result["detected_language"] = lc

    if not extracted.strip():
        result["error"] = "No readable text found."
        return result

    result["extracted_text"]  = extracted
    result["translated_text"] = (
        extracted if (lc == "en" and pack == "eng")
        else translate_full(extracted)
    )

    try:
        result["translated_image"] = render_overlay(
            image_path, gray, sx, sy, pack, lc
        )
    except Exception as e:
        logger.exception("Render failed")
        result["error"] = f"Render failed: {e}"

    return result