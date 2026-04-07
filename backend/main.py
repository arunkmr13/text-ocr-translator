# main.py  —  FastAPI entry point
# Minimal changes from original:
#   - structured_data added to success response
#   - partial success (render failed but text extracted) returns 200 not 422
#   - import path normalised for running from project root or backend/

from dotenv import load_dotenv
load_dotenv()

import logging
import os
import time
import uuid
from collections import defaultdict

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.ocr_engine import process_image, validate_image_bytes

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting middleware (unchanged — logic is correct)
# ---------------------------------------------------------------------------

RATE_LIMIT_MAX    = 20   # max requests
RATE_LIMIT_WINDOW = 60   # per N seconds


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window in-memory rate limiter. Keyed by client IP."""

    def __init__(self, app, max_requests: int, window_seconds: int):
        super().__init__(app)
        self.max_requests   = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/upload":
            ip  = request.client.host if request.client else "unknown"
            now = time.time()
            cutoff = now - self.window_seconds

            self._hits[ip] = [t for t in self._hits[ip] if t > cutoff]

            if len(self._hits[ip]) >= self.max_requests:
                logger.warning("Rate limit exceeded for IP=%s", ip)
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": (
                            f"Too many requests — limit is {self.max_requests} "
                            f"per {self.window_seconds}s. Try again shortly."
                        )
                    },
                )
            self._hits[ip].append(now)

        return await call_next(request)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="OCR Translator", version="2.0.0")
app.add_middleware(
    RateLimitMiddleware,
    max_requests=RATE_LIMIT_MAX,
    window_seconds=RATE_LIMIT_WINDOW,
)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_TTL_SECONDS      = 3600   # 1 hour
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE         = 5 * 1024 * 1024  # 5 MB

templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)
app.mount("/static",  StaticFiles(directory=os.path.join(BASE_DIR, "static")),  name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ---------------------------------------------------------------------------
# File cleanup (unchanged)
# ---------------------------------------------------------------------------

def _cleanup_old_files(*dirs: str, ttl: int = FILE_TTL_SECONDS) -> None:
    """Delete files older than `ttl` seconds from given directories."""
    now = time.time()
    for d in dirs:
        try:
            for fname in os.listdir(d):
                fpath = os.path.join(d, fname)
                if os.path.isfile(fpath):
                    try:
                        if now - os.path.getmtime(fpath) > ttl:
                            os.remove(fpath)
                            logger.debug("Cleaned up: %s", fpath)
                    except OSError as e:
                        logger.warning("Could not remove %s: %s", fpath, e)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/upload")
async def upload_image(request: Request, file: UploadFile = File(...)):

    # Cleanup stale files on each request
    _cleanup_old_files(UPLOAD_DIR, OUTPUT_DIR)

    # Server-side content-type check
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                "Please upload a JPEG, PNG, or WebP image."
            ),
        )

    contents = await file.read()

    # Size limit
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 5 MB size limit.",
        )

    # Magic-byte validation
    if not validate_image_bytes(contents):
        raise HTTPException(
            status_code=415,
            detail="File content does not match a valid image format.",
        )

    _EXT_MAP = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    ext        = _EXT_MAP.get(file.content_type, ".jpg")
    file_id    = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(image_path, "wb") as f:
        f.write(contents)

    logger.info(
        "Processing upload file_id=%s ext=%s size=%d bytes",
        file_id, ext, len(contents),
    )

    result = process_image(image_path)

    original_rel = f"uploads/{file_id}{ext}"

    # ── Hard failure: no text extracted at all ───────────────────────
    # Only return 422 when both text AND render failed.
    # If text was extracted but render failed, return 200 with a warning
    # so the frontend can still show the translated text.
    if result.get("error") and not result.get("extracted_text"):
        return JSONResponse(
            status_code=422,
            content={
                "error":          result["error"],
                "original_image": original_rel,
            },
        )

    # Normalise output image path to a relative URL
    translated_img = result.get("translated_image", "")
    if translated_img.startswith(BASE_DIR):
        translated_img = translated_img[len(BASE_DIR):].lstrip("/\\")

    # ── Success (full or partial) ────────────────────────────────────
    return JSONResponse({
        "original_image":    original_rel,
        "translated_image":  translated_img,
        "extracted_text":    result.get("extracted_text", ""),
        "translated_text":   result.get("translated_text", ""),
        "detected_language": result.get("detected_language", "unknown"),
        "structured_data":   result.get("structured_data", {}),   # ← NEW: Claude-extracted metrics
        "warning":           result.get("error"),                  # non-fatal render errors surface here
    })