# main.py — FastAPI entry point with SSE real-time render progress

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import defaultdict

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

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
# Rate limiting
# ---------------------------------------------------------------------------

RATE_LIMIT_MAX    = 20
RATE_LIMIT_WINDOW = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
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
                    content={"detail": f"Too many requests — limit is {self.max_requests} per {self.window_seconds}s."},
                )
            self._hits[ip].append(now)
        return await call_next(request)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="OCR Translator", version="2.0.0")
app.add_middleware(RateLimitMiddleware, max_requests=RATE_LIMIT_MAX, window_seconds=RATE_LIMIT_WINDOW)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_TTL_SECONDS      = 3600
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE         = 5 * 1024 * 1024

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static",  StaticFiles(directory=os.path.join(BASE_DIR, "static")),  name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ---------------------------------------------------------------------------
# File cleanup
# ---------------------------------------------------------------------------

def _cleanup_old_files(*dirs: str, ttl: int = FILE_TTL_SECONDS) -> None:
    now = time.time()
    for d in dirs:
        try:
            for fname in os.listdir(d):
                fpath = os.path.join(d, fname)
                if os.path.isfile(fpath):
                    try:
                        if now - os.path.getmtime(fpath) > ttl:
                            os.remove(fpath)
                    except OSError:
                        pass
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
    _cleanup_old_files(UPLOAD_DIR, OUTPUT_DIR)

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{file.content_type}'.")

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds the 5 MB size limit.")

    if not validate_image_bytes(contents):
        raise HTTPException(status_code=415, detail="File content does not match a valid image format.")

    _EXT_MAP = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    ext        = _EXT_MAP.get(file.content_type, ".jpg")
    file_id    = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(image_path, "wb") as f:
        f.write(contents)

    logger.info("Processing upload file_id=%s ext=%s size=%d bytes", file_id, ext, len(contents))

    # Run process_image without progress callback for regular JSON response
    result = process_image(image_path)

    original_rel = f"uploads/{file_id}{ext}"

    if result.get("error") and not result.get("extracted_text"):
        return JSONResponse(
            status_code=422,
            content={"error": result["error"], "original_image": original_rel},
        )

    translated_img = result.get("translated_image", "")
    if translated_img.startswith(BASE_DIR):
        translated_img = translated_img[len(BASE_DIR):].lstrip("/\\")

    return JSONResponse({
        "original_image":    original_rel,
        "translated_image":  translated_img,
        "extracted_text":    result.get("extracted_text", ""),
        "translated_text":   result.get("translated_text", ""),
        "detected_language": result.get("detected_language", "unknown"),
        "structured_data":   result.get("structured_data", {}),
        "region_count":      result.get("region_count", 0),
        "warning":           result.get("error"),
    })


@app.post("/upload-stream")
async def upload_image_stream(request: Request, file: UploadFile = File(...)):
    """
    SSE endpoint — streams real-time render progress then final result.

    Event types:
      progress  → {"done": 3, "total": 74}
      result    → full JSON result (same as /upload)
      error     → {"error": "message"}
    """
    _cleanup_old_files(UPLOAD_DIR, OUTPUT_DIR)

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        async def err_gen():
            yield f"event: error\ndata: {json.dumps({'error': 'Unsupported file type'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        async def err_gen():
            yield f"event: error\ndata: {json.dumps({'error': 'File too large'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    if not validate_image_bytes(contents):
        async def err_gen():
            yield f"event: error\ndata: {json.dumps({'error': 'Invalid image'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    _EXT_MAP = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    ext        = _EXT_MAP.get(file.content_type, ".jpg")
    file_id    = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(image_path, "wb") as f:
        f.write(contents)

    logger.info("SSE upload file_id=%s ext=%s size=%d bytes", file_id, ext, len(contents))

    # Use a thread-safe queue to pass progress events from worker thread to async generator
    progress_queue: queue.Queue = queue.Queue()

    def progress_callback(done: int, total: int):
        progress_queue.put({"done": done, "total": total})

    def worker():
        try:
            result = process_image(image_path, progress_callback=progress_callback)
            progress_queue.put({"__result__": result})
        except Exception as e:
            progress_queue.put({"__error__": str(e)})

    # Start processing in background thread
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    original_rel = f"uploads/{file_id}{ext}"

    async def event_generator():
        # Keep-alive comment to establish SSE connection immediately
        yield ": keep-alive\n\n"

        while True:
            try:
                # Poll queue with short timeout so we don't block the event loop
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: progress_queue.get(timeout=0.1)
                )
            except queue.Empty:
                # Send heartbeat to keep connection alive
                yield ": ping\n\n"
                continue

            if "__result__" in msg:
                result = msg["__result__"]

                if result.get("error") and not result.get("extracted_text"):
                    yield f"event: error\ndata: {json.dumps({'error': result['error'], 'original_image': original_rel})}\n\n"
                    return

                translated_img = result.get("translated_image", "")
                if translated_img.startswith(BASE_DIR):
                    translated_img = translated_img[len(BASE_DIR):].lstrip("/\\")

                payload = {
                    "original_image":    original_rel,
                    "translated_image":  translated_img,
                    "extracted_text":    result.get("extracted_text", ""),
                    "translated_text":   result.get("translated_text", ""),
                    "detected_language": result.get("detected_language", "unknown"),
                    "structured_data":   result.get("structured_data", {}),
                    "region_count":      result.get("region_count", 0),
                    "warning":           result.get("error"),
                }
                yield f"event: result\ndata: {json.dumps(payload)}\n\n"
                return

            elif "__error__" in msg:
                yield f"event: error\ndata: {json.dumps({'error': msg['__error__'], 'original_image': original_rel})}\n\n"
                return

            else:
                # Real progress event
                yield f"event: progress\ndata: {json.dumps(msg)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )