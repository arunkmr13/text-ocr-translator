# Text OCR Translator

An OCR pipeline that extracts text from images, auto-detects the language, translates it to English, and renders the translation as an overlay on the original image.

Powered by Google Gemini Vision API (free tier — no credit card needed).

## Features

- Multilingual OCR — Lao, Thai, Arabic, Hindi, Chinese, Japanese, Korean, and 50+ more
- Auto language detection via Unicode block analysis + langdetect fallback
- Structured table extraction with markdown pipe table output
- Translated text overlay rendered in-place over original text regions
- Real-time pipeline progress tracker (4-step animated UI)
- Dark / light theme frontend
- In-memory rate limiting (20 requests / 60s per IP)
- Magic-byte file validation (JPEG · PNG · WebP)
- Auto cleanup of uploaded and output files after 1 hour

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| OCR + Translation | Google Gemini Vision API (`gemini-flash-latest`) |
| Image processing | OpenCV, Pillow |
| Language detection | Unicode block analysis + langdetect |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Containerisation | Docker |

## AI Model

This project uses Google Gemini Vision API for OCR and translation in a single multimodal pass — no separate OCR library needed.

Primary model: `gemini-flash-latest`

Automatic fallback chain (if primary is quota-exhausted or unavailable):
1. `gemini-flash-latest`
2. `gemini-2.0-flash`
3. `gemini-2.5-flash`
4. `gemini-2.0-flash-lite`

Why Gemini Vision over traditional OCR:
- One API call handles OCR + language detection + translation + bounding box estimation simultaneously
- Understands document structure — extracts tables as structured data, not just raw text
- Free tier: 1,500 requests/day, 15 requests/minute — no credit card required
- Strong support for non-Latin scripts: Lao, Thai, Arabic, Devanagari, CJK, and more

Two-pass pipeline:
- Pass 1 — full OCR, translation, and bounding boxes for non-table elements (title, date, stat boxes, column headers, footers)
- Pass 2 — dedicated table row extraction using a compact format, preventing response truncation on large tables

Get a free API key: https://aistudio.google.com/app/apikey

## Project Structure

```
text-ocr-translator/
├── backend/
│   ├── main.py                  # FastAPI app, routes, rate limiting
│   ├── ocr_engine.py            # Pipeline orchestrator
│   ├── assets/fonts/            # Noto fonts for overlay rendering
│   └── services/
│       ├── ocr_service.py       # Gemini Vision API + two-pass strategy
│       ├── overlay_service.py   # OpenCV/Pillow overlay renderer
│       ├── language_service.py  # Unicode block + langdetect detection
│       └── layout_service.py    # Layout analysis (optional)
├── static/
│   └── script.js                # Frontend logic
├── templates/
│   └── index.html               # Single-page UI
├── uploads/                     # Runtime — git-ignored
├── outputs/                     # Runtime — git-ignored
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quickstart

### 1. Get a free Gemini API key

→ [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

No credit card. Free tier: 1,500 requests/day, 15 requests/minute.

### 2. Clone and configure

```bash
git clone https://github.com/arunkmr13/text-ocr-translator.git
cd text-ocr-translator
cp .env.example .env
# Edit .env and add your key:
# GEMINI_API_KEY=your_key_here
```

### 3a. Run with Docker (recommended)

```bash
docker compose up --build
```

App at `http://localhost:8000`

```bash
docker compose up -d      # background
docker compose down       # stop
```

### 3b. Run locally

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

App at `http://localhost:8000`

## Environment Variables

| Variable         | Required | Description                |
| -----------------|----------|----------------------------|
| `GEMINI_API_KEY` | ✅ Yes   | Your Google Gemini API key |

## Limitations

- Max file size: 5 MB
- Supported formats: JPEG, PNG, WebP
- Translated overlay positioning depends on Gemini bounding box accuracy
- Free tier quota: 1,500 requests/day, 15 requests/minute