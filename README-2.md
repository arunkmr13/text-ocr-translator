# Text OCR Translator

> Photograph any document in a foreign language. Get the English translation overlaid directly on the image — in seconds.

Powered by Google Gemini Vision API · Free tier.
---

## What It Does

Upload an image containing text in any language. The app:

1. Extracts every word from the image using AI vision
2. Detects the source language automatically
3. Translates everything to English
4. Renders the English translation directly over the original image — preserving layout, tables, and structure

Example: A Lao government health report with 18 provinces, statistics, and tables → fully translated overlay image + clean extracted text, all in under 90 seconds.

---

## Why This Exists

Traditional OCR tools extract text but lose all layout context. Translation tools accept text but don't know where it came from. This project combines both into a single pipeline — the output is an image you can actually read, not just a block of raw translated text.

Use cases:
- Translating foreign government reports, medical documents, or infographics
- Extracting structured table data from images in any language
- Any workflow where someone receives image-based documents in a language they don't read

---

## Live Demo

```
http://localhost:8001
```

Upload any JPEG, PNG, or WebP image → click Translate Image → results appear in under 90 seconds.

The UI shows:
- Original and Translated Overlay images side by side
- Source text (extracted, in original language) and English translation*as structured markdown
- 4-step pipeline tracker showing real-time progress
- Render complete toast showing exact time taken

---

## How It Works — The Pipeline

```
Image Upload
     │
     ▼
┌─────────────────────────────────────┐
│  Step 1: OCR Scan                   │
│  Gemini Vision reads every text     │
│  element and its position           │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Step 2: Language Detection         │
│  Unicode block analysis identifies  │
│  script → langdetect for Latin      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Step 3: Translation                │
│  Pass 1 — headers, stat boxes,      │
│  titles, footers + bounding boxes   │
│  Pass 2 — all table rows (compact   │
│  format, prevents truncation)       │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Step 4: Render Overlay             │
│  Each text region gets a            │
│  semi-transparent fill box +        │
│  English text sized to match the    │
│  original visual weight             │
└─────────────────────────────────────┘
```

Why two Gemini passes?
Gemini has a response length limit. Asking for all bounding boxes in one call causes truncation on dense documents (tables with 18+ rows). Pass 1 handles metadata, Pass 2 handles table rows using a compact array format — preventing truncation while keeping a single API key.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | Python 3.11+, FastAPI, Uvicorn | Async, production-ready, auto docs at `/docs` |
| AI — OCR + Translation | Google Gemini Vision API (`gemini-flash-latest`) | Single call handles OCR + translation + bounding boxes + table extraction |
| Image processing | OpenCV, Pillow | Overlay rendering, colour sampling, font compositing |
| Language detection | Unicode block analysis + langdetect | Fast, no API needed — detects 50+ scripts from character ranges |
| Frontend | Vanilla JS, HTML, CSS | Zero dependencies, instant load, dark/light theme |
| Containerisation | Docker | One-command deployment |

---

## AI Model Details

Primary: `gemini-flash-latest`

Automatic fallback chain (handles quota exhaustion gracefully):
```
gemini-flash-latest → gemini-2.0-flash → gemini-2.5-flash → gemini-2.0-flash-lite
```

Each model is tried in order. If one returns a 429 (rate limit), the code reads the `retryDelay` from Google's response and waits exactly that long before trying the next model — no wasted retries.

Free tier limits: 1,500 requests/day · 15 requests/minute · $0 cost

---

## Supported Languages

Any language Gemini Vision can read. Tested and confirmed working:

| Script | Languages |
|---|---|
| Lao | Lao |
| Thai | Thai |
| Arabic | Arabic, Urdu, Persian |
| Devanagari | Hindi, Sanskrit, Nepali |
| CJK | Chinese (Simplified + Traditional), Japanese, Korean |
| Cyrillic | Russian, Ukrainian, Bulgarian |
| Latin | English, French, German, Spanish, Portuguese, Vietnamese, Indonesian, and more |
| Other | Bengali, Tamil, Telugu, Kannada, Malayalam, Khmer, Myanmar, Hebrew |

---

## Project Structure

```
text-ocr-translator/
├── backend/
│   ├── main.py                  # FastAPI app — routes, rate limiting, file cleanup
│   ├── ocr_engine.py            # Pipeline orchestrator — connects all stages
│   ├── assets/fonts/            # Noto font family for multilingual overlay rendering
│   └── services/
│       ├── ocr_service.py       # Two-pass Gemini Vision strategy
│       ├── overlay_service.py   # OpenCV + Pillow overlay renderer
│       └── language_service.py  # Unicode block analysis + langdetect
├── static/
│   └── script.js                # Frontend — Promise.all fetch, toast, progress bar
├── templates/
│   └── index.html               # Single-page UI with dark/light theme
├── uploads/                     # Runtime — auto-cleaned after 1 hour
├── outputs/                     # Runtime — auto-cleaned after 1 hour
├── .env.example                 # API key template
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup

### 1. Get a free Gemini API key

→ [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) — no credit card, instant access

### 2. Configure

```bash
git clone https://github.com/arunkmr13/text-ocr-translator.git
cd text-ocr-translator
cp .env.example .env
# Add your key: GEMINI_API_KEY=your_key_here
```

### 3a. Docker (recommended)

```bash
docker compose up --build
# App at http://localhost:8000
```

### 3b. Local

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
# App at http://localhost:8000
```

---

## Key Engineering Decisions

Why Gemini Vision instead of Tesseract + Google Translate?**
Tesseract requires language packs installed per language, struggles with non-Latin scripts, and has no table understanding. Gemini Vision reads any script out of the box, understands document structure, and returns both translation and spatial coordinates in a single API call.

Why vanilla JS instead of React?
The UI state is simple — one upload, one result. A framework would add build complexity with no benefit. Vanilla JS with `Promise.all` handles the concurrent fetch + animation correctly with zero dependencies.

Why two Gemini passes instead of one?
Gemini's response token limit causes truncation on dense documents. A single pass requesting all 90+ bounding boxes for an 18-province table gets cut off mid-response. Pass 1 (metadata) + Pass 2 (table rows in compact array format) guarantees complete coverage without hitting limits.

Why height-driven font sizing?
The English overlay text must match the visual weight of the original language text. Since Gemini returns bounding boxes derived from the original glyph heights, using `font_size = (bbox_height - padding) × 1.25` produces cap-heights that match the original text scale — regardless of language or document size.

---

## Security & Reliability

| Concern | How it's handled |
|---|---|
| API key exposure | Stored in `.env`, git-ignored, never hardcoded |
| File uploads | Magic-byte validation (not just MIME type), 5 MB limit |
| Rate limiting | 20 requests / 60s per IP, in-memory middleware |
| Disk usage | Auto-cleanup of uploads + outputs after 1 hour |
| Quota exhaustion | 4-model fallback chain with retryDelay-aware sleep |
| Large tables | Two-pass Gemini strategy prevents response truncation |

---

## Limitations & Known Constraints

- Bounding box accuracy — Gemini estimates normalised coordinates rather than computing pixel-precise positions. Overlay alignment is approximate (±5% of image dimensions).
- Processing time — Two Gemini API calls per request = ~60-90 seconds total. Bottleneck is network latency to Google's API, not local computation.
- Free tier quota — 1,500 requests/day. Each translation uses 2 calls, so effectively 750 translations/day.
- Overlay on complex backgrounds — Semi-transparent fill boxes work well on solid/gradient backgrounds. Highly textured backgrounds may reduce overlay legibility.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key |

---

```
