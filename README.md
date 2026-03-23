# Image Text OCR Translator

An OCR pipeline that extracts text from images, automatically detects the language, translates it to English, and renders the translated text as an overlay back onto the original image.

## Features

- Supports JPEG, PNG, and WebP image uploads
- OCR using Tesseract with multi-language support
- Automatic language detection (30+ languages)
- Translation to English via Google Translator
- Translated text overlay rendered onto the original image
- Dark / light theme frontend
- In-memory rate limiting (20 requests / 60s per IP)
- File type validation via magic-byte checking
- Automatic cleanup of uploaded and output files after 1 hour

## Supported Languages

Japanese, Korean, Chinese (Simplified & Traditional), Hindi, Arabic, Urdu, Malayalam, Thai, Lao, Khmer, Myanmar, and all major Latin-script languages.

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **OCR:** Tesseract OCR
- **Image processing:** OpenCV, Pillow
- **Translation:** Deep Translator (Google Translate)
- **Language detection:** langdetect
- **Frontend:** HTML, CSS, JavaScript (vanilla)
- **Containerisation:** Docker

## Project Structure

```
text-ocr-translator/
├── backend/
│   ├── main.py          # FastAPI app, upload endpoint, rate limiting
│   └── ocr_engine.py    # OCR pipeline, language detection, overlay rendering
├── static/
│   ├── script.js        # Frontend logic
│   └── style.css
├── templates/
│   └── index.html       # Single-page UI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Run with Docker (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
git clone https://github.com/arunkmr13/text-ocr-translator.git
cd text-ocr-translator
docker compose up --build
```

App will be available at `http://localhost:8000`.

To run in the background:

```bash
docker compose up -d
```

To stop:

```bash
docker compose down
```

## Run Locally

Requires Python 3.10+ and Tesseract installed on your system.

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-all
```

Then:

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

App will be available at `http://localhost:8000`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TESSERACT_CMD` | `/usr/bin/tesseract` | Path to Tesseract binary |
| `TESSDATA_PREFIX` | `/usr/share/tesseract-ocr/5/tessdata` | Path to tessdata directory |

These are set automatically inside Docker. Override them if your local Tesseract is in a different location.
