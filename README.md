# Text OCR Translator

An OCR pipeline that extracts text from images, auto-detects the language, translates it to English, and renders the translation as an overlay on the original image.

Powered by **Google Gemini Vision API** (free tier — no credit card needed).

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
| OCR + Translation | Google Gemini Vision API |
| Image processing | OpenCV, Pillow |
| Language detection | Unicode block analysis + langdetect |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Containerisation | Docker |

## Project Structure
