# Image Text OCR Translator

An OCR pipeline that extracts text from images, detects language automatically, translates it to English, and renders the translated overlay back onto the image.

## Features

- OCR using Tesseract
- Smart language detection
- Poster/infographic text detection using CRAFT
- Translation using Google Translator
- Image overlay rendering

## Pipeline

Image → Text Detection (CRAFT) → OCR (Tesseract) → Language Detection → Translation → Overlay Rendering

## Tech Stack

- Python
- FastAPI
- OpenCV
- Tesseract OCR
- CRAFT Text Detector
- Deep Translator
- HTML / CSS / JS frontend

## Run locally

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
