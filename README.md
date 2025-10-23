## AI Voice Cloning for Multilingual Localized Customer Support

A production-ready, fully open‑source prototype for voice-based customer support. It transcribes user speech, understands intent across multiple languages, and responds in a cloned, natural-sounding voice. Runs locally or on Hugging Face Spaces without paid APIs.

### Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Customization](#customization)
- [Performance & Quality Tips](#performance--quality-tips)
- [Troubleshooting](#troubleshooting)
- [Deployment on Hugging Face Spaces](#deployment-on-hugging-face-spaces)
- [Security & Privacy](#security--privacy)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
- **Open-source end-to-end**: No paid APIs; all components are local and open.
- **Multilingual STT**: faster-whisper (CPU-friendly), with optional language override.
- **Multilingual understanding**: sentence-transformers with language-aware intent retrieval and confidence thresholding.
- **Voice cloning TTS**: Coqui TTS (XTTS v2 preferred), with automatic fallback to YourTTS and finally Tacotron2 if needed.
- **Languages out of the box**: English (en), Hindi (hi), Spanish (es), Tamil (ta), Arabic (ar). Easily extendable.
- **Modern UI**: Streamlit app with microphone recording or WAV upload and live audio playback.

## System Architecture
- **STT**: faster-whisper (CT2 int8) produces transcript + language.
- **NLP**: SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2) performs cosine similarity retrieval against language-specific intent examples with a minimum similarity threshold.
- **NLG**: Localized response templates per intent and language.
- **TTS**: Coqui TTS (XTTS v2) with `speaker_wav` cloning and `language` selection. If XTTS fails due to environment/version issues, the app transparently falls back to YourTTS or Tacotron2.

## Requirements
- **OS**: Windows, macOS, or Linux
- **Python**: 3.9–3.11
- **Hardware**: CPU supported; GPU (CUDA) recommended for best latency and quality

## Installation
1) Create and activate a virtual environment
```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
```bash
# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

3) Accept Coqui TTS Terms of Service (only if prompted)
The app sets this automatically via `COQUI_TOS_AGREED`, but if you still see a ToS error, set it manually and rerun:
```powershell
# Windows (PowerShell)
$env:COQUI_TOS_AGREED = "1"
```
```bash
# macOS/Linux
export COQUI_TOS_AGREED=1
```

## Quick Start
```bash
streamlit run app.py
```
Open the URL shown in the terminal (typically `http://localhost:8501`).

## Usage Guide
1) **Reference voice**
   - Upload a clean 5–10s mono WAV of the desired speaker (or use the included `harvard.wav`).
2) **STT model size** (sidebar)
   - `tiny`/`base`/`small`/`medium`. Larger models improve accuracy at the cost of latency.
3) **Record or upload audio**
   - Use the mic button or upload a WAV query.
4) **Force language** (optional)
   - If auto-detection is unreliable for your content, choose the correct language (`en`, `hi`, `es`, `ta`, `ar`).
5) **Review results**
   - The app displays detected language, your transcript, the inferred intent, and the localized agent response.
6) **Listen to the response**
   - TTS synthesizes the response in the cloned voice. If XTTS fails, the app falls back automatically and displays a warning.

## Configuration
- **Languages**: The supported languages are defined in `SUPPORTED_LANG_CODES` within `app.py`.
- **Intent bank**: Example utterances per intent and language are defined in `build_intent_bank()`.
- **Responses**: Localized response templates per intent are in `response_templates()`.
- **STT compute**: The app defaults to int8 CPU inference for faster-whisper. You can modify device/compute type in `load_stt_model()` if you have a GPU.

## Customization
### Add a new language
1) Add the ISO-639-1 code to `SUPPORTED_LANG_CODES` in `app.py`.
2) Extend `build_intent_bank()` with several example utterances per intent in the new language.
3) Add localized strings for each intent in `response_templates()`.

### Add or modify intents
- In `build_intent_bank()`, add a new intent key and provide multiple example phrases per supported language.
- In `response_templates()`, add the localized responses for that intent.

## Performance & Quality Tips
- **STT model size**: `medium` offers better accuracy; `small` balances speed and quality on CPU.
- **Language override**: If English speech is mis-detected, set Force language to `en`.
- **Voice cloning**: Use a clean, noise-free 5–10s reference WAV. Keep recording conditions consistent.
- **GPU**: If available, significantly improves XTTS/YourTTS speed and quality.

## Troubleshooting
- **Coqui ToS error**
  - Ensure `COQUI_TOS_AGREED=1` is set (see Installation step 3).

- **XTTS “generate” AttributeError or load issues**
  - This repo pins: `TTS==0.21.3` and `transformers==4.41.2` for compatibility.
  - Clear the XTTS model cache and relaunch to re-download weights:
    - Windows: delete `%APPDATA%/tts/tts_models/multilingual/multi-dataset/xtts_v2`
    - macOS/Linux: delete `~/.local/share/tts/tts_models/multilingual/multi-dataset/xtts_v2`
  - The app will automatically fall back to YourTTS, then Tacotron2, and show a warning in the UI.

- **Wrong language detected**
  - Use “Force language”, try a larger STT model, and ensure your mic audio is clear.

- **Clumsy/garbled audio**
  - Provide a better-quality reference WAV, reduce background noise, and try GPU if available.

## Deployment on Hugging Face Spaces
1) Create a Space
   - Type: Streamlit
   - Hardware: CPU works; GPU recommended for better latency
2) Add files
   - `app.py`, `requirements.txt`, `harvard.wav`, `README.md`
3) Environment
   - Add Space secret or variable: `COQUI_TOS_AGREED=1` (if needed)
4) Deploy
   - Push to the Space repo; it will build and launch automatically.

## Security & Privacy
- All inference runs locally or within your Space; no paid/3rd-party API calls.
- Do not upload sensitive data to public Spaces or repositories.
- Review model licenses for any additional obligations.

## Roadmap
- Pluggable LLM-based response generation (in addition to templates)
- Telephony integration (SIP/WebRTC)
- Session memory and CRM integration
- Enhanced real-time streaming with sub-2s end-to-end latency on CPU

## License
Open-source components only. Respect model and dataset licenses (Coqui TTS, Hugging Face models, etc.).

## Acknowledgments
- STT: faster-whisper / Whisper
- NLP: sentence-transformers
- TTS: Coqui TTS (XTTS v2, YourTTS, Tacotron2)


