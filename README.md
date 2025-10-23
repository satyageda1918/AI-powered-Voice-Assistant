<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=header&text=🎙️%20AI%20Voice%20Cloning&fontSize=40&fontAlignY=40&animation=fadeIn&desc=Multilingual%20Customer%20Support%20%7C%20Voice%20Synthesis%20%7C%20100%25%20Open-Source&descAlignY=65&descSize=16" />

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=24&duration=3000&pause=1000&color=9333EA&center=true&vCenter=true&width=800&lines=🌍+Multilingual+Voice+Support;🎤+Real-time+Voice+Cloning;🗣️+Speech-to-Speech+Pipeline;🔊+Natural+Voice+Synthesis" alt="Typing SVG" />

<p align="center">
  <img src="https://img.shields.io/badge/Voice_Cloning-XTTS_v2-9333EA?style=for-the-badge&logo=microphone&logoColor=white" alt="Voice Cloning" />
  <img src="https://img.shields.io/badge/Open_Source-100%25-00D4AA?style=for-the-badge&logo=opensource&logoColor=white" alt="Open Source" />
  <img src="https://img.shields.io/badge/Multilingual-5_Languages-FF6B6B?style=for-the-badge&logo=translate&logoColor=white" alt="Multilingual" />
  <img src="https://img.shields.io/badge/Streamlit-Interface-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/vinayabc1824/AI-Voice-Cloning-for-Customer-Support">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-Try_Now-9333EA?style=for-the-badge&logo=rocket&logoColor=white" alt="Live Demo" />
  </a>
  <img src="https://img.shields.io/github/stars/yourusername/ai-voice-cloning?style=social" alt="GitHub stars" />
</p>

---

### 🌟 Production-ready, 100% open-source solution for multilingual, localized, voice-based customer support

> 💬 Speak naturally in any supported language — the system transcribes, understands, and replies in a cloned voice, all without paid APIs.

</div>

---

<div align="center">

## ✨ Features & Capabilities

<img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="400">

</div>

<table>
<tr>
<td align="center" width="33%">
<img src="https://user-images.githubusercontent.com/74038190/212257472-08e52665-c503-4bd9-aa20-f5a4dae769b5.gif" width="100"><br>
<h3>🔓 Fully Open-Source</h3>
<p>All components run locally or on Hugging Face Spaces without any paid APIs</p>
</td>
<td align="center" width="33%">
<img src="https://user-images.githubusercontent.com/74038190/212257468-1e9a91f1-b626-4baa-b15d-5c385dfa7763.gif" width="100"><br>
<h3>🎤 Multilingual STT</h3>
<p>faster-whisper with optional language override for accurate transcription</p>
</td>
<td align="center" width="33%">
<img src="https://user-images.githubusercontent.com/74038190/212257465-7ce8d493-cac5-494e-982a-5a9deb852c4b.gif" width="100"><br>
<h3>🧠 Intent Detection</h3>
<p>Language-aware intent recognition with confidence thresholding</p>
</td>
</tr>
<tr>
<td align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257460-738ff738-247f-4445-a718-cdd0ca76e2db.gif" width="100"><br>
<h3>🗣️ Voice Cloning TTS</h3>
<p>Coqui TTS (XTTS v2) with YourTTS and Tacotron2 fallbacks</p>
</td>
<td align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257463-4d082cb4-7483-4eaf-bc25-6dde2628aabd.gif" width="100"><br>
<h3>🌍 Language Support</h3>
<p>Built-in support for English, Hindi, Spanish, Tamil, and Arabic</p>
</td>
<td align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257469-7e8c204f-c544-41f8-a292-85c262fcf4bd.gif" width="100"><br>
<h3>🎨 Modern UI</h3>
<p>Streamlit interface with mic recording, WAV upload, and real-time playback</p>
</td>
</tr>
</table>

---

## 🛠 System Architecture

1. **STT:** faster-whisper transcribes audio + detects language.
2. **NLP:** sentence-transformers retrieves intent via cosine similarity.
3. **NLG:** Localized template-based responses per language.
4. **TTS:** Coqui XTTS v2 clones voice; fallbacks ensure reliability.

### 📊 Architecture Diagram

```
┌───────────────────────────┐    ┌──────────────────────┐    ┌────────────────────────┐
│   User Audio Input        │───▶│ Speech-to-Text (STT) │──▶│ Language Detection     │
│ (Mic or File Upload)      │    │   Faster-Whisper     │    │                        │
└───────────────────────────┘    └──────────────────────┘    └────────────────────────┘
             │                           │                             │
             ▼                           ▼                             ▼
┌───────────────────────────┐    ┌──────────────────────┐    ┌────────────────────────┐
│ NLP Intent Recognition    │    │ Localized Response   │    │ Text-to-Speech (TTS)   │
│ (Transformers / spaCy)    │    │ Generation           │    │   XTTS v2 / Fallbacks  │
└───────────────────────────┘    └──────────────────────┘    └────────────────────────┘
                                                                │
                                                                ▼
                                                     ┌────────────────────────┐
                                                     │   Audio Playback       │
                                                     │   (User's Voice Clone) │
                                                     └────────────────────────┘


```

---

## 📦 Requirements

* **OS:** Windows / macOS / Linux
* **Python:** 3.9–3.11
* **Hardware:** CPU-friendly, GPU recommended for speed

---

## ⚡ Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

# Accept Coqui TTS Terms (if prompted)
export COQUI_TOS_AGREED=1  # Linux/macOS
$env:COQUI_TOS_AGREED="1"  # Windows
```

---

## 🚀 Quick Start

```bash
streamlit run app.py
```

Open `http://localhost:8501` or [try it live here](https://huggingface.co/spaces/vinayabc1824/AI-Voice-Cloning-for-Customer-Support).

---

## 🎯 Usage

1. Upload a **clean 5–10s mono WAV** as reference voice.
2. Choose **STT model size** (`tiny`/`base`/`small`/`medium`).
3. Record or upload query audio.
4. (Optional) Force language detection.
5. Review transcript, intent, and localized response.
6. Listen to the cloned-voice reply.

---

## 🔧 Configuration & Customization

* Add languages in `SUPPORTED_LANG_CODES`.
* Modify intent examples in `build_intent_bank()`.
* Update responses in `response_templates()`.
* Adjust STT device/precision in `load_stt_model()`.

---

## 💡 Performance Tips

* Use **GPU** for faster, higher-quality TTS.
* Keep reference voice **noise-free**.
* Use **medium** STT for better accuracy.

---

## 🐛 Troubleshooting

* **Coqui TOS error:** set `COQUI_TOS_AGREED=1`.
* **XTTS load errors:** use `TTS==0.21.3` & `transformers==4.41.2`; clear model cache.
* **Language mis-detection:** force language in UI.

---

## 📤 Deploy on Hugging Face Spaces

1. Create Space → Type: Streamlit.
2. Add `app.py`, `requirements.txt`, `harvard.wav`, `README.md`.
3. Add `COQUI_TOS_AGREED=1` in secrets.
4. Push to repo → auto-build.

**Live demo:** [https://huggingface.co/spaces/vinayabc1824/AI-Voice-Cloning-for-Customer-Support](https://huggingface.co/spaces/vinayabc1824/AI-Voice-Cloning-for-Customer-Support)

---

## 🔒 Security & Privacy

* All processing is local or within your Space.
* Avoid uploading sensitive data to public repos.

---

## 🗺 Roadmap

* LLM-powered dynamic replies
* Telephony integration
* CRM connectors
* Real-time streaming under 2s

---

## 📜 License

Open-source only — respect individual model licenses.

---

## 🙌 Acknowledgments

* **STT:** faster-whisper
* **NLP:** sentence-transformers
* **TTS:** Coqui TTS (XTTS v2, YourTTS, Tacotron2)

<div align="center">

---

## 🤝 Contributing

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif" width="100">

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

<table>
<tr>
<td align="center">
<img src="https://img.shields.io/badge/Issues-Welcome-brightgreen?style=for-the-badge&logo=github" /><br>
<b>Report Bugs</b>
</td>
<td align="center">
<img src="https://img.shields.io/badge/PRs-Welcome-blue?style=for-the-badge&logo=git" /><br>
<b>Submit PRs</b>
</td>
<td align="center">
<img src="https://img.shields.io/badge/Ideas-Welcome-purple?style=for-the-badge&logo=lightbulb" /><br>
<b>Share Ideas</b>
</td>
</tr>
</table>

## 📊 Project Stats

<div align="center">
<img src="https://github-readme-stats.vercel.app/api?username=yourusername&repo=ai-voice-cloning&show_icons=true&theme=radical" alt="GitHub Stats" />
</div>

## 📄 License

<img src="https://img.shields.io/badge/License-Open_Source-yellow.svg?style=for-the-badge" alt="Open Source License" />

Open-source only — respect individual model licenses.

---

<div align="center">

### 💜 Built with ❤️ by Rohith Cherukuri

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="500">


**⭐ Star this repo if you found it helpful!**

</div>

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer" />
</div>






