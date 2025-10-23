"""
Streamlit app: AI Voice Cloning for Multilingual Localized Customer Support

Pipeline:
  Mic/Upload (voice) -> faster-whisper (STT) -> multilingual intent classification
  -> templated response in detected language -> Coqui XTTS v2 (voice cloning TTS)

All components are open-source. No external paid APIs used.
"""

import os
import io
import tempfile
from typing import Dict, List, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
import streamlit as st
from audio_recorder_streamlit import audio_recorder

from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer, util
from langdetect import detect as lang_detect
from TTS.api import TTS

# ---- PyTorch safe unpickling patches for XTTS ----
try:
    from torch.serialization import add_safe_globals
    from TTS.tts.models.xtts import XttsArgs
    add_safe_globals([XttsArgs])
except Exception:
    pass

import torch
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    torch.serialization.add_safe_globals([XttsConfig])
except Exception:
    pass

try:
    from TTS.tts.models.xtts import XttsAudioConfig
    torch.serialization.add_safe_globals([XttsAudioConfig])
except Exception:
    pass

try:
    from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
    torch.serialization.add_safe_globals([BaseDatasetConfig, BaseAudioConfig])
except Exception:
    pass

# ------------------------------
# Configuration & Caching
# ------------------------------

SUPPORTED_LANG_CODES = ["en", "hi", "es", "ta", "ar"]
DEFAULT_LANGUAGE = "en"

# Agree to Coqui TTS Terms of Service for XTTS v2
os.environ.setdefault("COQUI_TOS_AGREED", "1")


@st.cache_resource(show_spinner=True)
def load_stt_model(model_size: str = "small", compute_type: str = "int8") -> WhisperModel:
    return WhisperModel(
        model_size_or_path=model_size,
        device="cpu",
        compute_type=compute_type,
    )


@st.cache_resource(show_spinner=True)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource(show_spinner=True)
def load_tts_model() -> TTS:
    try:
        return TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
    except Exception:
        return TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)


def get_speaker_wav_path(uploaded_file, default_fallback: str = "harvard.wav") -> str:
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            return tmp.name
    if os.path.exists(default_fallback):
        return default_fallback
    return ""


def prepare_audio_for_stt(wav_bytes: bytes) -> str:
    """Resample uploaded/recorded audio to 16kHz mono WAV."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        audio, _ = librosa.load(tmp.name, sr=16000, mono=True)
        sf.write(tmp.name, audio, 16000, format="WAV")
    return tmp.name


def transcribe_audio(stt: WhisperModel, wav_bytes: bytes, forced_language: Optional[str] = None) -> Tuple[str, str]:
    audio_path = prepare_audio_for_stt(wav_bytes)
    transcribe_kwargs = {"vad_filter": True}
    if forced_language and forced_language in SUPPORTED_LANG_CODES:
        transcribe_kwargs["language"] = forced_language
    segments, info = stt.transcribe(audio_path, **transcribe_kwargs)
    text_parts = [seg.text for seg in segments]
    transcript = " ".join(t.strip() for t in text_parts).strip()

    language = forced_language or (info.language if info and info.language else "")
    if not language:
        try:
            language = lang_detect(transcript)
        except Exception:
            language = DEFAULT_LANGUAGE

    lang = language[:2].lower() if language else DEFAULT_LANGUAGE
    # Heuristic: if mostly ASCII alphabetic text, prefer English when auto-detected otherwise
    if not forced_language and lang != "en":
        letters = sum(ch.isalpha() for ch in transcript)
        ascii_letters = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in transcript)
        if letters > 0 and ascii_letters / max(letters, 1) > 0.7:
            lang = "en"
    if lang not in SUPPORTED_LANG_CODES:
        lang = DEFAULT_LANGUAGE

    try:
        os.remove(audio_path)
    except Exception:
        pass
    return transcript, lang


def build_intent_bank() -> Dict[str, Dict[str, List[str]]]:
    return {
        "greeting": {
            "en": ["hi", "hello", "good morning", "hey", "is anyone there"],
            "hi": ["नमस्ते", "हाय", "क्या कोई है"],
            "es": ["hola", "buenos días", "hola, hay alguien"],
            "ta": ["வணக்கம்", "ஹாய்"],
            "ar": ["مرحبا", "أهلاً", "صباح الخير"],
        },
        "refund_status": {
            "en": ["refund status", "where is my refund", "refund request", "my refund details"],
            "hi": ["रिफंड की स्थिति", "मेरा रिफंड कहाँ है"],
            "es": ["estado del reembolso", "dónde está mi reembolso"],
            "ta": ["பணம் திருப்பி அளித்த நிலை", "என் ரீஃபண்ட் எங்கே"],
            "ar": ["حالة الاسترداد", "أين استردادي"],
        },
        "account_balance": {
            "en": ["account balance", "current balance", "how much money"],
            "hi": ["खाते का बैलेंस", "मौजूदा बैलेंस"],
            "es": ["saldo de la cuenta", "saldo actual"],
            "ta": ["கணக்கு இருப்பு", "தற்போதைய இருப்பு"],
            "ar": ["رصيد الحساب", "الرصيد الحالي"],
        },
        "product_info": {
            "en": ["product information", "tell me about the plan", "features"],
            "hi": ["उत्पाद की जानकारी", "योजना बताइए", "विशेषताएँ"],
            "es": ["información del producto", "háblame del plan", "características"],
            "ta": ["தயாரிப்பு தகவல்", "திட்டம் பற்றி சொல்லுங்கள்", "அம்சங்கள்"],
            "ar": ["معلومات المنتج", "أخبرني عن الخطة", "الميزات"],
        },
    }


@st.cache_resource(show_spinner=False)
def embed_intent_bank(_embedder: SentenceTransformer):
    bank = build_intent_bank()
    texts, keys = [], []
    for intent, langs in bank.items():
        for lang, examples in langs.items():
            for i, ex in enumerate(examples):
                texts.append(ex)
                keys.append((intent, lang, i))
    embs = _embedder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    idx_to_emb = {str(i): embs[i] for i in range(len(texts))}
    idx_lookup = [f"{keys[i][0]}::{keys[i][1]}::{keys[i][2]}" for i in range(len(keys))]
    return bank, idx_to_emb, idx_lookup


def classify_intent(embedder, bank_embs, idx_lookup, utterance, lang: str, min_score: float = 0.5):
    if not utterance.strip():
        return "fallback"
    query_emb = embedder.encode([utterance], convert_to_tensor=True, normalize_embeddings=True)[0]
    scores = []
    for i_str, emb in bank_embs.items():
        i = int(i_str)
        meta = idx_lookup[i]
        _, ex_lang, _ = meta.split("::")
        if ex_lang != lang:
            continue
        score = util.cos_sim(query_emb, emb).item()
        scores.append((float(score), i))
    if not scores:
        return "fallback"
    scores.sort(reverse=True)
    top_score, top_i = scores[0]
    if top_score < min_score:
        return "fallback"
    return idx_lookup[top_i].split("::")[0]


def response_templates() -> Dict[str, Dict[str, str]]:
    return {
        "greeting": {
            "en": "Hello! How can I help you today?",
            "hi": "नमस्ते! मैं आपकी कैसे सहायता कर सकता/सकती हूँ?",
            "es": "¡Hola! ¿En qué puedo ayudarte hoy?",
            "ta": "வணக்கம்! இன்று உங்களுக்கு எப்படி உதவலாம்?",
            "ar": "مرحباً! كيف يمكنني مساعدتك اليوم؟",
        },
        "refund_status": {
            "en": "I can help with refunds. Could you share your order ID?",
            "hi": "मैं रिफंड में सहायता कर सकता/सकती हूँ। कृपया अपना ऑर्डर आईडी बताएं।",
            "es": "Puedo ayudarte con reembolsos. ¿Puedes compartir tu ID de pedido?",
            "ta": "ரீஃபண்ட் தொடர்பாக உதவ முடியும். உங்கள் ஆர்டர் ஐடியை பகிரவும்.",
            "ar": "يمكنني المساعدة في الاسترداد. هل يمكنك مشاركة رقم طلبك؟",
        },
        "account_balance": {
            "en": "To check your balance, please verify your account number.",
            "hi": "बैलेंस देखने के लिए, कृपया अपना खाता नंबर सत्यापित करें।",
            "es": "Para verificar tu saldo, por favor valida tu número de cuenta.",
            "ta": "இருப்பை பார்க்க, தயவுசெய்து உங்கள் கணக்கு எண்ணை சரிபார்க்கவும்.",
            "ar": "للتحقق من رصيدك، يرجى تأكيد رقم حسابك.",
        },
        "product_info": {
            "en": "Sure, which product or plan would you like to know about?",
            "hi": "ज़रूर, किस उत्पाद या योजना के बारे में जानना चाहेंगे?",
            "es": "Claro, ¿sobre qué producto o plan te gustaría saber?",
            "ta": "நிச்சயமாக, எந்த தயாரிப்பு அல்லது திட்டம் பற்றி அறிய விரும்புகிறீர்கள்?",
            "ar": "بالتأكيد، عن أي منتج أو خطة تود معرفة المزيد؟",
        },
        "fallback": {
            "en": "Sorry, I didn't get that. Could you rephrase?",
            "hi": "क्षमा करें, मैं समझ नहीं पाया/पाई। कृपया दोबारा कहें।",
            "es": "Lo siento, no entendí. ¿Podrías reformular?",
            "ta": "மன்னிக்கவும், எனக்கு புரியவில்லை. தயவுசெய்து மறுபரிசீலனை செய்யவா?",
            "ar": "عذراً، لم أفهم. هل يمكنك إعادة الصياغة؟",
        },
    }


def pick_response(intent: str, lang: str) -> str:
    templates = response_templates()
    if intent not in templates:
        intent = "fallback"
    return templates[intent].get(lang, templates[intent]["en"])


def synthesize(tts: TTS, text: str, speaker_wav: str, language: str) -> Tuple[bytes, int]:
    lang = language if language in SUPPORTED_LANG_CODES else DEFAULT_LANGUAGE
    sample_rate = 22050
    wav = None

    try:
        wav = tts.tts(text=text, speaker_wav=speaker_wav or None, language=lang)
        sample_rate = getattr(tts.synthesizer, "output_sample_rate", 22050)
    except Exception:
        try:
            fallback = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
            wav = fallback.tts(text=text, speaker_wav=speaker_wav or None, language=lang)
            sample_rate = getattr(fallback.synthesizer, "output_sample_rate", 22050)
            st.warning("XTTS failed; used YourTTS fallback.")
        except Exception:
            basic = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
            wav = basic.tts(text=text)
            sample_rate = getattr(basic.synthesizer, "output_sample_rate", 22050)
            st.warning("XTTS and YourTTS failed; used basic English TTS.")

    if wav is None or len(wav) == 0:
        raise RuntimeError("TTS synthesis failed — no waveform generated.")

    with io.BytesIO() as bio:
        sf.write(bio, wav, sample_rate, format="WAV")
        bio.seek(0)
        return bio.read(), sample_rate


def main():
    st.set_page_config(page_title="AI Voice Cloning Support", layout="centered")
    st.title("AI Voice Cloning for Multilingual Localized Customer Support")
    st.caption("Open-source STT + NLP + TTS with voice cloning. No cloud APIs.")

    # Sidebar settings
    with st.sidebar:
        st.header("Voice & Model Settings")
        uploaded_voice = st.file_uploader("Upload reference voice (WAV, ~5-10s)", type=["wav"])
        speaker_wav_path = get_speaker_wav_path(uploaded_voice)
        st.write("Using reference:", os.path.basename(speaker_wav_path) if speaker_wav_path else "None")
        st.markdown("- Supported languages: en, hi, es, ta, ar")
        realtime_hint = st.toggle("Low-latency mode (shorter recordings)", value=True)
        stt_size = st.selectbox("STT model size", ["tiny", "base", "small", "medium"], index=2)
        forced_lang_choice = st.selectbox("Force language (optional)", ["auto", *SUPPORTED_LANG_CODES], index=0)

    # Load models
    stt = load_stt_model(model_size=stt_size)
    embedder = load_embedder()
    tts = load_tts_model()
    bank, bank_embs, idx_lookup = embed_intent_bank(embedder)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "text": "Hello! Speak or type your question to get started."}
        ]

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/wav")

    # Input: text and microphone
    user_prompt = st.chat_input("Type your message and press Enter…")
    audio_bytes = audio_recorder(
        text="Hold to Talk" if realtime_hint else "Talk",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        sample_rate=16000 if realtime_hint else 44100,
    )

    # Optional upload
    with st.expander("Upload a WAV instead of speaking"):
        upload = st.file_uploader("Upload WAV", type=["wav"])
        if upload is not None:
            audio_bytes = upload.read()

    def detect_lang_from_text(text: str, forced: Optional[str]) -> str:
        if forced and forced in SUPPORTED_LANG_CODES:
            return forced
        try:
            language = lang_detect(text)
        except Exception:
            language = DEFAULT_LANGUAGE
        lang = language[:2].lower() if language else DEFAULT_LANGUAGE
        letters = sum(ch.isalpha() for ch in text)
        ascii_letters = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in text)
        if lang != "en" and letters > 0 and ascii_letters / max(letters, 1) > 0.7:
            lang = "en"
        if lang not in SUPPORTED_LANG_CODES:
            lang = DEFAULT_LANGUAGE
        return lang

    lang_hint = None if forced_lang_choice == "auto" else forced_lang_choice

    # Handle microphone/upload turn
    if audio_bytes:
        with st.spinner("Transcribing…"):
            user_text, lang = transcribe_audio(stt, audio_bytes, forced_language=lang_hint)
        st.session_state.messages.append({"role": "user", "text": user_text, "audio": audio_bytes})
        with st.chat_message("user"):
            st.markdown(user_text)
            st.audio(audio_bytes, format="audio/wav")
        with st.spinner("Understanding…"):
            intent = classify_intent(embedder, bank_embs, idx_lookup, user_text, lang)
        agent_text = pick_response(intent, lang)
        with st.chat_message("assistant"):
            st.markdown(f"Detected language: {lang}\n\nIntent: {intent}\n\n{agent_text}")
            with st.spinner("Synthesizing…"):
                audio_out, _ = synthesize(tts, agent_text, speaker_wav_path, lang)
            st.audio(audio_out, format="audio/wav")
        st.session_state.messages.append({"role": "assistant", "text": agent_text, "audio": audio_out})

    # Handle typed turn
    elif user_prompt:
        lang = detect_lang_from_text(user_prompt, lang_hint)
        st.session_state.messages.append({"role": "user", "text": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.spinner("Understanding…"):
            intent = classify_intent(embedder, bank_embs, idx_lookup, user_prompt, lang)
        agent_text = pick_response(intent, lang)
        with st.chat_message("assistant"):
            st.markdown(f"Detected language: {lang}\n\nIntent: {intent}\n\n{agent_text}")
            with st.spinner("Synthesizing…"):
                audio_out, _ = synthesize(tts, agent_text, speaker_wav_path, lang)
            st.audio(audio_out, format="audio/wav")
        st.session_state.messages.append({"role": "assistant", "text": agent_text, "audio": audio_out})

    st.markdown("---")
    with st.expander("How to add a new language"):
        st.markdown(
            """
            - Add your ISO-639-1 code to `SUPPORTED_LANG_CODES`.
            - Extend `build_intent_bank()` with example utterances for that language.
            - Add localized response strings in `response_templates()` for each intent.
            - XTTS v2 supports many languages; set the same code in the `language` parameter of TTS.
            """
        )


if __name__ == "__main__":
    main()
