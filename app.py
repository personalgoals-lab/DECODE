import streamlit as st
import whisper
import os
from deep_translator import GoogleTranslator
from gtts import gTTS
from tempfile import NamedTemporaryFile

# -- UI & CUSTOM CSS --
st.set_page_config(page_title="ORANGE Pocket Translator", page_icon="🍊")

# This is where we "inject" the design
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6; /* Light grey/blue background */
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2e4053;
    }
    .stMarkdown {
        font-family: 'Georgia', serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌍 Global Pocket Translator")

# Expanded Language Map
LANG_MAP = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Chinese": "zh-CN",
    "Tagalog": "tl",
    "Thai": "th",
    "Vietnamese": "vi",
    "Dutch": "nl",
    "Icelandic": "is",
    "Korean": "ko",
    "Italian": "it"
}

@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# -- Input Section --
st.subheader("1. Provide Audio")
tab1, tab2 = st.tabs(["🎤 Live Record", "📁 Upload File"])

audio_source = None
with tab1:
    mic_audio = st.audio_input("Record speech")
    if mic_audio: audio_source = mic_audio
with tab2:
    uploaded_audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])
    if uploaded_audio: audio_source = uploaded_audio

# -- Settings --
st.subheader("2. Settings")
target_lang_name = st.selectbox("Translate into:", list(LANG_MAP.keys()))
target_lang_code = LANG_MAP[target_lang_name]

# -- Process --
if audio_source:
    if st.button("✨ Run Translation"):
        with st.spinner("Processing..."):
            try:
                suffix = ".wav" if not hasattr(audio_source, 'name') else os.path.splitext(audio_source.name)[1]
                with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_source.getvalue())
                    tmp_path = tmp.name

                rec_result = model.transcribe(tmp_path)
                original_text = rec_result["text"]
                
                translated_text = GoogleTranslator(source='auto', target=target_lang_code).translate(original_text)

                tts = gTTS(text=translated_text, lang=target_lang_code)
                tts_path = "speech.mp3"
                tts.save(tts_path)

                st.success("Done!")
                st.info(f"**Detected Original Text:**\n{original_text}")
                st.warning(f"**{target_lang_name} Translation:**\n{translated_text}")
                st.audio(tts_path)

                os.remove(tmp_path)
                if os.path.exists(tts_path): os.remove(tts_path)

            except Exception as e:
                st.error(f"Error: {e}")
