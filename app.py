import streamlit as st
import whisper
import os
from googletrans import Translator
from gtts import gTTS
from tempfile import NamedTemporaryFile

# -- App Config --
st.set_page_config(page_title="Global Pocket Translator", page_icon="🌍")
st.title("🌍 Global Pocket Translator")

# Language Map (Add more if you like!)
LANG_MAP = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Chinese": "zh-cn",
    "Tagalog": "tl"
}

@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

model = load_model()
translator = Translator()

# -- Step 1: Input --
st.subheader("1. Provide Audio")
tab1, tab2 = st.tabs(["🎤 Live Record", "📁 Upload File"])

audio_source = None
with tab1:
    mic_audio = st.audio_input("Record speech")
    if mic_audio: audio_source = mic_audio
with tab2:
    uploaded_audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])
    if uploaded_audio: audio_source = uploaded_audio

# -- Step 2: Settings --
st.subheader("2. Translation Settings")
target_lang_name = st.selectbox("Translate into:", list(LANG_MAP.keys()))
target_lang_code = LANG_MAP[target_lang_name]

# -- Step 3: Process --
if audio_source:
    if st.button("✨ Run Translation"):
        with st.spinner("Processing..."):
            # A. Save audio to temp file
            suffix = ".wav" if not hasattr(audio_source, 'name') else os.path.splitext(audio_source.name)[1]
            with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_source.getvalue())
                tmp_path = tmp.name

            # B. Transcribe (Detect language and turn to text)
            # We transcribe in 'original' language first
            rec_result = model.transcribe(tmp_path)
            original_text = rec_result["text"]
            
            # C. Translate Text
            translation = translator.translate(original_text, dest=target_lang_code)
            translated_text = translation.text

            # D. Generate Audio (Speech)
            tts = gTTS(text=translated_text, lang=target_lang_code)
            tts_path = "speech.mp3"
            tts.save(tts_path)

            # -- Results UI --
            st.success("Done!")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Original Text:**\n{original_text}")
            with col2:
                st.warning(f"**{target_lang_name} Translation:**\n{translated_text}")
            
            st.write("### 🔊 Listen to Translation:")
            st.audio(tts_path)

            # Cleanup
            os.remove(tmp_path)
            if os.path.exists(tts_path): os.remove(tts_path)
