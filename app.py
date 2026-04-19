import streamlit as st
import whisper
import os
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from deep_translator import GoogleTranslator
from gtts import gTTS
from tempfile import NamedTemporaryFile

# -- 🎨 CUSTOM THEME --
st.set_page_config(page_title="ORANGE Pocket Translator", page_icon="🍊")
st.markdown("""
    <style>
    .stApp { background-color: #fdfefe; }
    .stButton>button { background-color: #2e86c1; color: white; border-radius: 20px; }
    h1 { color: #1b4f72; font-family: 'Roboto'; }
    </style>
    """, unsafe_allow_html=True)

st.title("🍊 ORANGE Pocket Translator")

@st.cache_resource
def load_model():
    # 'small' is the highest we can go on free hosting safely!
    return whisper.load_model("small")

model = load_model()

# -- 🎤 AUDIO CLEANER FUNCTION --
def clean_audio(input_path):
    audio = AudioSegment.from_file(input_path)
    # Convert to numpy array for noise reduction
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    # Reduce noise (stationary noise reduction)
    reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate)
    # Convert back to audio
    cleaned_audio = audio._spawn(reduced_noise.astype(np.int16).tobytes())
    cleaned_path = "cleaned_audio.wav"
    cleaned_audio.export(cleaned_path, format="wav")
    return cleaned_path

# -- 1. INPUT --
tab1, tab2 = st.tabs(["🎤 Live Mic", "📁 Upload"])
audio_source = None
with tab1:
    mic = st.audio_input("Tap to record")
    if mic: audio_source = mic
with tab2:
    up = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
    if up: audio_source = up

# -- 2. SETTINGS --
target_lang = st.selectbox("Translate to:", ["English", "Spanish", "Thai", "Vietnamese", "German", "Japanese", "Chinese", "Tagalog"])
lang_codes = {"English": "en", "Spanish": "es", "Thai": "th", "Vietnamese": "vi", "German": "de", "Japanese": "ja", "Chinese": "zh-cn", "Tagalog": "tl"}

# -- 3. THE MAGIC --
if audio_source:
    if st.button("🚀 Translate with Noise Cleaning"):
        with st.spinner("Scrubbing audio and translating..."):
            try:
                # Save raw
                with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_source.getvalue())
                    raw_path = tmp.name

                # CLEAN THE AUDIO FIRST
                processed_path = clean_audio(raw_path)

                # TRANSLATE
                result = model.transcribe(processed_path)
                orig_text = result["text"]
                
                # FINAL POLISH (Deep Translator)
                final_text = GoogleTranslator(source='auto', target=lang_codes[target_lang]).translate(orig_text)

                # VOICE OUTPUT
                tts = gTTS(text=final_text, lang=lang_codes[target_lang])
                tts.save("out.mp3")

                # DISPLAY
                st.success("Translation Ready!")
                st.subheader(f"Results ({target_lang}):")
                st.write(f"**What I heard:** {orig_text}")
                st.write(f"**Translation:** {final_text}")
                st.audio("out.mp3")

                os.remove(raw_path)
                os.remove(processed_path)
            except Exception as e:
                st.error(f"Error: {e}")
