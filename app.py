import streamlit as st
import whisper
import os
from tempfile import NamedTemporaryFile

# -- UI Configuration --
st.set_page_config(page_title="Universal Audio Translator", page_icon="🌐")
st.title("🌐 Audio to English Text")
st.markdown("Upload an audio file in **any language** and I'll translate it to English.")

# -- Model Loading --
# We use 'base' for speed. Use 'small' or 'medium' for better accuracy.
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# -- File Uploader --
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a", "ogg", "flac"])

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    
    if st.button("Translate to English"):
        with st.spinner("Processing audio... this may take a minute."):
            try:
                # Save uploaded file to a temporary location
                with NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
                    tmp.write(audio_file.getvalue())
                    tmp_path = tmp.name

                # Perform Translation
                # task="translate" tells Whisper to output English regardless of input language
                result = model.transcribe(tmp_path, task="translate")

                # -- Display Result --
                st.success("Translation Complete!")
                st.subheader("English Transcription:")
                st.write(result["text"])
                
                # Clean up the temp file
                os.remove(tmp_path)

            except Exception as e:
                st.error(f"An error occurred: {e}")

---

### **How it works:**
* **The Engine:** It uses OpenAI's Whisper (running locally on your machine).
* **The Magic:** The `task="translate"` parameter is the secret sauce. It detects the source language automatically and maps it to English text.
* **The Interface:** Streamlit creates a web-based dashboard so you don't have to live in the terminal.
