import streamlit as st
import whisper
import os
from tempfile import NamedTemporaryFile

# -- UI Configuration --
st.set_page_config(page_title="Universal Audio Translator", page_icon="🌐")
st.title("🌐 Audio to English Text")
st.markdown("Upload audio in **any language** to get an English translation.")

# -- Model Loading --
# We use 'tiny' because it's the fastest and works best on free hosting
@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

model = load_model()

# -- File Uploader --
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a", "ogg", "flac"])

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    
    if st.button("Translate to English"):
        with st.spinner("Processing... this takes a moment on the free server."):
            try:
                # Save uploaded file to a temporary location
                with NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
                    tmp.write(audio_file.getvalue())
                    tmp_path = tmp.name

                # Perform Translation
                result = model.transcribe(tmp_path, task="translate")

                # -- Display Result --
                st.success("Translation Complete!")
                st.subheader("English Transcription:")
                st.write(result["text"])
                
                # Clean up the temp file
                os.remove(tmp_path)

            except Exception as e:
                st.error(f"An error occurred: {e}")
