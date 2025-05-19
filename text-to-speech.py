# # ✅ STEP 1: Install Required Libraries
# !pip install git+https://github.com/openai/whisper.git
# !pip install torch
# !pip install deep-translator
# !pip install TTS

# ✅ STEP 2: Upload Audio File
from google.colab import files
print("📤 Upload your audio file (e.g., input_audio.mp3)")
uploaded = files.upload()

# ✅ STEP 3: Import Libraries
import whisper
from deep_translator import GoogleTranslator
from TTS.api import TTS
from IPython.display import Audio
import os

# ✅ STEP 4: Transcribe Audio
def transcribe_audio(audio_path, language='English', model_size='medium'):
    print("🧠 Loading Whisper model...")
    model = whisper.load_model(model_size)
    print(f"🔍 Transcribing: {audio_path}")
    result = model.transcribe(audio_path, language=language)
    return result["text"]

# ✅ STEP 5: Translate to Hindi
def translate_text(text, target_lang='hi'):
    print("🌐 Translating to Hindi...")
    translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
    return translated

# ✅ STEP 6: Text to Speech (TTS)
def generate_tts(text, output_audio_path="dubbed_output.wav", model_name="tts_models/hi/cv/vits"):
    print("🎙️ Generating Hindi audio...")
    tts = TTS(model_name)
    tts.tts_to_file(text=text, file_path=output_audio_path)
    print(f"✅ Dubbed audio saved to: {output_audio_path}")

# ✅ STEP 7: Main Pipeline
def dub_audio(input_audio_path):
    # Transcribe
    original_text = transcribe_audio(input_audio_path)

    # Translate
    translated_text = translate_text(original_text)

    # Save translated text
    with open("translated_text.txt", "w", encoding="utf-8") as f:
        f.write(translated_text)

    # Generate dubbed audio
    generate_tts(translated_text)

    return "dubbed_output.wav"

# ✅ STEP 8: Run the Pipeline
input_filename = list(uploaded.keys())[0]  # Get uploaded filename
output_file = dub_audio(input_filename)

# ✅ STEP 9: Play & Download Output
print("🔊 Playing dubbed output:")
display(Audio(output_file))

print("⬇️ Click below to download:")
files.download(output_file)
