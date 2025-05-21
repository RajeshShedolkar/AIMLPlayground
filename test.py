# Step 1: Install required libraries
!pip install -q git+https://github.com/openai/whisper.git
!pip install -q git+https://github.com/coqui-ai/TTS
!pip install -q ffmpeg-python

# Step 2: Import libraries
import whisper
import IPython.display as ipd
from TTS.api import TTS

# Step 3: Load Whisper model
model = whisper.load_model("small")  # Or use "base", "medium", "large"

# Step 4: Upload Hindi audio file
from google.colab import files
uploaded = files.upload()

# Step 5: Transcribe uploaded audio
file_name = list(uploaded.keys())[0]
result = model.transcribe(file_name, language="hi")
print("Transcription:")
print(result["text"])

# Step 6: Text to Speech (Hindi)
# Check supported languages and voices
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)

# Generate speech
tts.tts_to_file(text=result["text"], file_path="output.wav", speaker="hin")

# Step 7: Play the output audio
ipd.Audio("output.wav")
