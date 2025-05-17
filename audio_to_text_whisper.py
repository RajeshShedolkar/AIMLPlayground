# !pip install -U openai-whisper
# !apt-get install -y ffmpeg
from google.colab import files
uploaded = files.upload()

# Step 3: Transcribe
import whisper

model = whisper.load_model("base")  # use "tiny", "small", etc. for faster performance
audio_file = list(uploaded.keys())[0]  # Get the uploaded filename

result = model.transcribe(audio_file)
print("Transcribed text:\n")
print(result["text"])
