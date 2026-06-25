# pipeline_test.py
from mpc001_wrapper import MPC001VSR
from nlp_corrector import NLPCorrector
from tts_engine import TTSEngine
import os

print("Loading models...")
vsr = MPC001VSR()
nlp = NLPCorrector()
tts = TTSEngine()
print("All models loaded.\n")

video = "S:\SilentVoice\data\grid\s1\swwp2n.mpg"

print(f"Step 1 — VSR on: {video}")
raw = vsr.transcribe(video)
print(f"Raw:       {raw}")

print(f"Step 2 — NLP correction")
corrected = nlp.correct(raw)
print(f"Corrected: {corrected}")

print(f"Step 3 — TTS")
audio_path = tts.speak_to_file(corrected)
print(f"Audio:     {audio_path}")

# Auto-play on Windows
os.startfile(audio_path)