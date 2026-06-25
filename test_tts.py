# test_tts.py
from tts_engine import TTSEngine

tts = TTSEngine()

sentences = [
    "Set the white one with B2 now.",
    "Place the blue item at F2 now.",
]

for s in sentences:
    print(f"Synthesizing: {s}")
    path = tts.speak_to_file(s)
    print(f"Saved: {path}\n")