# tts_engine.py
import edge_tts
import asyncio
import tempfile
import os

class TTSEngine:
    def __init__(self, voice="en-US-AriaNeural"):
        self.voice = voice  # natural female voice

    async def _synthesize(self, text: str, output_path: str):
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)

    def speak_to_file(self, text: str) -> str:
        """Returns path to generated .mp3 file"""
        tmp = tempfile.mktemp(suffix=".mp3")
        asyncio.run(self._synthesize(text, tmp))
        return tmp