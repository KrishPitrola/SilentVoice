# nlp_corrector.py
from groq import Groq
import os

class NLPCorrector:
    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.system = """You convert lip-reading output codes into natural spoken sentences.
Input is GRID corpus format: COMMAND COLOR PREPOSITION LETTER NUMBER ADVERB
Examples:
- "BIN BLUE AT F TWO NOW" → "Place the blue item at F2 now."
- "SET WHITE WITH B TWO NOW" → "Set the white one with B2 now."
Rules:
- Output one clean natural sentence only
- No explanation, no extra text"""

    def correct(self, raw_text: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": raw_text.strip().upper()}
                ],
                max_tokens=60,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[NLPCorrector] Error: {e}")
            return raw_text.capitalize()