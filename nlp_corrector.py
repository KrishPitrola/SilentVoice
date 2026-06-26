# nlp_corrector.py
from groq import Groq
import os

class NLPCorrector:
    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.system = """You are a speech correction assistant for a lip-reading system.
The input is raw transcription output from a visual speech recognition model.
It may contain errors, missing words, or slightly garbled text.

Your job:
- Fix transcription errors and make it a clean natural English sentence
- Do NOT change the meaning
- Do NOT add information that wasn't there
- Output one clean sentence only
- No explanation, no extra text

Examples:
- "AND HERE'S FIVE TIPS FOR GETTING COMFORTABLE" → "And here are five tips for getting comfortable."
- "i wnt to go hm" → "I want to go home."
- "the quik brwn fox" → "The quick brown fox."""

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