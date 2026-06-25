# test_nlp.py
from nlp_corrector import NLPCorrector
import time

corrector = NLPCorrector()

tests = [
    "SET WHITE WITH B TWO NOW",
    "BIN BLUE AT F TWO NOW",
    "i wnt to go hm",
]

for t in tests:
    print(f"IN:  {t}")
    print(f"OUT: {corrector.correct(t)}\n")
    time.sleep(4)  # stay under 15 RPM free tier