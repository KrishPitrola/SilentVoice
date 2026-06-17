from collections import Counter

PHRASE_MAP = {
    "bin blue":  "Place it in the bin",
    "place":     "Place it there",
    "lay":       "Lay it down",
    "bin":       "Put it in the bin",
    "blue":      "Select blue",
    "green":     "Select green",
    "white":     "Select white",
    "again":     "Please repeat that",
    "please":    "Please help me",
    "now":       "I need help now",
    "set red":   "reading unknown",
}

class FallbackMapper:
    def map(self, raw_text):
        lowered = raw_text.lower()
        for key, phrase in PHRASE_MAP.items():
            if key in lowered:
                return phrase
        return raw_text.strip().title()

    def is_garbage(self, text):
        stripped = text.strip()
        if len(stripped) < 2:
            return True
        counts = Counter(stripped)
        ratio = counts.most_common(1)[0][1] / len(stripped)
        return ratio > 0.60