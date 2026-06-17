"""
vocab.py — LipNet character-level CTC vocabulary and decoder utilities.

GRID_CHARS (28 tokens):
  Index 0  : CTC blank token      ← required by nn.CTCLoss
  Index 1  : space character ' '
  Index 2–27: lowercase a–z
"""

import torch

# ------------------------------------------------------------------
# Vocabulary definition
# ------------------------------------------------------------------

GRID_CHARS = [
    "-",   # 0 → CTC blank token
    " ",   # 1 → space
    "a", "b", "c", "d", "e", "f", "g", "h",  # 2–9
    "i", "j", "k", "l", "m", "n", "o", "p",  # 10–17
    "q", "r", "s", "t", "u", "v", "w", "x",  # 18–25
    "y", "z",                                  # 26–27
]

assert len(GRID_CHARS) == 28, "GRID_CHARS must contain exactly 28 tokens (blank + space + 26 letters)"

CTC_BLANK_IDX = 0

# ------------------------------------------------------------------
# Helper: indices → string
# ------------------------------------------------------------------

def indices_to_text(indices: list) -> str:
    """
    Maps a list of integer indices to a decoded string using GRID_CHARS.
    CTC blank tokens (index 0) are included — strip them if needed before passing.

    Args:
        indices: list of int

    Returns:
        str: concatenated character string
    """
    return "".join(GRID_CHARS[i] for i in indices if 0 <= i < len(GRID_CHARS))


# ------------------------------------------------------------------
# CTC greedy decoder
# ------------------------------------------------------------------

def ctc_greedy_decode(logits: torch.Tensor) -> list:
    """
    Greedy CTC decoder: argmax → collapse repeats → remove blanks → join to string.

    Args:
        logits: Tensor of shape (seq_len, batch, vocab_size=28)
                Values can be raw logits or log-softmax outputs.

    Returns:
        list[str]: one decoded string per batch item.
    """
    # (seq_len, batch, vocab_size) → (seq_len, batch)
    indices = logits.argmax(dim=-1)   # greedy argmax per timestep

    batch_size = indices.shape[1]
    decoded_sentences = []

    for b in range(batch_size):
        seq = indices[:, b].tolist()   # (seq_len,) for this batch item

        # 1. Collapse consecutive repeated tokens
        collapsed = []
        prev = None
        for token in seq:
            if token != prev:
                collapsed.append(token)
            prev = token

        # 2. Remove CTC blank tokens (index 0)
        tokens = [t for t in collapsed if t != CTC_BLANK_IDX]

        # 3. Convert indices → string
        text = indices_to_text(tokens)
        decoded_sentences.append(text)

    return decoded_sentences
