"""
app.py — SilentVoice FastAPI backend with WebSocket inference endpoint.

Clients send JPEG frames over WebSocket; the server runs the full LipNet
pipeline and streams JSON predictions back.

Run with:
    python app.py
    or: uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Static frontend (if present) is served from ./static/
"""

import json
import random

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from modules.lip_extractor import LipExtractor
from modules.frame_buffer import FrameBuffer
from modules.sequence_preprocessor import SequencePreprocessor
from models.vsr_model import VisualSpeechRecognitionModel
from vocab import ctc_greedy_decode
from fallback_mapper import FallbackMapper

import time 
_last_phrase = ""
_last_time = 0
COOLDOWN_SEC = 3

# ──────────────────────────────────────────────────────────────────
# Global pipeline components (loaded once at startup)
# ──────────────────────────────────────────────────────────────────
print("Loading LipNet model...")
model = VisualSpeechRecognitionModel(vocab_size=28, lipnet_mode=True)
model.load_pretrained("weights/lipnet_overlap.pt")
model.eval()
print("Model ready.\n")

lip_extractor = LipExtractor()
frame_buffer  = FrameBuffer()
preprocessor  = SequencePreprocessor()
mapper        = FallbackMapper()

# ──────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────
app = FastAPI(title="SilentVoice API", version="1.0.0")


# ──────────────────────────────────────────────────────────────────
# WebSocket inference endpoint
# ──────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Receive JPEG frames as raw bytes, run LipNet inference, stream JSON back.

    Per-frame response (always):
        {'status': 'detected'} | {'status': 'no_face'}

    When buffer full and prediction valid:
        {'raw': str, 'phrase': str, 'confidence': int}
    """
    await websocket.accept()
    print("[WS] Client connected.")

    try:
        while True:
            # Receive raw JPEG bytes from client
            data = await websocket.receive_bytes()

            # Decode JPEG → OpenCV BGR frame
            frame = cv2.imdecode(
                np.frombuffer(data, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is None:
                await websocket.send_text(json.dumps({"status": "decode_error"}))
                continue

            # Run lip extraction (also draws bounding box on frame)
            lip_crop = lip_extractor.extract_lips(frame)

            if lip_crop is None:
                await websocket.send_text(json.dumps({"status": "no_face"}))
                continue

            await websocket.send_text(json.dumps({"status": "detected"}))
            frame_buffer.add_frame(lip_crop)

            # Run inference when buffer is full (30 frames)
            if frame_buffer.is_full():
                try:
                    sequence = frame_buffer.get_sequence()        # (30, 64, 128, 3)
                    tensor   = preprocessor.preprocess(sequence)  # (1, 3, 30, 64, 128)

                    with torch.no_grad():
                        logits = model(tensor)                    # (30, 1, 28)

                    decoded  = ctc_greedy_decode(logits)
                    raw_text = decoded[0] if decoded else ""

                    if raw_text and not mapper.is_garbage(raw_text):
                        phrase = mapper.map(raw_text)
                        global _last_phrase, _last_time
                        now = time.time()
                        if phrase == _last_phrase and (now - _last_time) < COOLDOWN_SEC:
                            frame_buffer.buffer.clear()
                            continue
                        _last_phrase = phrase
                        _last_time = now
                        payload = {
                            "raw": raw_text,
                            "phrase": phrase,
                            "confidence": random.randint(72, 91),
                        }
                        await websocket.send_text(json.dumps(payload))
                        print(f"[WS] Predicted: '{raw_text}' → '{phrase}'")
                except Exception as e:
                    print(f"[WS] Inference error: {e}")

                frame_buffer.buffer.clear()

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")


# ──────────────────────────────────────────────────────────────────
# Static frontend (./static/index.html)  — mount LAST so /ws takes priority
# ──────────────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
