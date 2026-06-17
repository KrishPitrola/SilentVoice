"""
demo.py — Real-time LipNet inference script.

Pipeline:
  Webcam → LipExtractor → FrameBuffer (30 frames) → SequencePreprocessor
  → VisualSpeechRecognitionModel → ctc_greedy_decode → FallbackMapper → print
"""

import cv2
import torch
import threading
from datetime import datetime

from modules.lip_extractor import LipExtractor
from modules.frame_buffer import FrameBuffer
from modules.sequence_preprocessor import SequencePreprocessor
from models.vsr_model import VisualSpeechRecognitionModel
from vocab import ctc_greedy_decode
from fallback_mapper import FallbackMapper
from tts_engine import TTSEngine


def main():
    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    print("Loading LipNet model...")
    model = VisualSpeechRecognitionModel(vocab_size=28, lipnet_mode=True)
    model.load_pretrained("weights/lipnet_overlap.pt")
    model.eval()
    print("Model ready.\n")

    # ------------------------------------------------------------------
    # Module initialisation
    # ------------------------------------------------------------------
    lip_extractor  = LipExtractor()
    frame_buffer   = FrameBuffer()
    preprocessor   = SequencePreprocessor()
    mapper         = FallbackMapper()
    tts            = TTSEngine()

    # ------------------------------------------------------------------
    # Open webcam
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print("Running — press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame.")
            break

        # Extract lip crop from frame (also draws green bounding box)
        lip_crop = lip_extractor.extract_lips(frame)

        if lip_crop is not None:
            # Add resized (128x64) crop to the rolling buffer
            frame_buffer.add_frame(lip_crop)

        # Run inference when buffer holds 30 complete frames
        if frame_buffer.is_full():
            sequence = frame_buffer.get_sequence()           # (30, 64, 128, 3)
            tensor   = preprocessor.preprocess(sequence)     # (1, 3, 30, 64, 128)

            with torch.no_grad():
                logits = model(tensor)                       # (30, 1, 28)

            # CTC greedy decode — returns list[str], one per batch item
            decoded = ctc_greedy_decode(logits)
            text = decoded[0] if decoded else ""

            # Filter garbage and map to human-readable phrase
            if text and not mapper.is_garbage(text):
                phrase    = mapper.map(text)
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Raw: '{text}'  ->  '{phrase}'")
                
                # Speak phrase asynchronously to avoid freezing the camera feed
                threading.Thread(target=tts.speak, args=(phrase,), daemon=True).start()

            # Clear buffer so next window starts fresh
            frame_buffer.buffer.clear()

        # Show live camera feed with lip bounding box overlay
        cv2.imshow("SilentVoice - Lip Reading", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
