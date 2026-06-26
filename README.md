# SilentVoice

**Real-Time Visual Speech Recognition and Context-Aware Speech Synthesis System**

SilentVoice converts silent lip movements into audible, natural speech in real time — no microphone, no vocal cords required. Built for people who cannot speak aloud due to ALS, laryngectomy, stroke-related speech loss, or vocal cord damage, where typing is too slow and electrolarynx devices sound unnatural.

> Webcam → Lip Reading AI → Text → AI Correction → Audio Output

---

## Problem Statement

Existing assistive communication tools fall short:

| Method | Limitation |
|---|---|
| Typing / AAC boards | Slow, requires fine motor control |
| Electrolarynx | Unnatural sound, requires surgery/device |
| Eye-tracking systems | Expensive, fatiguing for long use |

**SilentVoice** offers a hands-free, camera-only alternative: speak silently by moving your lips, and the system reads, corrects, and speaks for you — in seconds, using just a webcam.

---

## How It Works

```
Webcam Feed
    ↓
Face Detection (MediaPipe FaceLandmarker)
    ↓
Lip Region Tracking + Motion-Based Segmentation
    ↓
Visual Speech Recognition (Auto-AVSR)
    ↓
Raw Transcript  →  e.g. "i wnt to go hm"
    ↓
NLP Correction (Groq llama-3.1-8b-instant)
    ↓
Corrected Sentence  →  "I want to go home"
    ↓
Text-to-Speech (edge-tts)
    ↓
Audio Output
```

The system buffers frames while it detects active lip movement, and automatically cuts the clip for transcription once it detects a natural pause — the visual equivalent of Voice Activity Detection (VAD), built from scratch on MediaPipe lip landmark motion.

---

## Architecture

- **Backend:** FastAPI + WebSocket, with `asyncio.Queue`-based pipeline decoupling
- **Concurrency model:** Three independent async workers (VSR → NLP → TTS) process overlapping speech segments concurrently rather than blocking end-to-end
- **Frontend:** Vanilla JS, live webcam streaming over WebSocket (no MediaRecorder — raw frame streaming for lower latency)
- **Models loaded once at startup** (not per-request) to avoid reload latency in the live pipeline

---

## Tech Stack

| Layer | Tool |
|---|---|
| Computer Vision | OpenCV, MediaPipe (FaceLandmarker) |
| Visual Speech Recognition | Open-vocabulary VSR model (trained on LRS3 + VoxCeleb2 corpora) |
| NLP Correction | Groq API — `llama-3.1-8b-instant` |
| Text-to-Speech | `edge-tts` |
| Backend | FastAPI + WebSocket + `asyncio.Queue` |
| Frontend | Vanilla JavaScript |
| Hardware (dev) | GTX 1650, CUDA 12.9 |

---

## Project Status — Semester 7

This is an active two-semester capstone project. Semester 1 delivered a complete working pipeline using LipNet on the GRID corpus (51-word vocabulary, preserved on `main`). Semester 7 is a full upgrade to open-vocabulary, natural-speech recognition.

**Completed:**
- ✅ Migrated VSR model: LipNet (GRID, 51 words) → Auto-AVSR (LRS3+VoxCeleb2, open vocabulary)
- ✅ Full offline pipeline verified: video → VSR → NLP correction → TTS audio
- ✅ Live FastAPI + WebSocket app: webcam → real-time transcript → corrected text → spoken audio
- ✅ Lip-motion-based auto-segmentation (visual VAD) replacing fixed-time chunking
- ✅ End-to-end test: *"I am going to show you"* transcribed, corrected, and spoken back

**In progress:**
- ⚠️ Auto-segmentation threshold tuning (distinguishing real speech motion from breathing/lighting noise)
- ⏳ Quantitative evaluation — WER/CER and latency benchmarking
- ⏳ Path/portability cleanup for reproducibility

**Known limitations (documented, not hidden):**
- Not frame-token-streaming real-time — current latency is ~4–8 sec per utterance, since Auto-AVSR processes complete clips rather than streaming token-by-token. A true real-time system would need an RNN-T or streaming conformer architecture (noted as future work).
- Tested on a single speaker; generalization to atypical speech patterns (e.g. reduced lip mobility in ALS patients) is untested.
- Runs on local `ws://`, not `wss://` — fine for localhost demo, would need HTTPS/WSS for any real deployment.

---

## Setup

```bash
git clone https://github.com/KrishPitrola/SilentVoice.git
cd SilentVoice
git checkout sem7-integration

pip install -r requirements.txt
```

Model weights are gitignored — download Auto-AVSR pretrained checkpoint separately and place per path config in `auto_avsr_wrapper.py`.

Set your Groq API key as an environment variable:
```bash
export GROQ_API_KEY=your_key_here
```

Run the server:
```bash
uvicorn app:app --reload
```

Open `http://localhost:8000` in your browser, allow camera access, and start speaking silently.

---

## Repository Structure

```
SilentVoice/
├── app.py                  # FastAPI server, WebSocket handling, pipeline orchestration
├── lip_segmenter.py         # Visual VAD — lip motion-based utterance segmentation
├── auto_avsr_wrapper.py     # Auto-AVSR model wrapper for VSR inference
├── nlp_corrector.py         # Groq-based grammar/context correction
├── tts_engine.py            # edge-tts audio generation
├── static/
│   └── index.html          # Frontend — webcam capture, WebSocket client, audio playback
├── evaluate.py              # WER/CER + latency benchmarking script
└── weights/                 # (gitignored) model checkpoints
```

---

## Evaluation

Evaluation methodology: 20–30 recorded test sentences with ground-truth transcripts, measuring Word Error Rate (WER) and Character Error Rate (CER) before and after NLP correction, plus per-stage latency (capture → VSR → NLP → TTS).

*(Results to be added once benchmarking is complete.)*

---

## Future Work

- Replace fixed-clip VSR with streaming conformer / RNN-T for true token-level real-time inference
- Fine-tune Auto-AVSR on atypical speech patterns (ALS, post-stroke) for clinical generalization
- Redis Streams for inter-stage communication if scaling to a multi-user deployed service (evaluated and intentionally deferred for single-user local scope)
- HTTPS/WSS deployment for use outside localhost

---

## Team

| Members | 
| Krish Pitrola |
| Atharva Palande|
| Akshata Bhavsar|

---

## License
Academic project 