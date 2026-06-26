"""
app.py — SilentVoice Phase 1: FastAPI WebSocket backend
        with async pipeline (asyncio.Queue) for clip-based lip-reading.

Pipeline:
    Browser MediaRecorder (.webm blob)
    ↓ WebSocket (binary)
    clip_queue
    ↓ VSR worker  (MPC001VSR.transcribe)      — threadpool
    nlp_queue
    ↓ NLP worker  (NLPCorrector.correct)      — threadpool
    tts_queue
    ↓ TTS worker  (TTSEngine.speak_to_file)   — threadpool
    ↓ JSON text + raw MP3 bytes → WebSocket

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000
    or: python app.py
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from auto_avsr_wrapper import AutoAVSRWrapper
from nlp_corrector import NLPCorrector
from tts_engine import TTSEngine

# ──────────────────────────────────────────────────────────────────
# App + global state
# ──────────────────────────────────────────────────────────────────
app = FastAPI(title="SilentVoice API", version="2.0.0")

# Models are populated at startup
vsr: AutoAVSRWrapper | None = None
nlp: NLPCorrector | None = None
tts: TTSEngine | None = None

# Each active WebSocket gets its own set of three queues so multiple
# clients can be served independently.  The queues are keyed by the
# WebSocket object itself.
ClientQueues = tuple[asyncio.Queue, asyncio.Queue, asyncio.Queue]
_client_queues: dict[int, ClientQueues] = {}


# ──────────────────────────────────────────────────────────────────
# Startup / shutdown
# ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def load_models() -> None:
    """Load all three blocking models once, in executor threads."""
    global vsr, nlp, tts
    loop = asyncio.get_event_loop()

    print("[startup] Loading AutoAVSRWrapper …", flush=True)
    vsr = await loop.run_in_executor(None, AutoAVSRWrapper)
    print("[startup] AutoAVSRWrapper ready.", flush=True)

    print("[startup] Loading NLPCorrector …", flush=True)
    nlp = await loop.run_in_executor(None, NLPCorrector)
    print("[startup] NLPCorrector ready.", flush=True)

    print("[startup] Loading TTSEngine …", flush=True)
    tts = await loop.run_in_executor(None, TTSEngine)
    print("[startup] TTSEngine ready.", flush=True)

    print("[startup] All models loaded — server is ready.\n", flush=True)


# ──────────────────────────────────────────────────────────────────
# Worker coroutines (one set per client)
# ──────────────────────────────────────────────────────────────────
async def vsr_worker(
    clip_queue: asyncio.Queue,
    nlp_queue: asyncio.Queue,
    ws: WebSocket,
) -> None:
    """Pull video paths from clip_queue, run VSR in executor, push to nlp_queue."""
    loop = asyncio.get_event_loop()
    while True:
        item = await clip_queue.get()
        if item is None:          # sentinel → shut down
            await nlp_queue.put(None)
            clip_queue.task_done()
            return

        video_path, raw_name = item
        try:
            raw_text: str = await loop.run_in_executor(
                None, vsr.transcribe, video_path
            )
            print(f"[VSR] {raw_text!r}", flush=True)
            await nlp_queue.put((raw_text, video_path))
        except Exception as exc:
            print(f"[VSR] Error: {exc}", flush=True)
            # Skip this clip — clean up temp file and send error to client
            _remove(video_path)
            try:
                await ws.send_text(
                    json.dumps({"status": "error", "message": str(exc)})
                )
            except Exception:
                pass
        finally:
            clip_queue.task_done()


async def nlp_worker(
    nlp_queue: asyncio.Queue,
    tts_queue: asyncio.Queue,
    ws: WebSocket,
) -> None:
    """Pull (raw_text, video_path) from nlp_queue, correct, push to tts_queue."""
    loop = asyncio.get_event_loop()
    while True:
        item = await nlp_queue.get()
        if item is None:
            await tts_queue.put(None)
            nlp_queue.task_done()
            return

        raw_text, video_path = item
        try:
            corrected: str = await loop.run_in_executor(
                None, nlp.correct, raw_text
            )
            print(f"[NLP] {corrected!r}", flush=True)
            await tts_queue.put((raw_text, corrected, video_path))
        except Exception as exc:
            print(f"[NLP] Error: {exc}", flush=True)
            _remove(video_path)
            try:
                await ws.send_text(
                    json.dumps({"status": "error", "message": str(exc)})
                )
            except Exception:
                pass
        finally:
            nlp_queue.task_done()


async def tts_worker(
    tts_queue: asyncio.Queue,
    ws: WebSocket,
) -> None:
    """Pull (raw, corrected, video_path), synthesise speech, send results to ws."""
    loop = asyncio.get_event_loop()
    while True:
        item = await tts_queue.get()
        if item is None:
            tts_queue.task_done()
            return

        raw_text, corrected, video_path = item
        mp3_path: str | None = None
        try:
            mp3_path = await loop.run_in_executor(
                None, tts.speak_to_file, corrected
            )
            print(f"[TTS] Audio written to {mp3_path!r}", flush=True)

            # Read audio bytes before sending
            audio_bytes = Path(mp3_path).read_bytes()

            # 1️⃣  JSON metadata
            payload = json.dumps(
                {"raw": raw_text, "corrected": corrected, "status": "done"}
            )
            await ws.send_text(payload)

            # 2️⃣  Raw MP3 bytes
            await ws.send_bytes(audio_bytes)

        except WebSocketDisconnect:
            # Client left — just clean up and stop
            pass
        except Exception as exc:
            print(f"[TTS] Error: {exc}", flush=True)
            try:
                await ws.send_text(
                    json.dumps({"status": "error", "message": str(exc)})
                )
            except Exception:
                pass
        finally:
            _remove(video_path)
            if mp3_path:
                _remove(mp3_path)
            tts_queue.task_done()


# ──────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """
    Accept a WebSocket connection.

    Protocol:
        CLIENT → SERVER : raw .webm video bytes (one message per clip)
        SERVER → CLIENT : text  {"raw": "...", "corrected": "...", "status": "done"}
        SERVER → CLIENT : bytes (MP3 audio)
    """
    await ws.accept()
    client_id = id(ws)
    print(f"[WS] Client {client_id} connected.", flush=True)

    # Per-client queues
    clip_queue: asyncio.Queue = asyncio.Queue()
    nlp_queue:  asyncio.Queue = asyncio.Queue()
    tts_queue:  asyncio.Queue = asyncio.Queue()
    _client_queues[client_id] = (clip_queue, nlp_queue, tts_queue)

    # Start worker tasks
    tasks = [
        asyncio.create_task(vsr_worker(clip_queue, nlp_queue, ws)),
        asyncio.create_task(nlp_worker(nlp_queue, tts_queue, ws)),
        asyncio.create_task(tts_worker(tts_queue, ws)),
    ]

    try:
        while True:
            # Receive raw .webm bytes from the browser MediaRecorder
            data = await ws.receive_bytes()

            if not data:
                continue

            # Save to temp file with .mp4 extension (VSR accepts mp4/mpg)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            try:
                os.write(tmp_fd, data)
            finally:
                os.close(tmp_fd)

            print(
                f"[WS] Received clip {len(data):,} bytes → {tmp_path}", flush=True
            )
            await clip_queue.put((tmp_path, tmp_path))

    except WebSocketDisconnect:
        print(f"[WS] Client {client_id} disconnected.", flush=True)
    except Exception as exc:
        print(f"[WS] Unexpected error: {exc}", flush=True)
    finally:
        # Send sentinel to drain the pipeline gracefully
        await clip_queue.put(None)

        # Wait for all workers to finish processing in-flight clips
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            for t in tasks:
                t.cancel()

        _client_queues.pop(client_id, None)
        print(f"[WS] Pipeline for client {client_id} shut down.", flush=True)


# ──────────────────────────────────────────────────────────────────
# Static frontend — mounted LAST so /ws takes priority
# ──────────────────────────────────────────────────────────────────
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def _remove(path: str) -> None:
    """Silently delete a file."""
    try:
        os.remove(path)
    except OSError:
        pass


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
