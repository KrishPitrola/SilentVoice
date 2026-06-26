"""
app.py — SilentVoice Phase 2: Auto-segmenting lip-reading backend
==================================================================

Pipeline (new):
    Browser canvas  →  JPEG frames over WebSocket  →  LipMotionSegmenter
    → utterance end detected → save temp .mp4
    → clip_queue
    → VSR worker  (AutoAVSRWrapper.transcribe)    — threadpool
    → nlp_queue
    → NLP worker  (NLPCorrector.correct)          — threadpool
    → tts_queue
    → TTS worker  (TTSEngine.speak_to_file)       — threadpool
    → JSON text + raw MP3 bytes → WebSocket → browser

Protocol (WebSocket /ws):
    CLIENT → SERVER : raw JPEG bytes per frame  (binary messages)
    CLIENT → SERVER : JSON text {"type":"control","action":"start"|"stop"}
    SERVER → CLIENT : JSON text {"type":"status","speaking":bool,"score":float}
    SERVER → CLIENT : JSON text {"type":"result","raw":"...","corrected":"...","status":"done"}
    SERVER → CLIENT : binary MP3 audio bytes

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000
    or: python app.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from auto_avsr_wrapper import AutoAVSRWrapper
from lip_segmenter import LipMotionSegmenter, frames_to_mp4
from nlp_corrector import NLPCorrector
from tts_engine import TTSEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("app")

# ──────────────────────────────────────────────────────────────────
# App + global state
# ──────────────────────────────────────────────────────────────────
app = FastAPI(title="SilentVoice API", version="3.0.0")

# Models populated at startup
vsr: AutoAVSRWrapper | None = None
nlp: NLPCorrector   | None = None
tts: TTSEngine      | None = None

# Each active WebSocket gets its own set of queues, keyed by id(ws)
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

    logger.info("[startup] Loading AutoAVSRWrapper …")
    vsr = await loop.run_in_executor(None, AutoAVSRWrapper)
    logger.info("[startup] AutoAVSRWrapper ready.")

    logger.info("[startup] Loading NLPCorrector …")
    nlp = await loop.run_in_executor(None, NLPCorrector)
    logger.info("[startup] NLPCorrector ready.")

    logger.info("[startup] Loading TTSEngine …")
    tts = await loop.run_in_executor(None, TTSEngine)
    logger.info("[startup] TTSEngine ready.")

    logger.info("[startup] All models loaded — server is ready.\n")


# ──────────────────────────────────────────────────────────────────
# Worker coroutines (one set per client, unchanged from Phase 1)
# ──────────────────────────────────────────────────────────────────
async def vsr_worker(
    clip_queue: asyncio.Queue,
    nlp_queue:  asyncio.Queue,
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
            logger.info("[VSR] %r", raw_text)
            await nlp_queue.put((raw_text, video_path))
        except Exception as exc:
            logger.error("[VSR] Error: %s", exc)
            _remove(video_path)
            try:
                await ws.send_text(
                    json.dumps({"type": "result", "status": "error", "message": str(exc)})
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
            logger.info("[NLP] %r", corrected)
            await tts_queue.put((raw_text, corrected, video_path))
        except Exception as exc:
            logger.error("[NLP] Error: %s", exc)
            _remove(video_path)
            try:
                await ws.send_text(
                    json.dumps({"type": "result", "status": "error", "message": str(exc)})
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
            logger.info("[TTS] Audio written to %r", mp3_path)

            audio_bytes = Path(mp3_path).read_bytes()

            # 1️⃣  JSON metadata
            payload = json.dumps(
                {"type": "result", "raw": raw_text, "corrected": corrected, "status": "done"}
            )
            await ws.send_text(payload)

            # 2️⃣  Raw MP3 bytes
            await ws.send_bytes(audio_bytes)

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error("[TTS] Error: %s", exc)
            try:
                await ws.send_text(
                    json.dumps({"type": "result", "status": "error", "message": str(exc)})
                )
            except Exception:
                pass
        finally:
            _remove(video_path)
            if mp3_path:
                _remove(mp3_path)
            tts_queue.task_done()


# ──────────────────────────────────────────────────────────────────
# WebSocket endpoint — Phase 2: continuous JPEG frame streaming
# ──────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """
    Accept a WebSocket connection.

    Protocol (CLIENT → SERVER):
        binary  : raw JPEG bytes for one canvas frame
        text    : JSON control message
                  {"type":"control","action":"start"}  — begin accumulating frames
                  {"type":"control","action":"stop"}   — stop and flush if speech buffered

    Protocol (SERVER → CLIENT):
        text    : {"type":"status","speaking":bool,"score":float}
                  (sent ~every 5 frames so the UI can show lip activity)
        text    : {"type":"result","raw":"…","corrected":"…","status":"done"}
        binary  : MP3 audio bytes (follows the "done" result message)
        text    : {"type":"result","status":"error","message":"…"}
    """
    await ws.accept()
    client_id = id(ws)
    logger.info("[WS] Client %d connected.", client_id)

    # Per-client pipeline queues (unchanged)
    clip_queue: asyncio.Queue = asyncio.Queue()
    nlp_queue:  asyncio.Queue = asyncio.Queue()
    tts_queue:  asyncio.Queue = asyncio.Queue()
    _client_queues[client_id] = (clip_queue, nlp_queue, tts_queue)

    # Start pipeline worker tasks
    tasks = [
        asyncio.create_task(vsr_worker(clip_queue, nlp_queue, ws)),
        asyncio.create_task(nlp_worker(nlp_queue, tts_queue, ws)),
        asyncio.create_task(tts_worker(tts_queue, ws)),
    ]

    # Per-client segmenter (created fresh; destroyed on disconnect)
    segmenter = LipMotionSegmenter(fps=20)
    streaming  = False   # True while START has been sent and no STOP received
    frame_seq  = 0       # frame counter for throttling status messages
    STATUS_EVERY = 5     # send lip status JSON every N frames

    try:
        while True:
            message = await ws.receive()

            # ── Control text message ───────────────────────────────
            if "text" in message:
                try:
                    ctrl = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                if ctrl.get("type") != "control":
                    continue

                action = ctrl.get("action", "")

                if action == "start":
                    streaming = True
                    segmenter.reset()
                    frame_seq = 0
                    logger.info("[WS] Client %d: streaming started.", client_id)

                elif action == "stop":
                    streaming = False
                    logger.info("[WS] Client %d: streaming stopped.", client_id)
                    # Force-flush whatever is buffered (even without silence timeout)
                    # Only if there are frames worth sending
                    if segmenter._speech_active and len(segmenter._frame_buffer) >= segmenter.min_speech_frames:
                        frames_np, n = segmenter._flush()
                        await _save_and_enqueue(frames_np, n, clip_queue, client_id)
                    segmenter.reset()

            # ── Binary JPEG frame ──────────────────────────────────
            elif "bytes" in message:
                if not streaming:
                    continue  # ignore frames when not started

                jpeg_bytes = message["bytes"]
                if not jpeg_bytes:
                    continue

                frame_seq += 1

                # Push frame to segmenter
                result = segmenter.push_frame(jpeg_bytes)

                # Send lip motion status to frontend (throttled)
                if frame_seq % STATUS_EVERY == 0:
                    try:
                        await ws.send_text(json.dumps({
                            "type":     "status",
                            "speaking": segmenter.is_speaking,
                            "score":    round(segmenter.last_score, 5),
                        }))
                    except Exception:
                        pass

                # Utterance complete!
                if result is not None:
                    frames_np, n = result
                    await _save_and_enqueue(frames_np, n, clip_queue, client_id)
                    # segmenter already reset its internal state after _flush();
                    # call public reset() to reinitialise for next utterance
                    segmenter.reset()
                    logger.info("[WS] Client %d: listening for next utterance.", client_id)

    except WebSocketDisconnect:
        logger.info("[WS] Client %d disconnected.", client_id)
    except Exception as exc:
        logger.error("[WS] Unexpected error for client %d: %s", client_id, exc)
    finally:
        segmenter.close()

        # Drain the pipeline gracefully
        await clip_queue.put(None)
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            for t in tasks:
                t.cancel()

        _client_queues.pop(client_id, None)
        logger.info("[WS] Pipeline for client %d shut down.", client_id)


# ──────────────────────────────────────────────────────────────────
# Helper: save frame array to a temp mp4 and push to clip_queue
# ──────────────────────────────────────────────────────────────────
async def _save_and_enqueue(
    frames_np: "np.ndarray",  # type: ignore[name-defined]  # noqa: F821
    n_frames: int,
    clip_queue: asyncio.Queue,
    client_id: int,
) -> None:
    """
    Save numpy frames to a temp .mp4 file in an executor thread
    (VideoWriter is blocking), then push the path to clip_queue.
    """
    loop = asyncio.get_event_loop()

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    try:
        await loop.run_in_executor(
            None,
            lambda: frames_to_mp4(frames_np, tmp_path, fps=20),
        )
        logger.info(
            "[WS] Client %d: utterance clip saved (%d frames) → %s",
            client_id, n_frames, tmp_path,
        )
        await clip_queue.put((tmp_path, tmp_path))
    except Exception as exc:
        logger.error("[WS] Client %d: failed to save clip: %s", client_id, exc)
        _remove(tmp_path)


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
