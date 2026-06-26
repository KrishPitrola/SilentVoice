"""
lip_segmenter.py — SilentVoice auto-segmentation module
=========================================================
Receives raw JPEG frames one at a time, detects lip motion using
MediaPipe Face Mesh (Tasks API — mp.tasks.vision), and automatically
segments utterances when lip movement drops below a threshold for N
consecutive frames.

MediaPipe Tasks API is used exclusively (mp.tasks.vision).
No mp.solutions is referenced anywhere.

Landmark indices used (10 points that capture mouth open/close well):
    Upper lip : 13, 14, 312, 311, 310
    Lower lip : 17, 84, 181,  91, 146

Usage
-----
    seg = LipMotionSegmenter(fps=15)
    result = seg.push_frame(jpeg_bytes)   # call once per frame
    if result is not None:
        frames_np, n_frames = result      # numpy array (N, H, W, 3), uint8 BGR
        # ... save to mp4, push to pipeline ...
        seg.reset()                       # ready for next utterance
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ── MediaPipe Tasks imports ────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

logger = logging.getLogger("lip_segmenter")

# ──────────────────────────────────────────────────────────────────
# Landmark indices
# ──────────────────────────────────────────────────────────────────
UPPER_LIP_IDX = [13, 14, 312, 311, 310]
LOWER_LIP_IDX = [17, 84, 181, 91, 146]
LIP_INDICES   = UPPER_LIP_IDX + LOWER_LIP_IDX  # 10 points total


# ──────────────────────────────────────────────────────────────────
# Helper: locate the bundled FaceLandmarker model
# ──────────────────────────────────────────────────────────────────
def _find_model_asset() -> str:
    """
    Return path to face_landmarker.task.

    Search order:
      1. $MEDIAPIPE_FACE_MODEL env var (explicit override)
      2. data/face_landmarker.task  (project data dir)
      3. The mediapipe package data directory (bundled copy)

    Raises FileNotFoundError if not found anywhere.
    """
    if env := os.environ.get("MEDIAPIPE_FACE_MODEL"):
        p = Path(env)
        if p.is_file():
            return str(p)

    local = Path(__file__).parent / "data" / "face_landmarker.task"
    if local.is_file():
        return str(local)

    # mediapipe >= 0.10 ships model bundles alongside the package
    pkg_root = Path(mp.__file__).parent
    for candidate in pkg_root.rglob("face_landmarker*.task"):
        return str(candidate)

    raise FileNotFoundError(
        "face_landmarker.task not found.\n"
        "Download it from:\n"
        "  https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task\n"
        "and place it at:  data/face_landmarker.task\n"
        "or set MEDIAPIPE_FACE_MODEL=/absolute/path/to/face_landmarker.task"
    )


# ──────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────
class LipMotionSegmenter:
    """
    Auto-segments lip-reading utterances from a continuous JPEG frame stream.

    Parameters
    ----------
    fps : int
        Expected frames-per-second from the browser (used for min_speech_frames
        and logging only; timing is frame-count-based, not wall-clock).
    movement_threshold : float
        Per-frame lip movement score (mean pixel-distance of landmark
        positions vs. previous frame, normalised to [0,1] by image width).
        Score < threshold → frame is "silent".
        Default: 0.003  (~0.3 % of frame width movement).
    silence_frames : int
        Number of consecutive "silent" frames required before an utterance
        is declared complete and the buffer flushed.
        Default: 20  (≈ 1.0 s at 20 fps — tolerates brief mid-sentence pauses).
    min_speech_frames : int
        Minimum buffered frames before a flush is allowed.
        Prevents very short noise bursts from being sent to VSR.
        Default: 15  (≈ 0.75 s at 20 fps).
    max_buffer_frames : int
        Hard cap on buffer size.  If reached, flush immediately regardless
        of silence state (prevents memory runaway on very long speech).
        Default: 600  (30 s at 20 fps).
    model_path : str | None
        Explicit path to face_landmarker.task.  If None, auto-detected.
    """

    def __init__(
        self,
        fps: int = 20,
        movement_threshold: float = 0.003,
        silence_frames: int = 20,
        min_speech_frames: int = 15,
        max_buffer_frames: int = 600,
        model_path: str | None = None,
    ) -> None:
        self.fps               = fps
        self.threshold         = movement_threshold
        self.silence_frames    = silence_frames
        self.min_speech_frames = min_speech_frames
        self.max_buffer_frames = max_buffer_frames

        # Resolve model
        resolved = model_path or _find_model_asset()
        logger.info("[LipSegmenter] Loading face landmarker from %s", resolved)

        # Build FaceLandmarker in IMAGE mode (synchronous, one frame at a time)
        base_opts = BaseOptions(model_asset_path=resolved)
        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            # min_face_presence_score=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(face_opts)
        logger.info("[LipSegmenter] FaceLandmarker ready.")

        # Runtime state
        self._frame_buffer: list[np.ndarray] = []  # BGR frames (H, W, 3)
        self._prev_lip_pts: np.ndarray | None = None  # shape (10, 2)
        self._silent_run: int = 0     # consecutive silent frames
        self._speech_active: bool = False
        self._frame_idx: int = 0      # global frame counter (for debugging)

        # Public observable state (updated each push_frame call)
        self.last_score: float = 0.0
        self.is_speaking: bool = False

    # ── Public API ─────────────────────────────────────────────────

    def push_frame(
        self, jpeg_bytes: bytes
    ) -> Optional[tuple[np.ndarray, int]]:
        """
        Process one JPEG frame.

        Parameters
        ----------
        jpeg_bytes : bytes
            Raw JPEG bytes from the browser canvas.

        Returns
        -------
        None
            Utterance not yet complete — keep streaming.
        (frames_np, n_frames) : tuple[np.ndarray, int]
            Utterance complete.  frames_np is shape (N, H, W, 3) uint8 BGR.
            After receiving this, call reset() to prepare for the next utterance.
        """
        self._frame_idx += 1

        # Decode JPEG → BGR numpy array
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("[LipSegmenter] Frame %d: JPEG decode failed — skipped.", self._frame_idx)
            return None

        h, w = bgr.shape[:2]

        # Extract lip landmarks via MediaPipe Tasks API
        lip_pts = self._extract_lip_points(bgr, w, h)

        # Compute movement score
        score = self._compute_score(lip_pts)
        self.last_score = score

        is_moving = (lip_pts is not None) and (score >= self.threshold)
        self.is_speaking = is_moving

        # Update previous landmark positions
        if lip_pts is not None:
            self._prev_lip_pts = lip_pts

        # ── State machine ──────────────────────────────────────────
        if is_moving:
            # Speech frame — buffer it, reset silence counter
            self._frame_buffer.append(bgr)
            self._silent_run = 0
            self._speech_active = True
        else:
            # Silent frame
            if self._speech_active:
                # Still buffer silent frames (they're part of trailing context)
                self._frame_buffer.append(bgr)
                self._silent_run += 1

                # Check flush conditions
                over_silence = self._silent_run >= self.silence_frames
                over_max     = len(self._frame_buffer) >= self.max_buffer_frames

                if over_max:
                    logger.info(
                        "[LipSegmenter] Frame %d: max buffer (%d) hit — force flush.",
                        self._frame_idx, self.max_buffer_frames,
                    )
                    return self._flush()

                if over_silence:
                    # Trim trailing silence frames (keep only half of silence_frames
                    # so VSR still sees natural end-of-utterance lip position)
                    n_trim = self.silence_frames // 2
                    n = len(self._frame_buffer)
                    keep = n - n_trim
                    if keep >= self.min_speech_frames:
                        logger.info(
                            "[LipSegmenter] Frame %d: utterance end detected "
                            "(silence=%d frames, buffer=%d frames).",
                            self._frame_idx, self._silent_run, n,
                        )
                        self._frame_buffer = self._frame_buffer[:keep]
                        return self._flush()
                    else:
                        # Buffer too short — likely a noise burst; discard
                        logger.debug(
                            "[LipSegmenter] Frame %d: buffer too short (%d < %d) — discarding.",
                            self._frame_idx, keep, self.min_speech_frames,
                        )
                        self._reset_state()
            # else: still silent before any speech → do nothing

        return None

    def reset(self) -> None:
        """
        Call after receiving a non-None result from push_frame to prepare
        for the next utterance.  Clears the frame buffer and silence counter
        but keeps the MediaPipe landmarker alive.
        """
        self._reset_state()
        logger.debug("[LipSegmenter] Segmenter reset — listening for next utterance.")

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        logger.info("[LipSegmenter] Closed.")

    # ── Internal helpers ───────────────────────────────────────────

    def _extract_lip_points(
        self, bgr: np.ndarray, w: int, h: int
    ) -> np.ndarray | None:
        """
        Run FaceLandmarker on one BGR frame; return (10, 2) float32 array of
        normalised (x, y) coordinates for LIP_INDICES, or None if no face found.
        """
        # MediaPipe Tasks expects RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None  # no face detected

        landmarks = result.face_landmarks[0]  # first face

        pts = np.array(
            [[landmarks[i].x, landmarks[i].y] for i in LIP_INDICES],
            dtype=np.float32,
        )
        return pts  # normalised [0,1] coords

    def _compute_score(self, lip_pts: np.ndarray | None) -> float:
        """
        Mean Euclidean distance of lip landmarks vs. previous frame,
        normalised so that 1.0 = full image width of movement.

        Returns 0.0 if no face was detected or no previous frame exists.
        """
        if lip_pts is None or self._prev_lip_pts is None:
            return 0.0
        delta = lip_pts - self._prev_lip_pts       # (10, 2)
        dist  = np.linalg.norm(delta, axis=1)      # (10,)
        return float(dist.mean())

    def _flush(self) -> tuple[np.ndarray, int]:
        """Stack buffered frames into (N, H, W, 3) array and return."""
        frames = np.stack(self._frame_buffer, axis=0)  # (N, H, W, 3)
        n = len(self._frame_buffer)
        self._reset_state()
        return frames, n

    def _reset_state(self) -> None:
        self._frame_buffer = []
        self._prev_lip_pts = None
        self._silent_run   = 0
        self._speech_active = False
        self.is_speaking    = False
        self.last_score     = 0.0


# ──────────────────────────────────────────────────────────────────
# Utility: save numpy frame array to an mp4 file
# ──────────────────────────────────────────────────────────────────
def frames_to_mp4(
    frames: np.ndarray,
    out_path: str,
    fps: int = 20,
) -> None:
    """
    Write (N, H, W, 3) BGR uint8 array to an mp4 file using cv2.VideoWriter.

    Parameters
    ----------
    frames   : np.ndarray  shape (N, H, W, 3), dtype uint8, BGR
    out_path : str         destination file path (should end with .mp4)
    fps      : int         frame rate to embed in the file header
    """
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"Expected (N, H, W, 3) array, got {frames.shape}")

    n, h, w, _ = frames.shape

    # mp4v is universally available; H.264 requires an openh264 build
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {out_path!r}")

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    logger.info("[frames_to_mp4] Wrote %d frames → %s", n, out_path)
