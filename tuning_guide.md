# SilentVoice — Lip Segmenter Tuning Guide

## Default Parameter Values

| Parameter | Default | Role |
|---|---|---|
| `movement_threshold` | `0.003` | Mean per-landmark displacement (normalised) needed to call a frame "speech" |
| `silence_frames` | `20` | Consecutive silent frames required to declare utterance end (~1 s at 20 fps) |
| `min_speech_frames` | `15` | Minimum frames before a flush is allowed (~0.75 s at 20 fps) |
| `max_buffer_frames` | `600` | Hard cap on buffer size before forced flush (30 s at 20 fps) |
| `fps` | `20` | Expected frame rate from browser (15–20 is typical) |

---

## What the Score Means

Each frame produces a **lip movement score** — the mean Euclidean distance (in normalised image coordinates) of the 10 lip landmarks versus the previous frame.

- `0.000` → face not detected, or lips perfectly still  
- `0.001–0.003` → micro-movements: breathing, small twitches  
- `0.003–0.010` → clear lip movement (speech)  
- `> 0.010` → large open-close gestures

The `movement_threshold` sits at the boundary between the first and second ranges.

---

## How to Tune

### Threshold Too Low (too sensitive)
**Symptom**: System triggers on breathing, swallowing, or ambient head motion. Short spurious clips flood the pipeline.

**Fix**: Raise `movement_threshold`:
```python
# Conservative — only captures deliberate, visible lip movement
seg = LipMotionSegmenter(movement_threshold=0.005)
```

### Threshold Too High (misses speech)
**Symptom**: Long utterances are never detected. The server shows score values above threshold in its logs but no clips are sent.

**Fix**: Lower `movement_threshold`:
```python
# Sensitive — picks up subtle lip movement (e.g. mumbling)
seg = LipMotionSegmenter(movement_threshold=0.0015)
```

### Clips Cut Too Early (mid-sentence pauses trigger flush)
**Symptom**: A single sentence gets split into two clips at every natural breath.

**Fix**: Raise `silence_frames` to tolerate longer pauses:
```python
# Tolerates ~1.5 s of silence within an utterance (at 20 fps)
seg = LipMotionSegmenter(silence_frames=30)
```

### Clips Take Too Long to Flush After Silence
**Symptom**: After the speaker stops, the server waits several seconds before processing.

**Fix**: Lower `silence_frames`:
```python
# Flushes after ~0.5 s of silence (at 20 fps)
seg = LipMotionSegmenter(silence_frames=10)
```

---

## Edge Cases

### 1 · Mid-Sentence Pauses
A speaker naturally pauses between phrases. With default `silence_frames=20` (~1 s), a pause shorter than 1 s will not trigger a flush — the utterance stays in one clip. If your speakers habitually pause longer, increase `silence_frames` to match.

> [!TIP]
> For read-aloud speech (slower, deliberate pacing), try `silence_frames=30`. For conversational speech (faster, fewer pauses), `silence_frames=15` works well.

### 2 · No Speech / Idle Camera
When no face is detected, `_extract_lip_points` returns `None` → score = 0.0 → silent frame. `_speech_active` stays `False`, so frames are never buffered and no clip is ever sent. No special handling needed.

> [!NOTE]
> If the user turns away briefly mid-utterance, the face may vanish for a frame or two, producing score=0. Because `silence_frames` is counted consecutively, a 1-2 frame dropout will not trigger a flush.

### 3 · Noisy Background Movement (Head Turns, Gestures)
The 10 lip-specific landmarks are much less affected by global head pose than a full-face centroid would be. However, large head shakes can still produce non-zero lip displacement because landmarks are in image space, not face-relative space.

**Fix A — Increase threshold** slightly so micro head-motion below speech magnitude is ignored.

**Fix B — Use relative coordinates**: Subtract the lip centroid from each landmark before comparing. This makes the score translation-invariant:

```python
# In _compute_score, add normalisation:
def _compute_score(self, lip_pts):
    if lip_pts is None or self._prev_lip_pts is None:
        return 0.0
    center_now  = lip_pts.mean(axis=0)
    center_prev = self._prev_lip_pts.mean(axis=0)
    rel_now  = lip_pts  - center_now
    rel_prev = self._prev_lip_pts - center_prev
    delta = rel_now - rel_prev
    return float(np.linalg.norm(delta, axis=1).mean())
```
This change makes the segmenter robust to head translation while still capturing mouth open/close.

### 4 · Very Short Utterances (single words)
A 3-syllable word at 20 fps takes ~15–20 frames. The default `min_speech_frames=15` allows this. Lower it carefully — going below 10 frames risks sending sub-second clips that VSR cannot reliably decode.

### 5 · High Ambient Light Variation / JPEG Compression Noise
JPEG encoding at quality 0.75 introduces quantisation noise. This manifests as spurious small displacements even for a still face. Raising `movement_threshold` by 0.001–0.002 will absorb this noise floor.

---

## Recommended Starting Profiles

```python
# ── Studio / controlled lighting ──────────────────────────────────
seg = LipMotionSegmenter(
    movement_threshold=0.003,
    silence_frames=20,
    min_speech_frames=15,
)

# ── Laptop webcam / typical home office ───────────────────────────
seg = LipMotionSegmenter(
    movement_threshold=0.004,   # +0.001 for JPEG noise floor
    silence_frames=22,          # slightly more forgiving pause tolerance
    min_speech_frames=15,
)

# ── Noisy background / lots of head motion ────────────────────────
seg = LipMotionSegmenter(
    movement_threshold=0.006,
    silence_frames=25,
    min_speech_frames=18,
)
```

---

## Quick Diagnostic Checklist

1. **Watch the server logs** — every `[LipSegmenter]` line shows the frame index, silence counter, and buffer length. This gives ground truth on what the segmenter "sees".
2. **Watch the score bar** in the browser — it mirrors the server score in real-time. If it never rises above zero during speech, the face may not be detected (check camera framing, lighting).
3. **Set `logging.basicConfig(level=logging.DEBUG)`** in `app.py` to see per-frame debug logs from the segmenter during development.
4. **Print score distributions** for a 30-second recording to calibrate threshold empirically:
   ```python
   scores = []
   for jpeg in my_test_frames:
       seg.push_frame(jpeg)
       scores.append(seg.last_score)
   import numpy as np
   print(f"silence p50={np.percentile(scores,50):.4f}  speech p90={np.percentile(scores,90):.4f}")
   # Set threshold between these two values
   ```
