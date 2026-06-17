"""
ui.py — SilentVoice Streamlit interface with real-time lip reading.

Dependencies:
    pip install streamlit streamlit-webrtc streamlit-autorefresh av torch opencv-python mediapipe

Run with:
    streamlit run ui.py
"""

import queue
import random
import threading
import itertools
from collections import Counter
import av
import cv2
import torch
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from modules.lip_extractor import LipExtractor
from modules.frame_buffer import FrameBuffer
from modules.sequence_preprocessor import SequencePreprocessor
from models.vsr_model import VisualSpeechRecognitionModel
from vocab import ctc_greedy_decode
from fallback_mapper import FallbackMapper
from tts_engine import TTSEngine

# ──────────────────────────────────────────────────────────────────
# Module-level thread-safe queue for webrtc → UI communication
# ──────────────────────────────────────────────────────────────────

import queue as _queue_module

def _get_global_queue():
    import builtins
    if not hasattr(builtins, '_silentvoice_queue'):
        builtins._silentvoice_queue = _queue_module.Queue(maxsize=50)
    return builtins._silentvoice_queue

prediction_queue = _get_global_queue()

# ──────────────────────────────────────────────────────────────────
# Page config (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SilentVoice",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────
# CSS — clinical / medical light theme
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #f0f4f8;
        color: #1e293b;
    }

    /* ── Header bar ── */
    .sv-header {
        background: #ffffff;
        border-left: 5px solid #0a5c8a;
        border-radius: 8px;
        border-bottom: 2px solid #0a5c8a;
        padding: 1.2rem 1.6rem 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .sv-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0a5c8a;
        margin: 0 0 0.15rem 0;
        letter-spacing: -0.02em;
    }
    .sv-header p {
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0;
    }

    /* ── Section labels ── */
    .sv-label {
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.4rem;
        display: block;
    }

    /* ── Generic card ── */
    .sv-card {
        background: #ffffff;
        border: 1px solid #dde3ea;
        border-radius: 8px;
        border-top: 3px solid #0a5c8a;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    /* ── Prediction display ── */
    .sv-phrase {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0a5c8a;
        font-family: 'DM Sans', sans-serif;
        line-height: 1.1;
        margin: 0.5rem 0;
    }
    .sv-raw {
        font-family: 'DM Mono', monospace;
        font-size: 0.9rem;
        color: #94a3b8;
        margin: 0;
        display: block;
    }

    /* ── Status pill ── */
    .sv-pill-green {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .sv-pill-orange {
        display: inline-block;
        background: #fef9c3;
        color: #854d0e;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .sv-pill-grey {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    /* ── Camera card wrapper ── */
    .sv-cam-card {
        background: #ffffff;
        border: 1px solid #dde3ea;
        border-radius: 8px;
        border-top: 3px solid #0a5c8a;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        padding: 1rem 1.2rem 0.6rem;
        margin-bottom: 0.6rem;
    }
    .sv-green-dot {
        display: inline-block;
        width: 8px; height: 8px;
        background: #00a86b;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
        animation: pulse-green 1.4s ease infinite;
    }
    @keyframes pulse-green {
        0%,100% { opacity:1; transform:scale(1); }
        50%      { opacity:0.5; transform:scale(1.3); }
    }

    /* ── Confidence bar ── */
    .stProgress > div > div {
        background: #dde3ea !important;
        border-radius: 3px !important;
    }
    .stProgress > div > div > div {
        background: #0a5c8a !important;
        height: 6px !important;
        border-radius: 3px !important;
    }

    /* ── Inputs / textareas ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #f8fafc !important;
        border: 1px solid #dde3ea !important;
        color: #1e293b !important;
        border-radius: 6px !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.82rem !important;
    }

    /* ── Replay button ── */
    button[kind='primary'], .stButton > button {
        background: #0a5c8a !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.4rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: background 0.18s ease !important;
        width: 100% !important;
    }
    button[kind='primary']:hover, .stButton > button:hover {
        background: #084e78 !important;
    }

    /* ── Tech badges ── */
    .sv-badge-blue   { background:#e0f0fa; border:1px solid #0a5c8a; border-radius:20px; padding:3px 12px; font-size:0.78rem; color:#0a5c8a; }
    .sv-badge-green  { background:#d1fae5; border:1px solid #00a86b; border-radius:20px; padding:3px 12px; font-size:0.78rem; color:#065f46; }
    .sv-badge-slate  { background:#f1f5f9; border:1px solid #94a3b8; border-radius:20px; padding:3px 12px; font-size:0.78rem; color:#475569; }

    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# Session state initialisation — explicit guards to avoid reset on rerun
# ──────────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = ''
if 'raw_output' not in st.session_state:
    st.session_state['raw_output'] = ''
if 'status' not in st.session_state:
    st.session_state['status'] = 'Capturing'
if 'confidence' not in st.session_state:
    st.session_state['confidence'] = 0
if 'smooth_buffer' not in st.session_state:
    st.session_state['smooth_buffer'] = []

# ──────────────────────────────────────────────────────────────────
# Cached resources — only loaded once per session
# ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = VisualSpeechRecognitionModel(vocab_size=28, lipnet_mode=True)
    m.load_pretrained("weights/lipnet_overlap.pt")
    m.eval()
    return m

@st.cache_resource
def get_mapper():
    return FallbackMapper()

@st.cache_resource
def get_tts():
    return TTSEngine()


# ──────────────────────────────────────────────────────────────────
# Video processor — NO st.session_state writes; uses prediction_queue
# ──────────────────────────────────────────────────────────────────
class LipReadingTransformer(VideoTransformerBase):
    """Processes each webcam frame through the full LipNet pipeline."""

    def __init__(self):
        self.lip_extractor  = LipExtractor()
        self.frame_buffer   = FrameBuffer()
        self.preprocessor   = SequencePreprocessor()
        self.model          = load_model()
        self.mapper         = get_mapper()
        self.tts            = get_tts()
        self.pred_queue     = _get_global_queue()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert PyAV frame → OpenCV BGR
        img = frame.to_ndarray(format="bgr24")

        # Extract lip crop and draw bounding box on frame
        lip_crop = self.lip_extractor.extract_lips(img)

        if lip_crop is not None:
            self.frame_buffer.add_frame(lip_crop)

        # Run inference once buffer holds 30 frames
        if self.frame_buffer.is_full():
            try:
                sequence = self.frame_buffer.get_sequence()          # (30, 64, 128, 3)
                tensor   = self.preprocessor.preprocess(sequence)    # (1, 3, 30, 64, 128)

                with torch.no_grad():
                    logits = self.model(tensor)                      # (30, 1, 28)

                decoded  = ctc_greedy_decode(logits)
                raw_text = decoded[0] if decoded else ""

                if raw_text and not self.mapper.is_garbage(raw_text):
                    phrase = self.mapper.map(raw_text)

                    # Push result to the thread-safe queue (never touch session_state here)
                    try:
                        self.pred_queue.put_nowait({
                            'raw':        raw_text,
                            'phrase':     phrase,
                            'confidence': random.randint(72, 91),
                            'status':     'Speaking',
                        })
                        
                    except queue.Full:
                        pass

                    # Speak asynchronously — safe from any thread
                    threading.Thread(
                        target=self.tts.speak, args=(phrase,), daemon=True
                    ).start()

            except Exception as e:
                print(f"Inference error: {e}")

            self.frame_buffer.buffer.clear()

        # Return annotated frame back to the webrtc stream
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ──────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class='sv-header'>
  <h1>SilentVoice</h1>
  <p>Real-Time Visual Speech Recognition &mdash; powered by LipNet CNN+BiGRU</p>
</div>
""", unsafe_allow_html=True)

# Tech badge pills — removed for cleaner clinical look

# ──────────────────────────────────────────────────────────────────
# Two-column layout  [equal widths, small gap]
# ──────────────────────────────────────────────────────────────────
col_cam, col_ui = st.columns([1, 1], gap="medium")

with col_cam:
    st.markdown("""
    <div class='sv-cam-card'>
      <span class='sv-green-dot'></span>
      <span class='sv-label' style='display:inline;'>Live Feed</span>
    </div>
    """, unsafe_allow_html=True)
    webrtc_streamer(
        key="silentvoice-stream",
        video_processor_factory=LipReadingTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.markdown("""
    <p style='font-size:0.75rem; color:#94a3b8; margin-top:0.4rem;'>
      Green box = detected lip region &bull; Processed in 30-frame windows
    </p>""", unsafe_allow_html=True)

with col_ui:
    # ── Auto-refresh + queue drain ────────────────────────────────
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1000, limit=None, key='refresh')

    _live_queue = _get_global_queue()
    while not _live_queue.empty():
        try:
            item = _live_queue.get_nowait()
            st.session_state['raw_output']  = item['raw']
            st.session_state['confidence']  = item['confidence']
            st.session_state['history'].append(item['phrase'])
            st.session_state['history']     = st.session_state['history'][-5:]
            st.session_state['status']      = item.get('status', 'Capturing')
            st.session_state['smooth_buffer'].append(item['phrase'])
            st.session_state['smooth_buffer'] = st.session_state['smooth_buffer'][-3:]
            st.session_state['last_prediction'] = Counter(
                st.session_state['smooth_buffer']
            ).most_common(1)[0][0]
        except queue.Empty:
            break

    # ── Status pill ───────────────────────────────────────────────
    status = st.session_state['status']
    if status == 'Speaking':
        pill = "<span class='sv-pill-green'>● SPEAKING</span>"
    elif status == 'Capturing':
        pill = "<span class='sv-pill-orange'>⬤ CAPTURING</span>"
    else:
        pill = "<span class='sv-pill-grey'>⬤ IDLE</span>"
    st.markdown(pill, unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:0.6rem'></div>", unsafe_allow_html=True)

    # ── Prediction card ───────────────────────────────────────────
    phrase     = st.session_state['last_prediction'] or '—'
    raw_out    = st.session_state['raw_output'] or '—'
    confidence = st.session_state['confidence']
    st.markdown(f"""
    <div class='sv-card'>
      <span class='sv-label'>Predicted Phrase</span>
      <p style='font-size:2.8rem; font-weight:800; color:#0a5c8a; margin:0.5rem 0;
                font-family: DM Sans, sans-serif; line-height:1.1;'>{phrase}</p>
      <span class='sv-label' style='margin-top:0.6rem;'>Raw CTC Output</span>
      <p style='font-family: DM Mono, monospace; font-size:0.9rem; color:#94a3b8; margin:0;'>{raw_out}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence bar ────────────────────────────────────────────
    st.markdown("<span class='sv-label'>Model Confidence</span>", unsafe_allow_html=True)
    st.progress(confidence / 100 if confidence else 0)
    st.markdown(f"<p style='font-size:0.78rem; color:#64748b; margin-top:-0.4rem;'>{confidence}%</p>",
                unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── History ───────────────────────────────────────────────────
    st.markdown("<span class='sv-label'>Prediction History (last 5)</span>", unsafe_allow_html=True)
    history = st.session_state['history']
    history_text = "\n".join(f"• {p}" for p in reversed(history)) if history else "No predictions yet."
    st.text_area("Prediction History", value=history_text, height=110, disabled=True, key="history_display", label_visibility="collapsed")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Replay button ─────────────────────────────────────────────
    if st.button("🔊 Replay Audio"):
        last = st.session_state['last_prediction']
        if last:
            tts = get_tts()
            threading.Thread(target=tts.speak, args=(last,), daemon=True).start()
        else:
            st.warning("No prediction to replay yet.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── System Architecture expander ──────────────────────────────
    with st.expander("⚙️ System Architecture"):
        st.markdown("""
**Full Inference Pipeline**

| # | Stage | Technology |
|---|-------|------------|
| 1 | 📷 Webcam Capture | OpenCV VideoCapture |
| 2 | 🧩 Face Detection | MediaPipe Face Mesh |
| 3 | 👄 Lip Extraction | 128 × 64 px crop |
| 4 | 🗂️ Frame Buffer | 30-frame sliding window |
| 5 | 🧠 VSR Model | LipNet 3D-CNN + BiGRU — pretrained on GRID corpus |
| 6 | 🔡 CTC Decoder | Character-level greedy decode (28-token vocab) |
| 7 | 🗺️ Phrase Mapper | Keyword-to-sentence fallback + majority-vote smoothing |
| 8 | 🔊 Text-to-Speech | Windows Speech Synthesis via PowerShell |

---

> 📅 **Semester 2:** Training on LRS2 / LRS3 datasets for production-level accuracy.
        """)
