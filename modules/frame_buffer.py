from collections import deque

import cv2
import numpy as np


class FrameBuffer:
    """Fixed-size buffer for lip frame sequences."""

    def __init__(self):
        self.buffer = deque(maxlen=30)

    def add_frame(self, frame):
        """Resize a lip frame to 128x64 and append it to the buffer."""
        if frame is None:
            return

        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        resized = cv2.resize(frame, (128, 64), interpolation=cv2.INTER_AREA)
        self.buffer.append(np.array(resized))

    def is_full(self):
        """Return True when 30 frames are available."""
        return len(self.buffer) == 30

    def get_sequence(self):
        """Return a (30, 64, 128, 3) array when the buffer is full, else None."""
        if not self.is_full():
            return None

        sequence = np.stack(self.buffer, axis=0)

        if sequence.ndim == 3:
            sequence = np.repeat(sequence[..., np.newaxis], 3, axis=-1)

        return sequence
