import cv2
import numpy as np
import torch


class SequencePreprocessor:
    """Convert buffered lip frame sequences into model-ready PyTorch tensors."""

    def preprocess(self, sequence):
        """
        Args:
            sequence: NumPy array with shape (30, 112, 112, 3) in BGR format.

        Returns:
            torch.Tensor with shape (1, 30, 3, 112, 112).
        """
        if sequence is None:
            return None

        if not isinstance(sequence, np.ndarray):
            sequence = np.array(sequence)

        processed_frames = []

        for frame in sequence:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            normalized = rgb_frame.astype(np.float32) / 255.0
            chw = np.transpose(normalized, (2, 0, 1))
            processed_frames.append(chw)

        stacked = np.stack(processed_frames, axis=0)
        tensor = torch.from_numpy(stacked)

        # Add batch dimension: (30, 3, 112, 112) -> (1, 30, 3, 112, 112)
        tensor = tensor.unsqueeze(0)
        return tensor
