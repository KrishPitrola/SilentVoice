import cv2
import numpy as np
import torch


class SequencePreprocessor:
    """Convert buffered lip frame sequences into model-ready PyTorch tensors."""

    def preprocess(self, sequence):
        """
        Args:
            sequence: NumPy array with shape (30, 64, 128, 3) in BGR format.

        Returns:
            torch.Tensor with shape (1, 3, 30, 64, 128).
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

        # Add batch dimension: (30, 3, 64, 128) -> (1, 30, 3, 64, 128)
        tensor = tensor.unsqueeze(0)
        
        # Rearrange to match 3D CNN input: (batch, channels, seq_len, H, W) -> (1, 3, 30, 64, 128)
        tensor = tensor.permute(0, 2, 1, 3, 4).contiguous()
        
        return tensor
