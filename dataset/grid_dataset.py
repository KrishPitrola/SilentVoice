import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from modules.lip_extractor import LipExtractor


class GridDataset(Dataset):

    def __init__(self, data_path, seq_length=30, max_videos=10):
        self.data_path = data_path
        self.seq_length = seq_length
        self.lip_extractor = LipExtractor()

        self.video_files = sorted(
            [f for f in os.listdir(data_path) if f.endswith(".mpg")]
        )

        # Use limited videos for testing
        self.video_files = self.video_files[:max_videos]

        # Create label mapping
        self.label_map = {name: idx for idx, name in enumerate(self.video_files)}

        self.samples = []
        self._prepare_dataset()

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames

    def _build_sequences(self, frames):
        sequences = []

        for i in range(0, len(frames) - self.seq_length + 1):
            seq = frames[i:i + self.seq_length]
            sequences.append(np.array(seq))

        return sequences

    def _prepare_dataset(self):

        print("Preparing dataset...")

        for video_name in self.video_files:

            video_path = os.path.join(self.data_path, video_name)

            frames = self._load_video(video_path)

            lip_frames = []

            for frame in frames:
                lip = self.lip_extractor.extract_lips(frame)

                if lip is not None:
                    lip = cv2.resize(lip, (112, 112))
                    lip_frames.append(lip)

            sequences = self._build_sequences(lip_frames)

            label = self.label_map[video_name]

            for seq in sequences:
                self.samples.append((seq, label))

        print("Total samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]

        # Normalize
        seq = torch.tensor(seq, dtype=torch.float32) / 255.0

        # Convert (30,112,112,3) → (30,3,112,112)
        seq = seq.permute(0, 3, 1, 2)

        return seq, label