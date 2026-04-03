import torch
import torch.nn as nn


class VisualSpeechRecognitionModel(nn.Module):
    """Lightweight CNN model for visual speech recognition (no LSTM)."""

    def __init__(self, vocab_size=50):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 → 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 → 28
        )

        # Feature dimension after CNN
        self.feature_dim = 64 * 28 * 28

        # Final classifier (no LSTM now)
        self.classifier = nn.Linear(self.feature_dim, vocab_size)

    def forward(self, x):
        """
        Args:
            x: (batch, 30, 3, 112, 112)
        Returns:
            logits: (batch, vocab_size)
        """

        batch_size, seq_len, C, H, W = x.shape

        # Merge batch + sequence
        x = x.reshape(batch_size * seq_len, C, H, W)

        # CNN processing
        features = self.cnn(x)

        # Flatten
        features = features.reshape(batch_size, seq_len, -1)

        # Take LAST frame only (fast + stable)
        last_frame_features = features[:, -1, :]

        # Classification
        logits = self.classifier(last_frame_features)

        return logits