import torch
import torch.nn as nn


class VisualSpeechRecognitionModel(nn.Module):
    """Simple CNN + BiLSTM model for visual speech recognition."""

    def __init__(self, vocab_size=50, hidden_size=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Input frames are 112x112. After two 2x2 pools -> 28x28.
        self.feature_dim = 64 * 28 * 28

        self.bilstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 30, 3, 112, 112)

        Returns:
            logits: Tensor of shape (batch, vocab_size)
        """
        batch_size, seq_len, channels, height, width = x.shape

        # Merge batch and time so CNN processes each frame independently.
        x = x.reshape(batch_size * seq_len, channels, height, width)

        frame_features = self.cnn(x)
        frame_features = frame_features.reshape(batch_size * seq_len, -1)

        # Restore temporal layout for BiLSTM: (batch, seq_len, feature_dim).
        sequence_features = frame_features.reshape(batch_size, seq_len, self.feature_dim)

        _, (h_n, _) = self.bilstm(sequence_features)

        # h_n layout for 1-layer BiLSTM: [forward_last, backward_last].
        forward_last = h_n[0]
        backward_last = h_n[1]
        final_hidden = torch.cat([forward_last, backward_last], dim=1)

        logits = self.classifier(final_hidden)
        return logits
