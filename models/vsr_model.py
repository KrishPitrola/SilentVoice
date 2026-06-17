import torch
import torch.nn as nn


class VisualSpeechRecognitionModel(nn.Module):
    """
    LipNet-style Visual Speech Recognition model in PyTorch.

    Architecture (lipnet_mode=True):
      - 3D CNN frontend: 3x (Conv3d → BatchNorm3d → ReLU → MaxPool3d → Dropout)
      - Reshape to sequence
      - 2x Bidirectional GRU (hidden_size=256) — single nn.GRU with num_layers=2
      - Linear output layer (named 'classifier')

    Input:  (batch, 3, T, H, W)             — channels-first video clip
    Output: (seq_len, batch, vocab_size)     — ready for nn.CTCLoss

    Checkpoint key mapping (VIPL-AV LipNet-PyTorch):
      conv1/conv2/conv3 → frontend_3d.0 / .5 / .10
      gru1 (l0)         → gru (l0)
      gru2 (l0)         → gru (l1)
      FC                → classifier

    Legacy mode (lipnet_mode=False):
      Retains the original lightweight 2D CNN + Linear head.
    """

    def __init__(
        self,
        vocab_size: int = 28,
        lipnet_mode: bool = True,
    ):
        super().__init__()
        self.lipnet_mode = lipnet_mode
        self.vocab_size = vocab_size

        if self.lipnet_mode:
            self.gru_input_size = 96 * 4 * 8
            # ------------------------------------------------------------------
            # 3D CNN Frontend — Sequential so indices match remapped keys:
            #   frontend_3d.0  → Conv3d  (block 1)   conv1.weight / conv1.bias
            #   frontend_3d.1  → BN3d    (block 1)   — no ckpt key, init only
            #   frontend_3d.2  → ReLU    (block 1)
            #   frontend_3d.3  → MaxPool (block 1)
            #   frontend_3d.4  → Dropout (block 1)
            #   frontend_3d.5  → Conv3d  (block 2)   conv2.weight / conv2.bias
            #   frontend_3d.6  → BN3d    (block 2)
            #   ...
            #   frontend_3d.10 → Conv3d  (block 3)   conv3.weight / conv3.bias
            # ------------------------------------------------------------------
            self.frontend_3d = nn.Sequential(
                # Block 1  (indices 0–4)
                nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.Dropout(0.5),

                # Block 2  (indices 5–9)
                nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.Dropout(0.5),

                # Block 3  (indices 10–14)
                nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.BatchNorm3d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.Dropout(0.5),
            )

            # ------------------------------------------------------------------
            # Bidirectional GRU — single module, 2 layers, named 'gru'
            # gru1 (ckpt) → layer 0 of this module
            # gru2 (ckpt) → layer 1 of this module
            # ------------------------------------------------------------------
            self.gru = nn.GRU(
                input_size=self.gru_input_size,
                hidden_size=256,
                num_layers=2,
                batch_first=False,    # (seq, batch, features)
                bidirectional=True,
                dropout=0.5,
            )

            # ------------------------------------------------------------------
            # Output Linear — named 'classifier'  (ckpt key: FC)
            # ------------------------------------------------------------------
            self.classifier = nn.Linear(256 * 2, vocab_size)

        else:
            # ------------------------------------------------------------------
            # Legacy lightweight 2D CNN mode (original architecture)
            # ------------------------------------------------------------------
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),   # 112 → 56

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),   # 56 → 28
            )
            self.feature_dim = 64 * 28 * 28
            self.classifier = nn.Linear(self.feature_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, T, H, W)       in lipnet_mode
               (batch, T, 3, H, W)       in legacy mode
        Returns:
            lipnet_mode: (seq_len, batch, vocab_size) — for CTCLoss
            legacy mode: (batch, vocab_size)
        """
        if self.lipnet_mode:
            # x: (batch, 3, T, H, W)
            x = self.frontend_3d(x)
            # x: (batch, 96, T, H', W')

            batch_size, C, T, H, W = x.shape

            # Reshape → (T, batch, C*H*W) for GRU
            x = x.permute(2, 0, 1, 3, 4)            # (T, batch, 96, H', W')
            x = x.reshape(T, batch_size, C * H * W)  # (T, batch, gru_input_size)

            # Bidirectional GRU × 2 layers
            x, _ = self.gru(x)   # (T, batch, 512)

            # Linear projection to vocab
            x = self.classifier(x)  # (T, batch, vocab_size)
            return x

        else:
            # Legacy mode: (batch, T, 3, H, W)
            batch_size, seq_len, C, H, W = x.shape
            x = x.reshape(batch_size * seq_len, C, H, W)
            features = self.cnn(x)
            features = features.reshape(batch_size, seq_len, -1)
            last_frame_features = features[:, -1, :]
            return self.classifier(last_frame_features)

    def load_pretrained(self, path: str) -> None:
        """
        Load pretrained VIPL-AV LipNet checkpoint with key remapping.

        Checkpoint key  →  Model key
        ─────────────────────────────────────────────────────────────
        conv1.weight/bias        → frontend_3d.0.weight/bias
        conv2.weight/bias        → frontend_3d.5.weight/bias
        conv3.weight/bias        → frontend_3d.10.weight/bias
        gru1.*_l0[_reverse]      → gru.*_l0[_reverse]
        gru2.*_l0[_reverse]      → gru.*_l1[_reverse]
        FC.weight/bias           → classifier.weight/bias
        (BN keys absent in ckpt; BN layers initialise from scratch)
        """
        checkpoint = torch.load(path, map_location="cpu")

        # Support both raw state_dicts and {'state_dict': ...} wrappers
        state_dict = checkpoint.get("state_dict", checkpoint)

        # ------------------------------------------------------------------
        # Explicit key remapping table
        # ------------------------------------------------------------------
        KEY_MAP = {
            # Conv weights/biases
            "conv1.weight": "frontend_3d.0.weight",
            "conv1.bias":   "frontend_3d.0.bias",
            "conv2.weight": "frontend_3d.5.weight",
            "conv2.bias":   "frontend_3d.5.bias",
            "conv3.weight": "frontend_3d.10.weight",
            "conv3.bias":   "frontend_3d.10.bias",

            # GRU layer 0  (gru1 → gru l0)
            "gru1.weight_ih_l0":         "gru.weight_ih_l0",
            "gru1.weight_hh_l0":         "gru.weight_hh_l0",
            "gru1.bias_ih_l0":           "gru.bias_ih_l0",
            "gru1.bias_hh_l0":           "gru.bias_hh_l0",
            "gru1.weight_ih_l0_reverse": "gru.weight_ih_l0_reverse",
            "gru1.weight_hh_l0_reverse": "gru.weight_hh_l0_reverse",
            "gru1.bias_ih_l0_reverse":   "gru.bias_ih_l0_reverse",
            "gru1.bias_hh_l0_reverse":   "gru.bias_hh_l0_reverse",

            # GRU layer 1  (gru2 l0 → gru l1)
            "gru2.weight_ih_l0":         "gru.weight_ih_l1",
            "gru2.weight_hh_l0":         "gru.weight_hh_l1",
            "gru2.bias_ih_l0":           "gru.bias_ih_l1",
            "gru2.bias_hh_l0":           "gru.bias_hh_l1",
            "gru2.weight_ih_l0_reverse": "gru.weight_ih_l1_reverse",
            "gru2.weight_hh_l0_reverse": "gru.weight_hh_l1_reverse",
            "gru2.bias_ih_l0_reverse":   "gru.bias_ih_l1_reverse",
            "gru2.bias_hh_l0_reverse":   "gru.bias_hh_l1_reverse",

            # Output linear
            "FC.weight": "classifier.weight",
            "FC.bias":   "classifier.bias",
        }

        # Build remapped state dict — skip keys not in the map (e.g. BN running stats)
        remapped = {}
        skipped  = []
        for ckpt_key, tensor in state_dict.items():
            if ckpt_key in KEY_MAP:
                remapped[KEY_MAP[ckpt_key]] = tensor
            else:
                skipped.append(ckpt_key)

        # ------------------------------------------------------------------
        # Load with strict=False (BN params will remain at default init)
        # ------------------------------------------------------------------
        result = self.load_state_dict(remapped, strict=False)

        matched_count = len(remapped) - len(result.unexpected_keys)
        print(f"Matched keys    : {matched_count} / {len(remapped)}")
        print(f"Missing keys    : {len(result.missing_keys)}")
        print(f"Unexpected keys  : {len(result.unexpected_keys)}")
        print(f"Skipped (no map) : {len(skipped)}")

        if result.missing_keys:
            print("\nMissing (in model, not remapped from ckpt):")
            for k in result.missing_keys:
                print(f"  - {k}")

        if skipped:
            print("\nSkipped checkpoint keys (no remapping defined):")
            for k in skipped:
                print(f"  ~ {k}")