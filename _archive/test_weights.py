import torch
from models.vsr_model import VisualSpeechRecognitionModel

model = VisualSpeechRecognitionModel(vocab_size=28, lipnet_mode=True)
model.load_pretrained("weights/lipnet_overlap.pt")

# Test forward pass with correct shape (batch, channels, seq, H, W)
dummy = torch.zeros(1, 3, 30, 64, 128)
out = model(dummy)
print(f"Output shape: {out.shape}")  # expect (30, 1, 28)