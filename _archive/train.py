import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.grid_dataset import GridDataset
from models.vsr_model import VisualSpeechRecognitionModel


# -----------------------
# SAFE CPU SETTINGS
# -----------------------
torch.set_num_threads(2)   # limit CPU threads (prevents overload)

device = torch.device("cpu")
print("Using device:", device)


# -----------------------
# DATASET (REDUCED SIZE)
# -----------------------
dataset = GridDataset("data/grid/s1", max_videos=10)

dataloader = DataLoader(
    dataset,
    batch_size=1,   # VERY IMPORTANT (fix memory issue)
    shuffle=True
)


# -----------------------
# MODEL (LIGHT VERSION)
# -----------------------
model = VisualSpeechRecognitionModel(vocab_size=50)

model.to(device)


# -----------------------
# LOSS + OPTIMIZER
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# -----------------------
# TRAINING LOOP
# -----------------------
epochs = 10

for epoch in range(epochs):
    print(f"\nStarting Epoch {epoch+1}")
    total_loss = 0

    for i, (X, y) in enumerate(dataloader):
        print(f"Batch {i}")   # DEBUG LINE

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        print("Forward pass")   # DEBUG
        outputs = model(X)

        print("Loss calculation")   # DEBUG
        loss = criterion(outputs, y)

        print("Backward pass")   # DEBUG
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


# -----------------------
# SAVE MODEL
# -----------------------
torch.save(model.state_dict(), "vsr_model.pth")

print("Model saved successfully!")