import torch

from models.vsr_model import VisualSpeechRecognitionModel


def main():

    # create model
    model = VisualSpeechRecognitionModel()

    # dummy input tensor
    dummy_sequence = torch.randn(1, 30, 3, 112, 112)

    # forward pass
    output = model(dummy_sequence)

    print("Input shape:", dummy_sequence.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()