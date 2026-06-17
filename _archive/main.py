import cv2
import torch
from collections import deque

from modules.camera import Camera
from modules.frame_buffer import FrameBuffer
from modules.lip_extractor import LipExtractor
from modules.sequence_preprocessor import SequencePreprocessor
from models.vsr_model import VisualSpeechRecognitionModel


def main():

    # -------------------------
    # Initialize components
    # -------------------------
    camera = Camera()
    lip_extractor = LipExtractor()
    frame_buffer = FrameBuffer()
    preprocessor = SequencePreprocessor()

    # Prediction smoothing buffer
    pred_history = deque(maxlen=10)

    # -------------------------
    # Load trained model
    # -------------------------
    model = VisualSpeechRecognitionModel(vocab_size=50)

    try:
        model.load_state_dict(torch.load("vsr_model.pth", map_location="cpu"))
        print("✅ Trained model loaded successfully!")
    except Exception as e:
        print("⚠️ Could not load trained model:", e)
        print("Using untrained model...")

    model.to("cpu")
    model.eval()

    # -------------------------
    # Main loop
    # -------------------------
    while True:

        # 1. Captures frames from webcam
        frame = camera.get_frame()

        if frame is None:
            break

        # 2 & 3. Detects face using MediaPipe Face Mesh and Extracts lip region
        lip_region = lip_extractor.extract_lips(frame)

        # Handle None safely (if lips not detected)
        if lip_region is not None:
            frame_buffer.add_frame(lip_region)
            # 4. Display Separate window showing lip region
            cv2.imshow("Lip Region", lip_region)

        # -------------------------
        # When buffer is full → inference
        # -------------------------
        if frame_buffer.is_full():
            sequence = frame_buffer.get_sequence()

            if sequence is not None:
                tensor = preprocessor.preprocess(sequence)

                if tensor is not None:
                    tensor = tensor.to("cpu")

                    with torch.no_grad():
                        output = model(tensor)

                        # Convert to probabilities
                        probs = torch.softmax(output, dim=1)
                        confidence, prediction = torch.max(probs, dim=1)

                        prediction = prediction.item()
                        confidence = confidence.item()

                    # -------------------------
                    # Prediction smoothing
                    # -------------------------
                    pred_history.append(prediction)

                    # Get most frequent prediction
                    final_pred = max(set(pred_history), key=pred_history.count)

                    # Print clean output
                    print(f"Stable Prediction: {final_pred}, Confidence: {confidence:.2f}")

                    # Display on screen
                    cv2.putText(
                        frame,
                        f"Pred: {final_pred} ({confidence:.2f})",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # Show sequence info
            cv2.putText(
                frame,
                "Sequence Ready",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        else:
            # Buffer progress display
            cv2.putText(
                frame,
                f"Buffer: {len(frame_buffer.buffer)}/30",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 4. Display Full camera feed
        cv2.imshow("Camera Feed", frame)

        # Keep loop real-time (no delay)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # -------------------------
    # Cleanup
    # -------------------------
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()