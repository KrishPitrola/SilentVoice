import cv2
import torch
from modules.camera import Camera
from modules.frame_buffer import FrameBuffer
from modules.lip_extractor import LipExtractor
from modules.sequence_preprocessor import SequencePreprocessor
from models.vsr_model import VisualSpeechRecognitionModel

def main():

    camera = Camera()
    lip_extractor = LipExtractor()
    frame_buffer = FrameBuffer()
    preprocessor = SequencePreprocessor()
    model = VisualSpeechRecognitionModel()
    model.to("cpu")
    model.eval()

    while True:

        frame = camera.get_frame()

        if frame is None:
            break

        lip_region = lip_extractor.extract_lips(frame)

        if lip_region is not None:
            frame_buffer.add_frame(lip_region)
            cv2.imshow("Lip Region", lip_region)

        if frame_buffer.is_full():
            sequence = frame_buffer.get_sequence()
            if sequence is not None:
                tensor = preprocessor.preprocess(sequence)
                if tensor is not None:
                    tensor = tensor.to("cpu")
                    with torch.no_grad():
                        output = model(tensor)
                        prediction = torch.argmax(output, dim=1)
                    print(f"Predicted token: {prediction.item()}")

                cv2.putText(
                    frame,
                    f"Sequence ready: {sequence.shape}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
        else:
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

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
    
    print(output)


if __name__ == "__main__":
    main()