import cv2
import os
import numpy as np

from modules.lip_extractor import LipExtractor


# -------------------------------
# Load video and extract frames
# -------------------------------
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()
    return frames


# -------------------------------
# Build 30-frame sequences
# -------------------------------
def build_sequences(frames, seq_length=30):
    sequences = []

    for i in range(0, len(frames) - seq_length + 1):
        seq = frames[i:i + seq_length]
        sequences.append(np.array(seq))

    return sequences


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    data_path = "data/grid/s1"

    lip_extractor = LipExtractor()

    video_files = [f for f in os.listdir(data_path) if f.endswith(".mpg")]

    print("Total videos found:", len(video_files))

    # Only test a few videos
    for video_name in video_files[:10]:

        video_path = os.path.join(data_path, video_name)

        print("\nProcessing:", video_name)

        # Step 1: Load frames
        frames = load_video(video_path)
        print("Frame count:", len(frames))

        # Step 2: Extract lip regions
        lip_frames = []

        for frame in frames:
            lip_region = lip_extractor.extract_lips(frame)

            if lip_region is not None:
                lip_region = cv2.resize(lip_region, (112, 112))
                lip_frames.append(lip_region)

        print("Valid lip frames:", len(lip_frames))

        # Step 3: Build sequences
        sequences = build_sequences(lip_frames)

        print("Total sequences:", len(sequences))

        # Debug: print shape
        if len(sequences) > 0:
            print("Sequence shape:", sequences[0].shape)

    cv2.destroyAllWindows()