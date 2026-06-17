import cv2
import mediapipe as mp
import numpy as np

class LipExtractor:

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

        # Mouth landmark indices from MediaPipe
        self.lip_indices = [
            61,146,91,181,84,17,314,405,321,375,
            291,308,324,318,402,317,14,87,178,88
        ]

    def extract_lips(self, frame):

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        points = []

        for idx in self.lip_indices:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))

        points = np.array(points)

        x, y, w_box, h_box = cv2.boundingRect(points)

        padding = 10

        x = max(0, x - padding)
        y = max(0, y - padding)

        # Use .copy() so the green box doesn't appear in the cropped lip_region
        lip_region = frame[y:y+h_box+padding, x:x+w_box+padding].copy()

        # Resize the crop to (width=128, height=64) for LipNet 3D CNN frontend
        if lip_region.size > 0:
            lip_region = cv2.resize(lip_region, (128, 64))

        # Draw a green bounding box around the lip region on the main frame
        cv2.rectangle(frame, (x, y), (x+w_box+padding, y+h_box+padding), (0, 255, 0), 2)

        return lip_region