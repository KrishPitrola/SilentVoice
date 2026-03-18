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

        lip_region = frame[y:y+h_box+padding, x:x+w_box+padding]

        return lip_region