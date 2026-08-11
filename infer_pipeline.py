# S:\auto_avsr\infer_pipeline.py
from huggingface_hub.inference._generated.types import zero_shot_image_classification
import os, sys, argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))

from lightning import ModelModule
from datamodule.transforms import VideoTransform

class AutoAVSRPipeline(torch.nn.Module):
    def __init__(self, ckpt_path, detector="mediapipe", device="cpu"):
        super().__init__()
        if detector == "mediapipe":
            from preparation.detectors.mediapipe.detector import LandmarksDetector
            from preparation.detectors.mediapipe.video_process import VideoProcess
            self.landmarks_detector = LandmarksDetector()
            self.video_process = VideoProcess(convert_gray=False)
        elif detector == "retinaface":
            from preparation.detectors.retinaface.detector import LandmarksDetector
            from preparation.detectors.retinaface.video_process import VideoProcess
            self.landmarks_detector = LandmarksDetector(device=device)
            self.video_process = VideoProcess(convert_gray=False)

        self.video_transform = VideoTransform(subset="test")

        args = argparse.Namespace()
        setattr(args, 'modality', 'video')

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.modelmodule = ModelModule(args)
        self.modelmodule.model.load_state_dict(ckpt)
        self.modelmodule.eval()

    def forward(self, data_filename):
        import av
        import numpy as np
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"{data_filename} does not exist."
        
        # load video with av instead of torchvision
        container = av.open(data_filename)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
        container.close()
        video = np.array(frames)
        
        landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video).permute((0, 3, 1, 2))
        video = self.video_transform(video)
        with torch.no_grad():
            transcript = self.modelmodule(video)
        return transcript