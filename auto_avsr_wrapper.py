# # S:\SilentVoice\auto_avsr_wrapper.py
# import sys, os
# sys.path.insert(0,'S:/auto_avsr')
# from infer_pipeline import AutoAVSRPipeline

# class AutoAVSRWrapper:
#     def __init__(
#         self,
#         model_path="S:/auto_avsr/models/vsr_trlrs2lrs3vox2avsp_base.pth",
#         detector="mediapipe"
#     ):
#         self.base = "S:/auto_avsr"
#         prev = os.getcwd()
#         os.chdir(self.base)
#         try:
#             self.pipeline = AutoAVSRPipeline(model_path, detector=detector)
#         finally:
#             os.chdir(prev)

#     def transcribe(self, video_path: str) -> str:
#         prev = os.getcwd()
#         os.chdir(self.base)
#         try:
#             result = self.pipeline(video_path)
#             return result.strip() if result else ""
#         except Exception as e:
#             print(f"[AutoAVSR] Error: {e}")
#             return ""
#         finally:
#             os.chdir(prev)
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
AUTO_AVSR_DIR = BASE_DIR / "auto_avsr"

sys.path.insert(0, str(AUTO_AVSR_DIR))

from infer_pipeline import AutoAVSRPipeline


class AutoAVSRWrapper:
    def __init__(
        self,
        model_path=None,
        detector="mediapipe"
    ):
        self.base = str(AUTO_AVSR_DIR)

        if model_path is None:
            model_path = AUTO_AVSR_DIR / "models" / "vsr_trlrs2lrs3vox2avsp_base.pth"

        model_path = str(model_path)

        prev = os.getcwd()
        os.chdir(self.base)

        try:
            self.pipeline = AutoAVSRPipeline(
                model_path,
                detector=detector
            )
        finally:
            os.chdir(prev)

    def transcribe(self, video_path: str) -> str:
        prev = os.getcwd()
        os.chdir(self.base)

        try:
            result = self.pipeline(video_path)
            return result.strip() if result else ""
        except Exception as e:
            print(f"[AutoAVSR] Error: {e}")
            return ""
        finally:
            os.chdir(prev)