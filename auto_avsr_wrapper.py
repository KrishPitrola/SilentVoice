# S:\SilentVoice\auto_avsr_wrapper.py
import sys, os

class AutoAVSRWrapper:
    def __init__(
        self,
        model_path="S:/auto_avsr/models/vsr_trlrs2lrs3vox2avsp_base.pth",
        detector="mediapipe"
    ):
        self.base = "S:/auto_avsr"
        sys.path.insert(0, self.base)
        prev = os.getcwd()
        os.chdir(self.base)
        try:
            from infer_pipeline import AutoAVSRPipeline
            self.pipeline = AutoAVSRPipeline(model_path, detector=detector)
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