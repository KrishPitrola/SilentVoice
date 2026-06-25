# mpc001_wrapper.py
import sys, os

class MPC001VSR:
    def __init__(self, config_path="mpc001_vsr/configs/GRID_V_WER1.2.ini", detector="mediapipe"):
        sys.path.append(os.path.join(os.path.dirname(__file__), "mpc001_vsr"))
        from mpc001_vsr.pipelines.pipeline import InferencePipeline

        self.base = os.path.dirname(os.path.abspath(__file__))
        config_full_path = os.path.join(self.base, config_path)

        prev = os.getcwd()
        os.chdir(os.path.join(self.base, "mpc001_vsr"))
        try:
            self.pipeline = InferencePipeline(
                config_full_path,
                device="cpu",
                detector=detector,
                face_track=True,
            )
        finally:
            os.chdir(prev)

    def transcribe(self, video_path):
        prev = os.getcwd()
        os.chdir(os.path.join(self.base, "mpc001_vsr"))
        try:
            result = self.pipeline(video_path)
            return result.strip() if result else ""
        except Exception as e:
            print(f"[MPC001VSR] Inference error: {e}")
            return ""
        finally:
            os.chdir(prev)