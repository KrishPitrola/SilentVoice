from mpc001_wrapper import MPC001VSR

print("Loading model...")
vsr = MPC001VSR()
print("Model loaded.\n")

result = vsr.transcribe("S:\SilentVoice\data\grid\s1\swwp2n.mpg")
print(f"Transcription: {result}")