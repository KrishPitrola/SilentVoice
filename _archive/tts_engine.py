import subprocess
import platform
import threading

class TTSEngine:
    """Text-to-Speech engine using native OS commands (no pyttsx3)."""

    def __init__(self):
        self.os_name = platform.system()

    def is_available(self):
        """Always returns True as it relies on built-in OS tools."""
        return True

    def _execute_tts(self, text):
        """Executes the OS-specific TTS command."""
        try:
            # Strip out single quotes to prevent command injection/syntax errors in PowerShell
            safe_text = text.replace("'", "")

            if self.os_name == "Windows":
                cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak(\'{safe_text}\')"'
                subprocess.Popen(cmd, shell=True)
                
            elif self.os_name == "Darwin":  # Mac
                subprocess.Popen(['say', safe_text])
                
            else:  # Linux
                subprocess.Popen(['espeak', safe_text])
                
        except Exception as e:
            print(f"TTS Engine Error: {e}")

    def speak(self, text):
        """
        Speak the given text.
        Runs in a daemon thread to strictly ensure no blocking occurs.
        """
        threading.Thread(target=self._execute_tts, args=(text,), daemon=True).start()
