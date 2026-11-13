
import time
from pathlib import Path

import soundfile as sf
from kokoro_onnx import Kokoro


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Kokoro model files
MODEL_PATH = PROJECT_ROOT / "models" / "kokoro" / "kokoro-v1.0.int8.onnx"
VOICES_PATH = PROJECT_ROOT / "models" / "kokoro" / "voices-v1.0.bin"


class KokoroTTSClient:

    def __init__(self, voice="af_sarah", speed=1.0):

        # Verify that the required model files exist.
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Kokoro model not found: {MODEL_PATH}"
            )

        if not VOICES_PATH.exists():
            raise FileNotFoundError(
                f"Kokoro voice data not found: {VOICES_PATH}"
            )

        self.voice = voice
        self.speed = speed

        print("\n[TTS] Loading Kokoro model...")

        start_time = time.perf_counter()

        self.kokoro = Kokoro(
            str(MODEL_PATH),
            str(VOICES_PATH)
        )

        loading_time = time.perf_counter() - start_time

        print(
            f"[TTS] Model loaded in {loading_time:.2f} seconds."
        )

    # ====================================================
    # Generate audio
    # ====================================================

    def generate_audio(self, text):

        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        text = text.strip()

        if not text:
            raise ValueError("Text cannot be empty.")

        print("\n[TTS] Starting audio generation...")

        print(
            "[TTS] Input characters:",
            len(text)
        )

        start_time = time.perf_counter()

        samples, sample_rate = self.kokoro.create(
            text,
            voice=self.voice,
            speed=self.speed,
            lang="en-us"
        )

        generation_time = time.perf_counter() - start_time

        audio_duration = len(samples) / sample_rate

        real_time_factor = (
            generation_time / audio_duration
            if audio_duration > 0
            else 0
        )

        print("\n[TTS] Audio generation completed.")

        print(
            f"[TTS] Generation time: {generation_time:.2f} seconds"
        )

        print(
            f"[TTS] Generated audio duration: {audio_duration:.2f} seconds"
        )

        print(
            f"[TTS] Real-time factor: {real_time_factor:.2f}"
        )

        return samples, sample_rate

    # ====================================================
    # Save audio
    # ====================================================

    def save_audio(self, text, output_path):

        samples, sample_rate = self.generate_audio(text)

        output_path = Path(output_path)

        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        sf.write(
            str(output_path),
            samples,
            sample_rate
        )

        print(
            f"\n[TTS] Audio saved: {output_path}"
        )

        return output_path
