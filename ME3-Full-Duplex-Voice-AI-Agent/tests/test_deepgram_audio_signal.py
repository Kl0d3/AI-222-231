
import numpy as np

from src.tts.deepgram_client import DeepgramTTSClient


client = DeepgramTTSClient()

audio_chunks = []


def receive_audio(audio_bytes):

    samples = np.frombuffer(
        audio_bytes,
        dtype="<i2"
    )

    if samples.size:
        audio_chunks.append(samples.copy())


try:

    print("\nGenerating test speech...")

    client.generate_audio_stream(
        text="Hello. This is an audio signal test.",
        on_audio_chunk=receive_audio
    )

    if not audio_chunks:
        print("\nNo audio samples received.")

    else:

        audio = np.concatenate(audio_chunks).astype(
            np.float32
        )

        audio /= 32768.0

        print("\n========== SIGNAL DIAGNOSTICS ==========")

        print("Total samples:", len(audio))
        print("Duration:", len(audio) / 24000, "seconds")
        print("Minimum amplitude:", np.min(audio))
        print("Maximum amplitude:", np.max(audio))
        print("Peak amplitude:", np.max(np.abs(audio)))
        print("RMS amplitude:", np.sqrt(np.mean(audio ** 2)))

        print("========================================")

finally:

    print("\nSignal test completed.")
