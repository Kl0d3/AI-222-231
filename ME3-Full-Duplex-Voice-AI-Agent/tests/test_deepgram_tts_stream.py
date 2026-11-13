
import io
import time
import wave
from pathlib import Path

from dotenv import load_dotenv
from deepgram import DeepgramClient
from deepgram.speak.v1.types import SpeakV1Text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

MODEL = "aura-2-asteria-en"
SAMPLE_RATE = 24000

TEST_TEXT = (
    "Hello! This is a test of Deepgram's real-time text-to-speech "
    "streaming connection. The goal is to receive audio as soon "
    "as it becomes available."
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "deepgram_stream_test.wav"


def main():

    print("\nConnecting to Deepgram Speak WebSocket...")

    client = DeepgramClient()

    audio_chunks = []
    total_audio_bytes = 0
    first_audio_time = None
    flushed = False

    start_time = time.perf_counter()

    try:

        with client.speak.v1.connect(
            model=MODEL,
            encoding="linear16",
            sample_rate=SAMPLE_RATE
        ) as connection:

            print("WebSocket connected.")

            connection.send_text(
                SpeakV1Text(text=TEST_TEXT)
            )

            print("Text sent.")

            connection.send_flush()

            print("Flush sent. Waiting for audio...\n")

            while not flushed:

                message = connection.recv()

                if isinstance(message, bytes):

                    if first_audio_time is None:
                        first_audio_time = time.perf_counter() - start_time

                        print(
                            f"[AUDIO] First audio received after "
                            f"{first_audio_time:.2f} seconds."
                        )

                    audio_chunks.append(message)
                    total_audio_bytes += len(message)

                    print(
                        f"[AUDIO] Received {len(message)} bytes. "
                        f"Total: {total_audio_bytes} bytes."
                    )

                else:

                    print(
                        f"[EVENT] {type(message).__name__}: {message}"
                    )

                    if type(message).__name__ == "SpeakV1Flushed":
                        flushed = True

        total_time = time.perf_counter() - start_time

        if not audio_chunks:
            raise RuntimeError(
                "Deepgram completed without returning audio."
            )

        audio_data = b"".join(audio_chunks)

        with wave.open(str(OUTPUT_FILE), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_data)

        print("\n========== STREAMING TEST RESULTS ==========")
        print(f"First audio: {first_audio_time:.2f} seconds")
        print(f"Total generation time: {total_time:.2f} seconds")
        print(f"Audio chunks: {len(audio_chunks)}")
        print(f"Audio bytes: {total_audio_bytes}")
        print(f"Audio duration: {total_audio_bytes / (SAMPLE_RATE * 2):.2f} seconds")
        print(f"Saved file: {OUTPUT_FILE}")
        print("============================================")

    finally:
        pass


if __name__ == "__main__":
    main()
