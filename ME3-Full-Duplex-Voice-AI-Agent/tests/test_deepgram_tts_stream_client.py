
import wave
import time
from pathlib import Path

import numpy as np

from src.tts.deepgram_client import DeepgramTTSClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "tests" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "deepgram_client_stream_test.wav"

TEST_TEXT = (
    "Hello! I am your voice assistant. "
    "How can I help you today?"
)


def main():

    print("\nInitializing Deepgram TTS client...")

    tts = DeepgramTTSClient()

    audio_chunks = []
    chunk_arrival_times = []
    chunk_sizes = []

    test_start = time.perf_counter()

    def receive_audio_chunk(chunk):

        arrival_time = time.perf_counter() - test_start

        audio_chunks.append(chunk)
        chunk_arrival_times.append(arrival_time)
        chunk_sizes.append(len(chunk))

        print(
            f"[CALLBACK] Chunk {len(audio_chunks):03d} | "
            f"Time: {arrival_time:.3f}s | "
            f"Size: {len(chunk)} bytes"
        )

    print("\nStarting streaming test...")

    statistics = tts.generate_audio_stream(
        text=TEST_TEXT,
        on_audio_chunk=receive_audio_chunk
    )

    test_end = time.perf_counter()

    if not audio_chunks:
        raise RuntimeError("No audio chunks received.")

    audio_data = b"".join(audio_chunks)

    # ================================================
    # Save WAV
    # ================================================

    with wave.open(str(OUTPUT_FILE), "wb") as wav_file:

        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(tts.sample_rate)
        wav_file.writeframes(audio_data)

    # ================================================
    # Analyze chunk arrival intervals
    # ================================================

    arrival_intervals = np.diff(chunk_arrival_times)

    chunk_audio_durations = (
        np.array(chunk_sizes, dtype=np.float64)
        / (tts.sample_rate * 2)
    )

    total_audio_duration = (
        len(audio_data) / (tts.sample_rate * 2)
    )

    total_test_time = test_end - test_start

    first_audio_time = chunk_arrival_times[0]

    last_audio_time = chunk_arrival_times[-1]

    audio_delivery_time = last_audio_time - first_audio_time

    # ================================================
    # Print results
    # ================================================

    print("\n========== TEST RESULTS ==========")

    for key, value in statistics.items():
        print(f"{key}: {value}")

    print("\n========== CHUNK TIMING ANALYSIS ==========")

    print(f"Total chunks: {len(audio_chunks)}")

    print(
        f"First audio arrival: "
        f"{first_audio_time:.3f} seconds"
    )

    print(
        f"Last audio arrival: "
        f"{last_audio_time:.3f} seconds"
    )

    print(
        f"Audio delivery period: "
        f"{audio_delivery_time:.3f} seconds"
    )

    print(
        f"Total test time: "
        f"{total_test_time:.3f} seconds"
    )

    print(
        f"Total audio duration: "
        f"{total_audio_duration:.3f} seconds"
    )

    if len(arrival_intervals) > 0:

        print("\n--- Arrival intervals ---")

        print(
            f"Average: "
            f"{np.mean(arrival_intervals) * 1000:.2f} ms"
        )

        print(
            f"Median: "
            f"{np.median(arrival_intervals) * 1000:.2f} ms"
        )

        print(
            f"Maximum: "
            f"{np.max(arrival_intervals) * 1000:.2f} ms"
        )

        print(
            f"Intervals above 50 ms: "
            f"{np.sum(arrival_intervals > 0.050)}"
        )

        print(
            f"Intervals above 100 ms: "
            f"{np.sum(arrival_intervals > 0.100)}"
        )

        print(
            f"Intervals above 250 ms: "
            f"{np.sum(arrival_intervals > 0.250)}"
        )

    print("\n--- Audio chunk sizes ---")

    print(
        f"Average chunk duration: "
        f"{np.mean(chunk_audio_durations) * 1000:.2f} ms"
    )

    print(
        f"Smallest chunk duration: "
        f"{np.min(chunk_audio_durations) * 1000:.2f} ms"
    )

    print(
        f"Largest chunk duration: "
        f"{np.max(chunk_audio_durations) * 1000:.2f} ms"
    )

    if audio_delivery_time > 0:

        delivery_speed = (
            total_audio_duration / audio_delivery_time
        )

        print(
            f"\nAudio delivery speed: "
            f"{delivery_speed:.2f}x real time"
        )

    print("\n==========================================")

    print(f"\nSaved audio: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
