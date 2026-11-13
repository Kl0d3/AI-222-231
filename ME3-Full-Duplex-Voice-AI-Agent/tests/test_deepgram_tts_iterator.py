
import time
import wave
from pathlib import Path

import numpy as np

from dotenv import load_dotenv
from deepgram import DeepgramClient

from deepgram.speak.v1.types import (
    SpeakV1Text,
    SpeakV1Flushed,
)


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "tests" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "deepgram_iterator_diagnostic.wav"

MODEL = "aura-2-asteria-en"
SAMPLE_RATE = 24000

TEST_TEXT = (
    "Hello! I am your voice assistant. "
    "How can I help you today?"
)


# ============================================================
# Main Test
# ============================================================

def main():

    print("\nInitializing Deepgram client...")

    client = DeepgramClient()

    audio_chunks = []
    arrival_times = []

    queue_sizes = []
    paused_states = []

    start_time = time.perf_counter()

    print("\nConnecting to Deepgram...")

    with client.speak.v1.connect(
        model=MODEL,
        encoding="linear16",
        sample_rate=SAMPLE_RATE,
    ) as connection:

        websocket = connection._websocket

        assembler = websocket.recv_messages

        print("Connected.")

        print("\n========== QUEUE CONFIGURATION ==========")
        print("Queue type:", type(assembler.frames).__name__)
        print("High-water mark:", assembler.high)
        print("Low-water mark:", assembler.low)
        print("=========================================\n")

        connection.send_text(
            SpeakV1Text(text=TEST_TEXT)
        )

        connection.send_flush()

        print("Text submitted.\n")

        for message in connection:

            current_time = time.perf_counter()

            queue_size = assembler.frames.qsize()
            paused = assembler.paused

            queue_sizes.append(queue_size)
            paused_states.append(paused)

            if isinstance(message, bytes):

                arrival_times.append(current_time)
                audio_chunks.append(message)

                print(
                    f"[AUDIO] Chunk {len(audio_chunks)} | "
                    f"{len(message)} bytes | "
                    f"Time: {current_time - start_time:.3f}s | "
                    f"Queue: {queue_size} | "
                    f"Paused: {paused}"
                )

            elif isinstance(message, SpeakV1Flushed):

                print("\nDeepgram stream flushed.")

                break

            else:

                print(
                    f"[EVENT] {type(message).__name__} | "
                    f"Queue: {queue_size} | "
                    f"Paused: {paused}"
                )

        connection.send_close()

    total_time = time.perf_counter() - start_time

    if not audio_chunks:
        raise RuntimeError("No audio chunks received.")

    # ========================================================
    # Save WAV
    # ========================================================

    audio_data = b"".join(audio_chunks)

    with wave.open(str(OUTPUT_FILE), "wb") as wav_file:

        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)

        wav_file.writeframes(audio_data)

    # ========================================================
    # Timing Analysis
    # ========================================================

    intervals = np.diff(arrival_times)

    total_audio_duration = (
        len(audio_data) / (SAMPLE_RATE * 2)
    )

    delivery_period = (
        arrival_times[-1] - arrival_times[0]
        if len(arrival_times) > 1
        else 0
    )

    print("\n========== ITERATOR DIAGNOSTIC RESULTS ==========")

    print(f"Total chunks: {len(audio_chunks)}")
    print(f"Total audio duration: {total_audio_duration:.2f}s")
    print(f"Total generation time: {total_time:.2f}s")
    print(f"Audio delivery period: {delivery_period:.2f}s")

    if len(intervals) > 0:

        print("\n--- Chunk arrival intervals ---")

        print(f"Average: {np.mean(intervals) * 1000:.2f} ms")
        print(f"Median: {np.median(intervals) * 1000:.2f} ms")
        print(f"Maximum: {np.max(intervals) * 1000:.2f} ms")

        print(
            "Intervals above 50 ms:",
            np.sum(intervals > 0.05)
        )

        print(
            "Intervals above 100 ms:",
            np.sum(intervals > 0.1)
        )

        print(
            "Intervals above 250 ms:",
            np.sum(intervals > 0.25)
        )

    if delivery_period > 0:

        delivery_speed = (
            total_audio_duration / delivery_period
        )

        print(
            "\nAudio delivery speed:",
            f"{delivery_speed:.2f}x real time"
        )

    # ========================================================
    # Queue Analysis
    # ========================================================

    print("\n--- Receive Queue Analysis ---")

    print("Samples:", len(queue_sizes))
    print("Maximum observed queue size:", max(queue_sizes))
    print("Average observed queue size:", f"{np.mean(queue_sizes):.2f}")

    print(
        "Observations while paused:",
        sum(paused_states)
    )

    print(
        "Percentage of observations while paused:",
        f"{np.mean(paused_states) * 100:.2f}%"
    )

    print(
        "Queue above high-water mark:",
        sum(size > assembler.high for size in queue_sizes)
    )

    # ========================================================
    # Completion
    # ========================================================

    print("\nSaved WAV:", OUTPUT_FILE)

    print("=================================================")


if __name__ == "__main__":
    main()
