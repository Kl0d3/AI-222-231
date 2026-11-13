
import queue
import time

import numpy as np

from src.audio.audio_playback import AudioPlayback


def main():

    audio_queue = queue.Queue()

    playback = AudioPlayback()

    sample_rate = 24000

    # Generate a 440 Hz test tone.
    duration = 2.0
    frequency = 440

    total_samples = int(sample_rate * duration)

    samples = np.arange(total_samples)

    audio = (
        0.2 * np.sin(
            2 * np.pi * frequency * samples / sample_rate
        )
    ).astype(np.float32)

    # Divide audio into smaller chunks.
    chunk_size = 2400

    for start in range(0, len(audio), chunk_size):

        chunk = audio[start:start + chunk_size]

        audio_queue.put(chunk)

    # Signal that all audio chunks have been submitted.
    audio_queue.put(None)

    print("\nStarting streaming playback test...")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Total samples: {len(audio)}")
    print(f"Duration: {duration} seconds")
    print(f"Chunk size: {chunk_size}")

    start_time = time.perf_counter()

    try:

        playback.play_stream(
            audio_queue,
            sample_rate=sample_rate
        )

        elapsed = time.perf_counter() - start_time

        print(f"\nPlayback test completed in {elapsed:.2f} seconds.")

    finally:

        playback.close()


if __name__ == "__main__":
    main()
