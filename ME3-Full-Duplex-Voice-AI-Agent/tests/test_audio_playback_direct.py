
import queue
import threading

import numpy as np

from src.audio.audio_playback import AudioPlayback


SAMPLE_RATE = 24000
DURATION = 2
FREQUENCY = 440

audio_queue = queue.Queue()

time_axis = np.arange(
    SAMPLE_RATE * DURATION
) / SAMPLE_RATE

audio = (
    0.2 * np.sin(
        2 * np.pi * FREQUENCY * time_axis
    )
).astype(np.float32)

# Divide the signal into 20 ms chunks.
chunk_size = int(SAMPLE_RATE * 0.020)

for start in range(0, len(audio), chunk_size):

    audio_queue.put(
        audio[start:start + chunk_size]
    )

audio_queue.put(None)

playback = AudioPlayback(device=3)

try:

    print("\nTesting AudioPlayback directly...")

    playback.play_stream(
        audio_queue,
        sample_rate=SAMPLE_RATE
    )

    print("\nDirect playback test completed.")

finally:

    playback.close()
