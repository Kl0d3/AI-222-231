
import asyncio
import queue
import threading

import numpy as np
from scipy.io import wavfile

from src.audio.deepgram_flux import DeepgramFluxStreamer


# Configuration
AUDIO_FILE = "tests/audio/sample3_16k.wav"

SAMPLE_RATE = 16000
FRAME_SIZE = 512

FRAME_DURATION = FRAME_SIZE / SAMPLE_RATE


def load_audio():

    print("Loading audio...")

    sample_rate, audio = wavfile.read(AUDIO_FILE)

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz, got {sample_rate} Hz"
        )

    if audio.ndim != 1:
        raise ValueError(
            "Expected mono audio."
        )

    if audio.dtype != np.int16:
        raise ValueError(
            f"Expected int16 audio, got {audio.dtype}"
        )

    # Convert int16 audio into float32.
    # This matches the format produced by sounddevice.
    audio = audio.astype(np.float32) / 32768.0

    print("Audio loaded successfully!")
    print("Sample rate:", sample_rate)
    print("Duration:", len(audio) / sample_rate, "seconds")
    print("Total samples:", len(audio))

    return audio


async def feed_audio_in_real_time(audio, audio_queue, stop_event):

    total_frames = 0

    print("\nFeeding audio at real-time speed...")

    for start in range(0, len(audio), FRAME_SIZE):

        frame = audio[start:start + FRAME_SIZE]

        # Match the shape produced by sounddevice:
        # (samples, channels)
        frame = frame.reshape(-1, 1)

        audio_queue.put(frame)

        total_frames += 1

        # Simulate the time taken by the microphone
        # to capture this frame.
        frame_duration = len(frame) / SAMPLE_RATE

        await asyncio.sleep(frame_duration)

    # Tell the streamer that no more audio is coming.
    stop_event.set()

    print("\nAudio feeding completed.")
    print("Total frames fed:", total_frames)

    return total_frames


async def main():

    audio = load_audio()

    audio_queue = queue.Queue()

    stop_event = threading.Event()

    streamer = DeepgramFluxStreamer()

    print("\nStarting Deepgram Flux module test...")

    # Start the streamer first.
    streamer_task = asyncio.create_task(
        streamer.run(
            audio_queue=audio_queue,
            stop_event=stop_event
        )
    )

    # Wait until the Deepgram connection is established.
    while streamer.connection is None:

        if streamer_task.done():
            await streamer_task

        await asyncio.sleep(0.01)

    print("\nDeepgram connection is ready.")

    # Feed audio while the streamer consumes it.
    total_frames = await feed_audio_in_real_time(
        audio=audio,
        audio_queue=audio_queue,
        stop_event=stop_event
    )

    # Wait for the streamer to finish.
    await streamer_task

    print("\n========== TEST SUMMARY ==========")

    print(
        "Total frames fed:",
        total_frames
    )

    print(
        "Total chunks sent:",
        streamer.total_chunks_sent
    )

    print(
        "Remaining audio frames:",
        audio_queue.qsize()
    )

    # Collect the Flux events received during streaming.
    events = []

    while not streamer.event_queue.empty():
        events.append(streamer.event_queue.get())

    print(
        "Total Flux events received:",
        len(events)
    )

    print("\n========== RECEIVED EVENT TYPES ==========")

    for event in events:

        event_value = event.get("event")

        print(
            "Event:",
            repr(event_value),
            "| Type:",
            type(event_value).__name__
        )

    # Find the final transcript.
    final_transcript = None

    for event in events:

        if event.get("event") == "EndOfTurn":

            final_transcript = event.get("transcript")

    print("\n========== FINAL TRANSCRIPT ==========")

    if final_transcript:
        print(final_transcript)
    else:
        print("No EndOfTurn transcript was received.")

    print("\nTest completed.")


if __name__ == "__main__":
    asyncio.run(main())
