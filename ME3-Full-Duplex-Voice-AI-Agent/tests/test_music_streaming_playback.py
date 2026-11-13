
import time

import numpy as np

from src.mcp_servers.music.playback_engine import PlaybackEngine


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 24000
CHANNELS = 2

DURATION_SECONDS = 7

INITIAL_BUFFER_SECONDS = 3.0
MAX_BUFFER_SECONDS = 12.0

INITIAL_VOLUME = 0.30
DUCK_FACTOR = 0.20


# ============================================================
# GENERATE TEST AUDIO
# ============================================================

def generate_test_audio():

    print("\nGenerating test audio...")

    total_frames = SAMPLE_RATE * DURATION_SECONDS

    time_axis = (
        np.arange(total_frames) / SAMPLE_RATE
    )

    signal = (
        0.3 * np.sin(2 * np.pi * 440 * time_axis)
    ).astype(np.float32)

    audio = np.column_stack(
        (signal, signal)
    )

    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS}")
    print(f"Duration: {DURATION_SECONDS} seconds")
    print(f"Shape: {audio.shape}")

    return audio


# ============================================================
# WAIT FOR PLAYBACK STATE
# ============================================================

def wait_for_state(
    engine,
    expected_state,
    timeout=5
):

    start_time = time.monotonic()

    while time.monotonic() - start_time < timeout:

        status = engine.get_status()

        if status["state"] == expected_state:
            return status

        time.sleep(0.05)

    raise TimeoutError(
        f"Playback did not reach state: {expected_state}. "
        f"Last status: {engine.get_status()}"
    )


# ============================================================
# MAIN STREAMING PLAYBACK TEST
# ============================================================

def main():

    print("=" * 60)
    print("STREAMING PLAYBACK ENGINE TEST")
    print("=" * 60)

    engine = PlaybackEngine(
        fade_duration=0.25
    )

    try:

        audio = generate_test_audio()

        # ----------------------------------------------------
        # 1. Configure volume
        # ----------------------------------------------------

        print("\n1. Setting initial music volume...")

        engine.set_volume(INITIAL_VOLUME)

        print(engine.get_status())

        # ----------------------------------------------------
        # 2. Initialize streaming playback
        # ----------------------------------------------------

        print("\n2. Initializing streaming playback...")

        engine.start_streaming(
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            initial_buffer_seconds=INITIAL_BUFFER_SECONDS,
            max_buffer_seconds=MAX_BUFFER_SECONDS
        )

        print("Streaming playback initialized.")
        print(engine.get_status())

        # ----------------------------------------------------
        # 3. Supply first 2.5 seconds
        # ----------------------------------------------------

        print("\n3. Supplying first 2.5 seconds of audio...")

        first_end = int(
            2.5 * SAMPLE_RATE
        )

        engine.add_audio_chunk(
            audio[:first_end]
        )

        time.sleep(0.3)

        status = engine.get_status()

        print("Playback status:")
        print(status)

        if status["state"] != "buffering":

            raise AssertionError(
                "Playback started before the 3-second "
                "initial buffer was reached."
            )

        print("Initial buffering behavior verified.")

        # ----------------------------------------------------
        # 4. Complete the initial buffer
        # ----------------------------------------------------

        print("\n4. Supplying another 0.75 seconds...")

        second_end = int(
            3.25 * SAMPLE_RATE
        )

        engine.add_audio_chunk(
            audio[first_end:second_end]
        )

        status = wait_for_state(
            engine,
            "playing",
            timeout=3
        )

        print("Playback started.")
        print(status)

        # ----------------------------------------------------
        # 5. Test pause
        # ----------------------------------------------------

        print("\n5. Testing pause...")

        engine.pause()

        time.sleep(0.5)

        paused_status = engine.get_status()

        print(paused_status)

        if paused_status["state"] != "paused":

            raise AssertionError(
                "Playback did not enter paused state."
            )

        # ----------------------------------------------------
        # 6. Test resume
        # ----------------------------------------------------

        print("\n6. Testing resume...")

        engine.resume()

        wait_for_state(
            engine,
            "playing",
            timeout=3
        )

        print(engine.get_status())

        # ----------------------------------------------------
        # 7. Supply remaining audio progressively
        # ----------------------------------------------------

        print("\n7. Supplying remaining audio chunks...")

        chunk_frames = int(
            0.25 * SAMPLE_RATE
        )

        for position in range(
            second_end,
            len(audio),
            chunk_frames
        ):

            chunk = audio[
                position:position + chunk_frames
            ]

            engine.add_audio_chunk(chunk)

            time.sleep(0.05)

        print("Remaining audio supplied.")

        print("Current playback status:")
        print(engine.get_status())

        # ----------------------------------------------------
        # 8. Signal end of stream
        # ----------------------------------------------------

        print("\n8. Signaling end of audio stream...")

        engine.finish_streaming()

        print(engine.get_status())

        # ----------------------------------------------------
        # 9. Test ducking
        # ----------------------------------------------------

        print("\n9. Testing music ducking...")

        engine.duck(
            factor=DUCK_FACTOR
        )

        print(engine.get_status())

        time.sleep(0.5)

        engine.unduck()

        print("Music volume restored.")
        print(engine.get_status())

        # ----------------------------------------------------
        # 10. Wait for playback completion
        # ----------------------------------------------------

        print("\n10. Waiting for playback to finish...")

        final_status = wait_for_state(
            engine,
            "finished",
            timeout=10
        )

        print("Playback completed.")
        print(final_status)

        print("\nStreaming playback test passed.")

    except Exception as error:

        print("\nStreaming playback test failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error: {error}")

    finally:

        print("\nStopping playback engine...")

        engine.stop()

        print(engine.get_status())


if __name__ == "__main__":
    main()
