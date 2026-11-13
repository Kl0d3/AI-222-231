
import threading
import time

from src.tts.kokoro_client import KokoroTTSClient
from src.audio.audio_playback import AudioPlayback


def main():

    tts = None
    playback = None
    playback_thread = None

    try:

        print("\nInitializing Kokoro TTS...")
        tts = KokoroTTSClient()

        print("\nInitializing audio playback...")
        playback = AudioPlayback()

        # First assistant response.
        first_text = (
            "This is the original response. "
            "I will be interrupted after two seconds. "
            "This sentence should not finish."
        )

        print("\nGenerating original response...")

        samples, sample_rate = tts.generate_audio(first_text)

        # Start the original response in a background thread.
        playback_thread = threading.Thread(
            target=playback.play,
            args=(samples, sample_rate)
        )

        print("\nStarting original response...")
        playback_thread.start()

        # Simulate user interruption after two seconds.
        time.sleep(2)

        print("\nSimulating user interruption...")
        playback.stop()

        playback_thread.join(timeout=5)

        if playback_thread.is_alive():
            raise RuntimeError(
                "Original playback did not stop."
            )

        print("\nOriginal response successfully interrupted.")

        # Generate a completely new assistant response.
        second_text = (
            "This is a new assistant response. "
            "Playback has successfully recovered "
            "after the previous interruption."
        )

        print("\nGenerating new response...")

        new_samples, new_sample_rate = tts.generate_audio(
            second_text
        )

        print("\nPlaying the new response...")

        # Reuse the same AudioPlayback instance.
        playback.play(
            new_samples,
            new_sample_rate
        )

        print("\n========== PLAYBACK RECOVERY TEST SUCCESS ==========")

    except Exception as error:

        print("\nPlayback recovery test failed:", error)

    finally:

        if playback is not None:
            playback.close()

        if tts is not None:
            del tts


if __name__ == "__main__":
    main()
