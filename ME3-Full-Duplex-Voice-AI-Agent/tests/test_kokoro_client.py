
from src.tts.kokoro_client import KokoroTTSClient


def main():

    tts = None

    try:
        # Initialize Kokoro.
        tts = KokoroTTSClient()

        # Sample text.
        text = (
            "Hello! I am your voice assistant. "
            "I am currently testing my text to speech system."
        )

        print("\nGenerating speech...")

        # Generate and save the audio.
        output_path = tts.save_audio(
            text,
            "outputs/kokoro_test.wav"
        )

        print("\n========== TTS TEST SUCCESS ==========")
        print(f"Output file: {output_path}")
        print("======================================")

    except Exception as error:
        print(f"\nKokoro TTS test failed: {error}")

    finally:
        if tts is not None:
            del tts


if __name__ == "__main__":
    main()
