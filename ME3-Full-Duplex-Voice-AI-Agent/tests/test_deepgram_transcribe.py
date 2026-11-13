
import os
from pathlib import Path

from dotenv import load_dotenv
from deepgram import DeepgramClient


# Locate project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("DEEPGRAM_API_KEY")

if not api_key:
    raise RuntimeError("DEEPGRAM_API_KEY was not found.")

# Locate test audio
AUDIO_FILE = PROJECT_ROOT / "tests" / "audio" / "sample3.wav"

if not AUDIO_FILE.exists():
    raise FileNotFoundError(
        f"Audio file not found: {AUDIO_FILE}"
    )


def main():

    client = DeepgramClient(api_key=api_key)

    try:
        print("Reading audio file...")

        with open(AUDIO_FILE, "rb") as audio:
            audio_data = audio.read()

        print("Audio loaded successfully!")
        print("Sending audio to Deepgram...")

        response = client.listen.v1.media.transcribe_file(
            request=audio_data,
            model="nova-3",
            smart_format=True,
        )

        alternative = (
            response.results.channels[0].alternatives[0]
        )

        print("\n========== TRANSCRIPTION ==========")
        print(alternative.transcript)

        print("\n========== OVERALL CONFIDENCE ==========")
        print(alternative.confidence)

        print("\n========== WORD-LEVEL DETAILS ==========")

        for word in alternative.words:
            print(
                f"{word.punctuated_word:<20} "
                f"Start: {word.start:.2f}s | "
                f"End: {word.end:.2f}s | "
                f"Confidence: {word.confidence:.4f}"
            )

        print("\n===================================")

    except Exception as error:
        print("\nTranscription failed.")
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
