import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Locate project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load API key
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY was not found.")

# Locate test audio
AUDIO_FILE = PROJECT_ROOT / "tests" / "audio" / "sample.wav"

if not AUDIO_FILE.exists():
    raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE}")


def main():
    # Use v1beta for Gemini 3.5 Transcribe
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta")
    )

    try:
        print("Uploading audio to Gemini...")

        audio_file = client.files.upload(file=str(AUDIO_FILE))

        print("Audio uploaded successfully!")
        print("Sending audio for transcription...")

        # Pass audio_transcription_config inside GenerateContentConfig
        response = client.models.generate_content(
            model="gemini-3.5-transcribe",
            contents=[audio_file],
            config=types.GenerateContentConfig(
                audio_transcription_config=types.AudioTranscriptionConfig(
                    language_codes=[],  # Pass [] for auto-detect (85+ languages supported)
                )
            ),
        )

        print("\n========== TRANSCRIPTION ==========")
        print(response.text)
        print("===================================")

    except Exception as error:
        print("\nTranscription failed.")
        print(f"Error: {error}")

    finally:
        client.close()


if __name__ == "__main__":
    main()