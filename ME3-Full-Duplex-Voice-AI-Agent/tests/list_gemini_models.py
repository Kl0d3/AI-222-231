
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# Locate project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load API key
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY was not found.")


client = genai.Client(api_key=api_key)

try:
    print("Checking available Gemini models...\n")

    models = client.models.list()

    transcription_models = []

    for model in models:
        name = getattr(model, "name", "") or ""
        display_name = getattr(model, "display_name", "") or ""

        if "transcri" in name.lower() or "transcri" in display_name.lower():
            transcription_models.append(model)

    if transcription_models:
        print("Transcription-related models found:\n")

        for model in transcription_models:
            print("Model ID:", model.name)
            print("Display name:", model.display_name)
            print("-" * 40)

    else:
        print("No transcription-related models found.")
        print("The model may not appear in the standard model listing.")

finally:
    client.close()
