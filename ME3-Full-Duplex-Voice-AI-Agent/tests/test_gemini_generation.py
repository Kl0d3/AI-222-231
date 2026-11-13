
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# Load the project's .env file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY was not found.")

client = genai.Client(api_key=api_key)

try:
    print("Sending prompt to Gemini...")

    response = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input="Hello! Introduce yourself in one short sentence."
    )

    print("\n========== GEMINI RESPONSE ==========")
    print(response.output_text)
    print("=====================================")

except Exception as error:
    print("\nGemini text generation failed.")
    print(f"Error: {error}")

finally:
    client.close()
