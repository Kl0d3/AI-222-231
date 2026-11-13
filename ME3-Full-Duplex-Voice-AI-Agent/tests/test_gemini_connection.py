
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# Load the project's .env file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# Retrieve the API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY was not found. "
        "Check your project's .env file."
    )


print("API key found.")
print("Connecting to Gemini API...")


# Initialize the Gemini client
client = genai.Client(api_key=api_key)


# Send a request to verify API authentication
try:
    models = client.models.list()
    first_model = next(iter(models), None)

    if first_model is None:
        print("Connection succeeded, but no models were returned.")
    else:
        print("Gemini API authentication successful!")
        print("API is responding correctly.")

except Exception as error:
    print("Gemini API connection failed.")
    print(f"Error: {error}")

finally:
    client.close()
