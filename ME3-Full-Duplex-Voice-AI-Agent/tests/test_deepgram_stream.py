
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.speak.v1.types import SpeakV1Text


# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("DEEPGRAM_API_KEY"):
    raise RuntimeError(
        "DEEPGRAM_API_KEY not found. "
        "Check the .env file in the project root."
    )


MODEL = "aura-2-asteria-en"
SAMPLE_RATE = 24000

client = DeepgramClient()

audio_chunks = []
audio_event = threading.Event()
flushed_event = threading.Event()
error_event = threading.Event()


def on_message(message):

    if isinstance(message, bytes):

        audio_chunks.append(message)
        audio_event.set()

        print(
            f"[STREAM] Received audio chunk: "
            f"{len(message)} bytes"
        )

    else:

        event_type = getattr(message, "type", "Unknown")

        print("[STREAM EVENT]", event_type)

        if str(event_type).lower() == "flushed":
            flushed_event.set()


def on_error(error):

    print("[STREAM ERROR]", repr(error))
    error_event.set()


def on_close(event):

    print("[STREAM] Connection closed.")


try:

    print("\nConnecting to Deepgram Speak WebSocket...")

    start_time = time.perf_counter()

    with client.speak.v1.connect(
        model=MODEL,
        encoding="linear16",
        sample_rate=SAMPLE_RATE
    ) as connection:

        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.ERROR, on_error)
        connection.on(EventType.CLOSE, on_close)

        connection.start_listening()

        print("[STREAM] Connected.")

        connection.send_text(
            SpeakV1Text(
                text=(
                    "Hello! This is a streaming text to "
                    "speech test using Deepgram."
                )
            )
        )

        connection.send_flush()

        print("[STREAM] Text submitted.")

        if not audio_event.wait(timeout=30):
            raise TimeoutError(
                "No audio received within 30 seconds."
            )

        first_audio_time = time.perf_counter() - start_time

        print(
            f"\n[RESULT] Time to first audio: "
            f"{first_audio_time:.2f} seconds."
        )

        if not flushed_event.wait(timeout=30):
            raise TimeoutError(
                "Did not receive the Flushed event."
            )

    total_bytes = sum(len(chunk) for chunk in audio_chunks)

    print("\n========== STREAMING RESULTS ==========")
    print("Audio chunks:", len(audio_chunks))
    print("Total audio bytes:", total_bytes)
    print("Error detected:", error_event.is_set())
    print("=======================================")

finally:

    client.close()
