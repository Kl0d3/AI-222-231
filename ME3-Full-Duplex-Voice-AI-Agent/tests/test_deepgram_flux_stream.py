
import asyncio
import os
import wave
from pathlib import Path
from contextlib import suppress

from dotenv import load_dotenv
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType


# Load environment variables from project root
load_dotenv()

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

# 80 ms of 16 kHz mono 16-bit PCM audio
CHUNK_SIZE = 2560

# File location
AUDIO_FILE = (
    Path(__file__).parent
    / "audio"
    / "sample3_16k.wav"
)


async def main():

    # Verify API key exists
    if not os.getenv("DEEPGRAM_API_KEY"):
        raise RuntimeError("DEEPGRAM_API_KEY not found in .env")

    # Read the WAV file
    print("Reading audio file...")

    with wave.open(str(AUDIO_FILE), "rb") as wav:

        if wav.getframerate() != SAMPLE_RATE:
            raise ValueError("Expected 16000 Hz audio.")

        if wav.getnchannels() != CHANNELS:
            raise ValueError("Expected mono audio.")

        if wav.getsampwidth() != SAMPLE_WIDTH:
            raise ValueError("Expected 16-bit PCM audio.")

        audio_data = wav.readframes(wav.getnframes())

    print(f"Audio loaded: {len(audio_data)} bytes")

    # Create Deepgram client
    client = AsyncDeepgramClient()

    # Event used to detect completed transcription
    end_of_turn = asyncio.Event()

    def on_message(message):

        message_type = getattr(message, "type", "Unknown")

        if message_type == "Connected":
            print("\nConnected to Deepgram Flux!")

        elif message_type == "TurnInfo":

            event = getattr(message, "event", "Unknown")
            transcript = getattr(message, "transcript", "")

            print(f"\nFlux event: {event}")

            if transcript:
                print(f"Transcript: {transcript}")

            if event == "EndOfTurn":
                end_of_turn.set()

        else:
            print(f"\nReceived event: {message_type}")

    def on_error(error):
        print(f"\nDeepgram error: {error}")

    # Connect to Deepgram Flux
    print("\nConnecting to Deepgram Flux...")

    async with client.listen.v2.connect(
        model="flux-general-en",
        encoding="linear16",
        sample_rate=SAMPLE_RATE
    ) as connection:

        # Register event handlers
        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.ERROR, on_error)
        connection.on(
            EventType.CLOSE,
            lambda _: print("\nConnection closed.")
        )

        # Start receiving server events in the background
        listener_task = asyncio.create_task(
            connection.start_listening()
        )

        try:

            # Give the connection time to initialize
            await asyncio.sleep(0.5)

            print("\nStreaming audio to Deepgram...")
            print("Chunk duration: 80 ms")

            total_chunks = 0

            # Send audio in 80 ms chunks
            for position in range(0, len(audio_data), CHUNK_SIZE):

                chunk = audio_data[
                    position:position + CHUNK_SIZE
                ]

                await connection.send_media(chunk)

                total_chunks += 1

                # Pace transmission according to actual chunk duration
                chunk_duration = len(chunk) / (
                    SAMPLE_RATE * SAMPLE_WIDTH
                )

                await asyncio.sleep(chunk_duration)

            print(f"\nAudio transmission completed.")
            print(f"Total chunks sent: {total_chunks}")

            # Tell Deepgram that no more audio will be sent
            await connection.send_close_stream()

            print("\nWaiting for final transcription...")

            await asyncio.wait_for(
                end_of_turn.wait(),
                timeout=20
            )

            print("\n========== FLUX TEST COMPLETED ==========")
            print("End-of-turn event received.")
            print("=========================================")

        finally:

            # Stop the background listener
            listener_task.cancel()

            with suppress(asyncio.CancelledError):
                await listener_task


if __name__ == "__main__":
    asyncio.run(main())
