
import io
import time
import threading
from typing import Callable

import numpy as np
import soundfile as sf

from dotenv import load_dotenv
from deepgram import DeepgramClient
from deepgram.core.events import EventType

from deepgram.speak.v1.types import (
    SpeakV1Text,
    SpeakV1Flushed,
    SpeakV1Warning
)


load_dotenv()


class DeepgramTTSClient:

    def __init__(
        self,
        model="aura-2-asteria-en",
        sample_rate=24000
    ):

        self.model = model
        self.sample_rate = sample_rate

        print("\n[TTS] Initializing Deepgram TTS client...")

        self.client = DeepgramClient()

        print("[TTS] Deepgram TTS client initialized.")

    # ====================================================
    # Existing REST-based audio generation
    # ====================================================

    def generate_audio(self, text):

        if not isinstance(text, str) or not text.strip():
            raise ValueError("TTS text must be a non-empty string.")

        print("\n[TTS] Generating speech with Deepgram...")

        start_time = time.perf_counter()

        chunks = self.client.speak.v1.audio.generate(
            text=text,
            model=self.model,
            encoding="linear16",
            container="wav",
            sample_rate=self.sample_rate
        )

        audio_bytes = b"".join(chunks)

        generation_time = time.perf_counter() - start_time

        if not audio_bytes:
            raise RuntimeError("Deepgram returned empty audio.")

        samples, actual_sample_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32"
        )

        if samples.size == 0:
            raise RuntimeError("Decoded Deepgram audio is empty.")

        samples = np.asarray(samples, dtype=np.float32)

        print("[TTS] Deepgram audio generated.")
        print(f"[TTS] Generation time: {generation_time:.2f} seconds.")
        print(f"[TTS] Sample rate: {actual_sample_rate} Hz.")
        print(f"[TTS] Audio shape: {samples.shape}.")

        return samples, actual_sample_rate

    # ====================================================
    # Streaming TTS using the Deepgram event listener
    # ====================================================

    def generate_audio_stream(
        self,
        text: str,
        on_audio_chunk: Callable[[bytes], None]
    ):

        if not isinstance(text, str) or not text.strip():
            raise ValueError("TTS text must be a non-empty string.")

        if not callable(on_audio_chunk):
            raise TypeError("on_audio_chunk must be callable.")

        print("\n[TTS] Starting Deepgram streaming generation...")

        start_time = time.perf_counter()

        first_audio_time = None
        chunk_count = 0
        total_audio_bytes = 0

        # Thread coordination.
        stream_finished = threading.Event()
        listener_failed = threading.Event()

        listener_error = None
        callback_error = None

        # ------------------------------------------------
        # WebSocket message handler
        # ------------------------------------------------

        def on_message(message):

            nonlocal first_audio_time
            nonlocal chunk_count
            nonlocal total_audio_bytes
            nonlocal callback_error

            try:

                # Audio chunks arrive as raw bytes.
                if isinstance(message, bytes):

                    if first_audio_time is None:

                        first_audio_time = (
                            time.perf_counter() - start_time
                        )

                        print(
                            "[TTS] First audio received after "
                            f"{first_audio_time:.2f} seconds."
                        )

                    chunk_count += 1
                    total_audio_bytes += len(message)

                    # Deliver audio immediately.
                    on_audio_chunk(message)

                # Server confirms that all queued text
                # has finished processing.
                elif isinstance(message, SpeakV1Flushed):

                    print("[TTS] Deepgram stream flushed.")

                    stream_finished.set()

                elif isinstance(message, SpeakV1Warning):

                    print(
                        "[TTS] Deepgram warning:",
                        message
                    )

                else:

                    print(
                        "[TTS] WebSocket event:",
                        type(message).__name__
                    )

            except Exception as error:

                callback_error = error
                listener_failed.set()
                stream_finished.set()

        # ------------------------------------------------
        # WebSocket error handler
        # ------------------------------------------------

        def on_error(error):

            nonlocal listener_error

            listener_error = error

            print(
                "\n[TTS] WebSocket listener error:",
                error
            )

            listener_failed.set()
            stream_finished.set()

        # ------------------------------------------------
        # Listener thread
        # ------------------------------------------------

        def listener_worker(connection):

            nonlocal listener_error

            try:

                connection.start_listening()

            except Exception as error:

                listener_error = error

                print(
                    "\n[TTS] Listener thread failed:",
                    error
                )

                listener_failed.set()
                stream_finished.set()

        # ------------------------------------------------
        # Open connection and start receiving
        # ------------------------------------------------

        with self.client.speak.v1.connect(
            model=self.model,
            encoding="linear16",
            sample_rate=self.sample_rate
        ) as connection:

            # Register event handlers before listening.
            connection.on(
                EventType.MESSAGE,
                on_message
            )

            connection.on(
                EventType.ERROR,
                on_error
            )

            # Start the receiver in a separate thread.
            listener_thread = threading.Thread(
                target=listener_worker,
                args=(connection,),
                name="Deepgram-TTS-Listener",
                daemon=True
            )

            listener_thread.start()

            # Give the listener a moment to start.
            if not listener_thread.is_alive():

                raise RuntimeError(
                    "Deepgram listener thread failed to start."
                )

            # Send text after the listener is running.
            connection.send_text(
                SpeakV1Text(text=text)
            )

            connection.send_flush()

            print("[TTS] Text submitted to Deepgram.")

            # Wait until Deepgram confirms completion.
            completed = stream_finished.wait(
                timeout=90.0
            )

            if not completed:

                raise TimeoutError(
                    "Deepgram TTS did not finish within 90 seconds."
                )

            if callback_error is not None:

                raise RuntimeError(
                    "Audio callback failed."
                ) from callback_error

            if listener_failed.is_set():

                raise RuntimeError(
                    "Deepgram WebSocket listener failed."
                ) from listener_error

            if chunk_count == 0:

                raise RuntimeError(
                    "Deepgram completed without returning audio."
                )

            # Close the WebSocket after successful flushing.
            connection.send_close()

            listener_thread.join(timeout=5.0)

        # ------------------------------------------------
        # Streaming statistics
        # ------------------------------------------------

        total_time = time.perf_counter() - start_time

        audio_duration = (
            total_audio_bytes / (self.sample_rate * 2)
        )

        statistics = {
            "first_audio_seconds": first_audio_time,
            "total_generation_seconds": total_time,
            "audio_chunks": chunk_count,
            "audio_bytes": total_audio_bytes,
            "audio_duration_seconds": audio_duration,
            "sample_rate": self.sample_rate
        }

        print("\n[TTS] Streaming generation completed.")

        print(
            f"[TTS] First audio: "
            f"{first_audio_time:.2f} seconds."
        )

        print(
            f"[TTS] Total generation: "
            f"{total_time:.2f} seconds."
        )

        print(f"[TTS] Audio chunks: {chunk_count}")

        print(
            f"[TTS] Audio duration: "
            f"{audio_duration:.2f} seconds."
        )

        return statistics
