
import asyncio
import os
import queue
import threading
import traceback
from contextlib import suppress

import numpy as np
from dotenv import load_dotenv
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType


load_dotenv()


MODEL = "flux-general-en"

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

CHUNK_DURATION_MS = 80

CHUNK_SAMPLES = int(
    SAMPLE_RATE * CHUNK_DURATION_MS / 1000
)

CHUNK_BYTES = CHUNK_SAMPLES * SAMPLE_WIDTH


class DeepgramFluxStreamer:

    def __init__(self):

        self.event_queue = queue.Queue()

        self.client = None
        self.connection = None

        self.total_chunks_sent = 0
        self.total_events_received = 0

        self.close_event = None

        self.connection_ready = threading.Event()
        self.worker_finished = threading.Event()

        self.last_error = None

    # ====================================================
    # Error reporting
    # ====================================================

    def report_error(self, stage, error):

        self.last_error = error

        print(
            f"\n========== DEEPGRAM {stage} ERROR =========="
        )

        print(
            "Exception type:",
            type(error).__name__
        )

        print(
            "Exception repr:",
            repr(error)
        )

        print("\nFull traceback:")

        traceback.print_exc()

        print("============================================")

        self.event_queue.put({
            "type": "ERROR",
            "error": repr(error),
            "stage": stage
        })

    # ====================================================
    # Audio conversion
    # ====================================================

    def convert_to_pcm16(self, audio_frame):

        audio_frame = np.asarray(
            audio_frame,
            dtype=np.float32
        ).flatten()

        audio_frame = np.clip(
            audio_frame,
            -1.0,
            1.0
        )

        audio_int16 = np.rint(
            audio_frame * 32768.0
        )

        audio_int16 = np.clip(
            audio_int16,
            -32768,
            32767
        ).astype("<i2")

        return audio_int16.tobytes()

    # ====================================================
    # Deepgram event handlers
    # ====================================================

    def on_message(self, message):

        message_type = getattr(
            message,
            "type",
            "Unknown"
        )

        event = getattr(
            message,
            "event",
            None
        )

        transcript = getattr(
            message,
            "transcript",
            ""
        )

        event_data = {
            "type": message_type,
            "event": event,
            "transcript": transcript,
            "turn_index": getattr(
                message,
                "turn_index",
                None
            )
        }

        self.event_queue.put(event_data)

        self.total_events_received += 1

        if event:

            event_name = str(event).lower()

            if event_name == "startofturn":

                print(
                    "\n[DEEPGRAM] User started speaking."
                )

            elif event_name == "endofturn":

                print(
                    "\n[DEEPGRAM] User finished speaking."
                )

    def on_error(self, error):

        self.report_error(
            "EVENT",
            error
        )

    # ====================================================
    # Deepgram streaming worker
    # ====================================================

    async def run(self, audio_queue, stop_event):

        listener_task = None

        try:

            if not os.getenv("DEEPGRAM_API_KEY"):

                raise RuntimeError(
                    "DEEPGRAM_API_KEY not found."
                )

            print(
                "\n[DEEPGRAM] Initializing client..."
            )

            self.client = AsyncDeepgramClient()

            self.close_event = asyncio.Event()

            print(
                "\n[DEEPGRAM] Connecting to Flux..."
            )

            async with self.client.listen.v2.connect(
                model=MODEL,
                encoding="linear16",
                sample_rate=SAMPLE_RATE
            ) as connection:

                self.connection = connection

                connection.on(
                    EventType.MESSAGE,
                    self.on_message
                )

                connection.on(
                    EventType.ERROR,
                    self.on_error
                )

                connection.on(
                    EventType.CLOSE,
                    lambda _: self.close_event.set()
                )

                listener_task = asyncio.create_task(
                    connection.start_listening()
                )

                pending_audio = bytearray()

                self.connection_ready.set()

                print(
                    "\n[DEEPGRAM] Connected successfully."
                )

                try:

                    while (
                        not stop_event.is_set()
                        or not audio_queue.empty()
                    ):

                        # Detect listener failure.
                        if listener_task.done():

                            await listener_task

                            raise RuntimeError(
                                "Deepgram listener stopped unexpectedly."
                            )

                        try:

                            audio_frame = await asyncio.to_thread(
                                audio_queue.get,
                                True,
                                0.1
                            )

                        except queue.Empty:

                            continue

                        try:

                            pcm_audio = self.convert_to_pcm16(
                                audio_frame
                            )

                            pending_audio.extend(
                                pcm_audio
                            )

                            while len(pending_audio) >= CHUNK_BYTES:

                                chunk = bytes(
                                    pending_audio[:CHUNK_BYTES]
                                )

                                del pending_audio[:CHUNK_BYTES]

                                await connection.send_media(
                                    chunk
                                )

                                self.total_chunks_sent += 1

                        finally:

                            audio_queue.task_done()

                    # Send remaining audio.
                    if pending_audio:

                        await connection.send_media(
                            bytes(pending_audio)
                        )

                        self.total_chunks_sent += 1

                    print(
                        "\n[DEEPGRAM] Audio streaming stopped."
                    )

                    await connection.send_close_stream()

                    print(
                        "\n[DEEPGRAM] Waiting for connection to close..."
                    )

                    try:

                        await asyncio.wait_for(
                            self.close_event.wait(),
                            timeout=10
                        )

                        print(
                            "\n[DEEPGRAM] Connection closed."
                        )

                    except asyncio.TimeoutError:

                        print(
                            "\n[DEEPGRAM] Warning: Close timeout."
                        )

                finally:

                    if listener_task is not None:

                        if not listener_task.done():

                            listener_task.cancel()

                        with suppress(asyncio.CancelledError):

                            await listener_task

                    self.connection = None

        except Exception as error:

            self.report_error(
                "WORKER",
                error
            )

        finally:

            self.connection_ready.set()

            self.worker_finished.set()

            print(
                "\n[DEEPGRAM] Worker stopped."
            )
