
import re
import queue
import threading
import time
import json

import numpy as np

from src.llm.gemini_client import GeminiClient
from src.tts.deepgram_client import DeepgramTTSClient
from src.tts.kokoro_client import KokoroTTSClient
from src.audio.audio_playback import AudioPlayback
from src.agent.speech_feedback import SpeechFeedbackManager


class ConversationCoordinator:

    AFFIRMATIVE_CONFIRMATIONS = {
        "yes",
        "yes please",
        "yes go ahead",
        "yes do it",
        "go ahead",
        "please go ahead",
        "confirm",
        "confirmed",
        "do it",
        "proceed",
        "please proceed",
    }

    NEGATIVE_CONFIRMATIONS = {
        "no",
        "no thanks",
        "no thank you",
        "cancel",
        "cancel it",
        "please cancel",
        "never mind",
        "nevermind",
        "dont do that",
        "do not do that",
        "i changed my mind",
    }

    def __init__(
        self,
        gemini_client=None,
        tts_client=None,
        playback_client=None,
        deepgram_tts_client=None,
        mcp_manager=None
    ):

        self.feedback_manager = SpeechFeedbackManager()

        if gemini_client is not None:
            self.gemini_client = gemini_client

            self.gemini_client.feedback_callback = (
                self.handle_feedback_event
            )

        else:
            self.gemini_client = GeminiClient(
                mcp_manager=mcp_manager,
                feedback_callback=self.handle_feedback_event
            )

        self.deepgram_tts_client = (
            deepgram_tts_client
            if deepgram_tts_client is not None
            else DeepgramTTSClient()
        )

        self.kokoro_client = tts_client

        self.playback_client = (
            playback_client
            if playback_client is not None
            else AudioPlayback()
        )

        self.audio_queue = queue.Queue()

        self.tts_error = None
        self.playback_error = None

        self.interruption_event = threading.Event()

        self.speech_lock = threading.Lock()
        self.playback_state_lock = threading.Lock()

        # Tracks the complete lifetime of assistant speech,
        # including TTS generation and audio playback.
        self.speech_output_active = threading.Event()

        self.sample_rate = 24000

        self.last_chunk_time = None
        self.chunk_intervals = []

    # ====================================================
    # Feedback event routing
    # ====================================================

    def handle_feedback_event(self, event, tool_name=None):

        try:

            if event == "processing_started":

                self.feedback_manager.notify_processing_started()

            elif event == "tool_started":

                self.feedback_manager.notify_tool_started(
                    tool_name
                )

        except Exception as error:

            print(
                "\n[FEEDBACK MANAGER ERROR]",
                error
            )

    # ====================================================
    # Lazy Kokoro initialization
    # ====================================================

    def get_kokoro_client(self):

        if self.kokoro_client is None:

            print(
                "\n[TTS] Initializing local Kokoro fallback..."
            )

            self.kokoro_client = KokoroTTSClient()

        return self.kokoro_client

    # ====================================================
    # Text normalization for TTS
    # ====================================================

    def clean_text_for_tts(self, text):

        if not isinstance(text, str):
            raise TypeError("Speech text must be a string.")

        text = re.sub(
            r"```[\s\S]*?```",
            " ",
            text
        )

        text = re.sub(
            r"(?m)^\s{0,3}#{1,6}\s*",
            "",
            text
        )

        text = re.sub(
            r"(\*\*\*|\*\*|\*|__|_|~~)",
            "",
            text
        )

        text = text.replace("`", "")

        text = re.sub(
            r"\[([^\]]+)\]\([^)]+\)",
            r"\1",
            text
        )

        text = re.sub(
            r"(?m)^\s*>\s?",
            "",
            text
        )

        text = re.sub(
            r"(?m)^\s*[-+]\s+",
            "",
            text
        )

        text = re.sub(
            r"(?m)^\s*\d+[.)]\s+",
            "",
            text
        )

        text = re.sub(
            r"(?m)^\s*[-*_]{3,}\s*$",
            "",
            text
        )

        text = "\n".join(
            line.strip()
            for line in text.splitlines()
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()

    # ====================================================
    # Sentence splitting
    # ====================================================

    def split_into_sentences(self, text):

        if not text:
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    # ====================================================
    # Confirmation classification
    # ====================================================

    def classify_confirmation(self, transcript):

        if not isinstance(transcript, str):
            return None

        normalized = transcript.lower().strip()

        normalized = normalized.replace("’", "'")
        normalized = normalized.replace("'", "")

        normalized = re.sub(
            r"[^\w\s]",
            " ",
            normalized
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized
        ).strip()

        if normalized in self.AFFIRMATIVE_CONFIRMATIONS:
            return "yes"

        if normalized in self.NEGATIVE_CONFIRMATIONS:
            return "no"

        if normalized.startswith("no "):

            remainder = normalized[3:].strip()

            if remainder in self.NEGATIVE_CONFIRMATIONS:
                return "no"

        return None

    # ====================================================
    # Confirmed operation result handling
    # ====================================================

    def process_confirmed_action(self):

        print(
            "\n[CONFIRMATION] User explicitly confirmed "
            "the pending operation."
        )

        try:

            result = self.gemini_client.execute_pending_action()

        except Exception as error:

            print(
                "\n[CONFIRMATION ERROR] Operation execution failed:",
                error
            )

            result = {
                "success": False,
                "error": str(error)
            }

        result_text = json.dumps(
            result,
            ensure_ascii=False,
            default=str
        )

        prompt = (
            "The user explicitly confirmed the previously "
            "proposed operation. The application has now "
            "attempted that operation.\n\n"
            "Here is the actual result returned by the MCP "
            "operation:\n"
            f"{result_text}\n\n"
            "Respond naturally and concisely to the user. "
            "Report the actual outcome honestly. Do not claim "
            "success unless the result confirms success. "
            "Do not call any tools."
        )

        return self.gemini_client.generate_response(
            prompt,
            allow_tools=False
        )

    # ====================================================
    # Handle user transcript
    # ====================================================

    def process_transcript(self, transcript):

        if not isinstance(transcript, str):
            raise TypeError("Transcript must be a string.")

        transcript = transcript.strip()

        if not transcript:
            raise ValueError("Transcript cannot be empty.")

        pending_action = getattr(
            self.gemini_client,
            "pending_action",
            None
        )

        if pending_action is not None:

            confirmation = self.classify_confirmation(
                transcript
            )

            if confirmation == "yes":

                return self.process_confirmed_action()

            elif confirmation == "no":

                print(
                    "\n[CONFIRMATION] User rejected the operation."
                )

                self.gemini_client.cancel_pending_action()

                return self.gemini_client.generate_response(
                    "The user declined the previously proposed "
                    "operation. Acknowledge the cancellation "
                    "briefly. Do not execute that operation "
                    "and do not call tools.",
                    allow_tools=False
                )

            else:

                print(
                    "\n[CONFIRMATION] Response was not an "
                    "explicit confirmation or rejection."
                )

                print(
                    "[CONFIRMATION] Cancelling the previous "
                    "pending operation and processing the "
                    "new user request."
                )

                self.gemini_client.cancel_pending_action()

        try:

            return self.gemini_client.generate_response(
                transcript
            )

        except Exception:

            self.gemini_client.cancel_pending_action()

            raise

    # ====================================================
    # Queue cleanup
    # ====================================================

    def clear_audio_queue(self):

        cleared_items = 0

        while True:

            try:

                self.audio_queue.get_nowait()

            except queue.Empty:

                break

            else:

                self.audio_queue.task_done()
                cleared_items += 1

        if cleared_items:

            print(
                f"\n[INTERRUPTION] Discarded "
                f"{cleared_items} queued audio chunks."
            )

    # ====================================================
    # Feedback validity and cancellation
    # ====================================================

    def is_feedback_valid(
        self,
        validity_check,
        cancellation_event
    ):

        if cancellation_event is not None:
            if cancellation_event.is_set():
                return False

        if self.interruption_event.is_set():
            return False

        if validity_check is None:
            return True

        try:

            return bool(validity_check())

        except Exception as error:

            print(
                "\n[FEEDBACK VALIDATION ERROR]",
                error
            )

            return False

    def cancel_feedback_playback(self, cancellation_event):

        if cancellation_event is None:
            return

        if cancellation_event.is_set():
            return

        cancellation_event.set()

        print(
            "\n[FEEDBACK] Feedback became stale. "
            "Cancelling its audio."
        )

        self.playback_client.stop()

        self.clear_audio_queue()

    def feedback_validity_monitor(
        self,
        validity_check,
        cancellation_event,
        workers_finished
    ):

        while not workers_finished.is_set():

            if not self.is_feedback_valid(
                validity_check,
                cancellation_event
            ):

                self.cancel_feedback_playback(
                    cancellation_event
                )

                return

            workers_finished.wait(0.05)

    # ====================================================
    # Deepgram streaming callback
    # ====================================================

    def enqueue_deepgram_chunk(
        self,
        audio_bytes,
        validity_check=None,
        cancellation_event=None
    ):

        if self.interruption_event.is_set():
            return

        if cancellation_event is not None:
            if cancellation_event.is_set():
                return

        if validity_check is not None:

            if not self.is_feedback_valid(
                validity_check,
                cancellation_event
            ):

                self.cancel_feedback_playback(
                    cancellation_event
                )

                return

        current_time = time.perf_counter()

        if self.last_chunk_time is not None:

            interval = current_time - self.last_chunk_time
            self.chunk_intervals.append(interval)

        self.last_chunk_time = current_time

        samples = np.frombuffer(
            audio_bytes,
            dtype="<i2"
        )

        if samples.size == 0:
            return

        audio_chunk = (
            samples.astype(np.float32) / 32768.0
        )

        if self.interruption_event.is_set():
            return

        if cancellation_event is not None:
            if cancellation_event.is_set():
                return

        self.audio_queue.put(audio_chunk)

    # ====================================================
    # TTS provider selection
    # ====================================================

    def generate_sentence_audio(
        self,
        sentence,
        validity_check=None,
        cancellation_event=None
    ):

        if not self.is_feedback_valid(
            validity_check,
            cancellation_event
        ):

            if cancellation_event is not None:
                self.cancel_feedback_playback(
                    cancellation_event
                )

            raise InterruptedError(
                "Feedback is no longer valid."
            )

        try:

            print(
                "\n[TTS] Using Deepgram streaming provider."
            )

            self.last_chunk_time = None
            self.chunk_intervals = []

            def on_audio_chunk(audio_bytes):

                self.enqueue_deepgram_chunk(
                    audio_bytes,
                    validity_check=validity_check,
                    cancellation_event=cancellation_event
                )

            statistics = (
                self.deepgram_tts_client.generate_audio_stream(
                    text=sentence,
                    on_audio_chunk=on_audio_chunk
                )
            )

            return "Deepgram", statistics

        except Exception as error:

            print(
                "\n[TTS] Deepgram streaming failed:",
                error
            )

            if self.interruption_event.is_set():

                raise InterruptedError(
                    "Speech interrupted during Deepgram generation."
                )

            if not self.is_feedback_valid(
                validity_check,
                cancellation_event
            ):

                if cancellation_event is not None:
                    self.cancel_feedback_playback(
                        cancellation_event
                    )

                raise InterruptedError(
                    "Feedback became stale during Deepgram generation."
                )

            print(
                "\n[TTS] Switching to local Kokoro fallback."
            )

            kokoro = self.get_kokoro_client()

            if not self.is_feedback_valid(
                validity_check,
                cancellation_event
            ):

                if cancellation_event is not None:
                    self.cancel_feedback_playback(
                        cancellation_event
                    )

                raise InterruptedError(
                    "Feedback became stale before Kokoro generation."
                )

            samples, sample_rate = kokoro.generate_audio(
                sentence
            )

            if sample_rate != self.sample_rate:

                raise ValueError(
                    f"Kokoro sample rate {sample_rate} does not "
                    f"match playback sample rate {self.sample_rate}."
                )

            samples = np.asarray(
                samples,
                dtype=np.float32
            )

            if (
                not self.interruption_event.is_set()
                and self.is_feedback_valid(
                    validity_check,
                    cancellation_event
                )
            ):

                self.audio_queue.put(samples)

            elif cancellation_event is not None:

                self.cancel_feedback_playback(
                    cancellation_event
                )

                raise InterruptedError(
                    "Feedback became stale after Kokoro generation."
                )

            return "Kokoro", None

    # ====================================================
    # TTS generation worker
    # ====================================================

    def tts_generation_worker(
        self,
        sentences,
        validity_check=None,
        cancellation_event=None
    ):

        try:

            total_sentences = len(sentences)

            for index, sentence in enumerate(
                sentences,
                start=1
            ):

                if self.interruption_event.is_set():

                    print(
                        "\n[TTS] Generation cancelled."
                    )

                    break

                if not self.is_feedback_valid(
                    validity_check,
                    cancellation_event
                ):

                    if cancellation_event is not None:
                        self.cancel_feedback_playback(
                            cancellation_event
                        )

                    print(
                        "\n[TTS] Feedback generation cancelled "
                        "because it is no longer valid."
                    )

                    break

                print(
                    f"\n[TTS] Generating sentence "
                    f"{index}/{total_sentences}..."
                )

                provider, statistics = (
                    self.generate_sentence_audio(
                        sentence,
                        validity_check=validity_check,
                        cancellation_event=cancellation_event
                    )
                )

                if provider == "Deepgram" and self.chunk_intervals:

                    intervals = np.asarray(
                        self.chunk_intervals,
                        dtype=np.float64
                    )

                    print("\n[TTS CHUNK TIMING]")

                    print(
                        f"Chunk intervals measured: {len(intervals)}"
                    )

                    print(
                        f"Average interval: "
                        f"{np.mean(intervals) * 1000:.2f} ms"
                    )

                    print(
                        f"Median interval: "
                        f"{np.median(intervals) * 1000:.2f} ms"
                    )

                    print(
                        f"Maximum interval: "
                        f"{np.max(intervals) * 1000:.2f} ms"
                    )

                    print(
                        f"Intervals above 50 ms: "
                        f"{np.sum(intervals > 0.050)}"
                    )

                    print(
                        f"Intervals above 100 ms: "
                        f"{np.sum(intervals > 0.100)}"
                    )

                if self.interruption_event.is_set():

                    print(
                        "\n[TTS] Generation stopped after interruption."
                    )

                    break

                if not self.is_feedback_valid(
                    validity_check,
                    cancellation_event
                ):

                    if cancellation_event is not None:
                        self.cancel_feedback_playback(
                            cancellation_event
                        )

                    break

                print(
                    f"[TTS] Sentence {index} submitted. "
                    f"Provider: {provider}"
                )

        except InterruptedError:

            print(
                "\n[TTS] Generation stopped by interruption "
                "or feedback cancellation."
            )

        except Exception as error:

            self.tts_error = error

            print(
                "\n[TTS GENERATION ERROR]",
                error
            )

        finally:

            self.audio_queue.put(None)

    # ====================================================
    # Audio playback worker
    # ====================================================

    def audio_playback_worker(self):

        try:

            self.playback_client.play_stream(
                self.audio_queue,
                sample_rate=self.sample_rate
            )

        except Exception as error:

            self.playback_error = error

            print(
                "\n[AUDIO PLAYBACK ERROR]",
                error
            )

    # ====================================================
    # Playback interruption lifecycle
    # ====================================================

    def begin_speech_output(self):

        with self.playback_state_lock:

            # Clear an interruption left over from an earlier
            # user utterance before starting new assistant speech.
            if self.interruption_event.is_set():
                self.interruption_event.clear()

            self.playback_client.reset_interruption()

            # Mark output active only after playback state
            # has been successfully reset.
            self.speech_output_active.set()

    def end_speech_output(self):

        with self.playback_state_lock:

            self.speech_output_active.clear()

    def reset_playback_interruption(self):

        with self.playback_state_lock:

            if self.interruption_event.is_set():
                self.interruption_event.clear()

            self.playback_client.reset_interruption()

    # ====================================================
    # Shared TTS playback execution
    # ====================================================

    def play_generated_audio(
        self,
        text,
        label,
        validity_check=None
    ):

        cleaned_text = self.clean_text_for_tts(text)

        if not cleaned_text:

            raise ValueError(
                "No readable text remains for TTS."
            )

        sentences = self.split_into_sentences(
            cleaned_text
        )

        if not sentences:

            raise ValueError(
                "No sentences found for TTS."
            )

        cancellation_event = (
            threading.Event()
            if validity_check is not None
            else None
        )

        if not self.is_feedback_valid(
            validity_check,
            cancellation_event
        ):

            print(
                f"\n[TTS] Skipping {label}: feedback is no longer valid."
            )

            return False

        print(
            f"\n[TTS] Preparing {label}..."
        )

        print(
            f"[TTS] Total sentences: {len(sentences)}"
        )

        self.audio_queue = queue.Queue()

        self.tts_error = None
        self.playback_error = None

        self.last_chunk_time = None
        self.chunk_intervals = []

        playback_thread = threading.Thread(
            target=self.audio_playback_worker,
            name="TTS-Playback-Worker",
            daemon=True
        )

        generation_thread = threading.Thread(
            target=self.tts_generation_worker,
            args=(
                sentences,
                validity_check,
                cancellation_event
            ),
            name="TTS-Generation-Worker",
            daemon=True
        )

        workers_finished = threading.Event()

        monitor_thread = None

        if validity_check is not None:

            monitor_thread = threading.Thread(
                target=self.feedback_validity_monitor,
                args=(
                    validity_check,
                    cancellation_event,
                    workers_finished
                ),
                name="Feedback-Validity-Monitor",
                daemon=True
            )

            monitor_thread.start()

        playback_thread.start()
        generation_thread.start()

        generation_thread.join()
        playback_thread.join()

        workers_finished.set()

        if monitor_thread is not None:
            monitor_thread.join()

        if self.tts_error is not None:

            raise RuntimeError(
                "Speech generation failed."
            ) from self.tts_error

        if self.playback_error is not None:

            raise RuntimeError(
                "Audio playback failed."
            ) from self.playback_error

        if cancellation_event is not None:
            if cancellation_event.is_set():

                print(
                    f"\n[TTS] {label.capitalize()} cancelled "
                    "because it became stale."
                )

                return False

        if self.interruption_event.is_set():

            print(
                f"\n[TTS] {label.capitalize()} interrupted."
            )

            return False

        if not self.is_feedback_valid(
            validity_check,
            cancellation_event
        ):

            if cancellation_event is not None:
                self.cancel_feedback_playback(
                    cancellation_event
                )

            return False

        print(
            f"\n[TTS] {label.capitalize()} completed."
        )

        return True

    # ====================================================
    # Normal assistant response
    # ====================================================

    def speak_response(self, text):

        try:

            with self.speech_lock:

                self.begin_speech_output()

                try:

                    cleaned_text = self.clean_text_for_tts(text)

                    if not cleaned_text:

                        raise ValueError(
                            "No readable text remains for TTS."
                        )

                    sentences = self.split_into_sentences(
                        cleaned_text
                    )

                    if not sentences:

                        raise ValueError(
                            "No sentences found for TTS."
                        )

                    print(
                        "\n[TTS] Preparing streaming speech..."
                    )

                    print(
                        f"[TTS] Total sentences: {len(sentences)}"
                    )

                    print("\n[TTS TEXT]")
                    print(cleaned_text)

                    completed = self.play_generated_audio(
                        cleaned_text,
                        "assistant speech"
                    )

                    if not completed:

                        self.gemini_client.discard_pending_interaction()
                        self.gemini_client.cancel_pending_action()

                    else:

                        self.gemini_client.commit_pending_interaction()

                finally:

                    self.end_speech_output()

        except Exception:

            self.gemini_client.discard_pending_interaction()
            self.gemini_client.cancel_pending_action()

            raise

    # ====================================================
    # Short feedback playback
    # ====================================================

    def speak_feedback(
        self,
        text,
        validity_check=None
    ):

        if not isinstance(text, str):
            raise TypeError("Feedback text must be a string.")

        text = text.strip()

        if not text:
            return False

        try:

            with self.speech_lock:

                # Start a fresh output lifecycle first.
                # This clears an old interruption left by the
                # user's previous utterance.
                self.begin_speech_output()

                try:

                    # Validate after resetting the old
                    # interruption state.
                    if not self.is_feedback_valid(
                        validity_check,
                        None
                    ):

                        print(
                            "\n[FEEDBACK] Skipping stale feedback:",
                            text
                        )

                        return False

                    print(
                        f"\n[FEEDBACK TTS] {text}"
                    )

                    completed = self.play_generated_audio(
                        text,
                        "feedback speech",
                        validity_check=validity_check
                    )

                    return completed

                finally:

                    self.end_speech_output()

        except Exception as error:

            print(
                "\n[FEEDBACK TTS ERROR]",
                error
            )

            return False

    # ====================================================
    # Interrupt speech
    # ====================================================

    def stop_speech(self):

        with self.playback_state_lock:

            # SPEECH_STARTED also occurs when the user begins
            # an ordinary new request. Do not create an
            # interruption unless assistant output is active.
            if not self.speech_output_active.is_set():
                return

            if self.interruption_event.is_set():
                return

            print(
                "\n[INTERRUPTION] User speech detected."
            )

            self.interruption_event.set()

            self.playback_client.stop()

        self.clear_audio_queue()

    # ====================================================
    # Shutdown
    # ====================================================

    def close(self):

        self.stop_speech()

        self.playback_client.close()

        self.gemini_client.close()
