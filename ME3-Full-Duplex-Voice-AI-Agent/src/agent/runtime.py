
import asyncio
import queue
import threading
import time

import sounddevice as sd

from src.audio.audio_buffer import AudioBuffer
from src.audio.deepgram_flux import DeepgramFluxStreamer
from src.agent.coordinator import ConversationCoordinator
from src.mcp.background_manager import MCPBackgroundManager


class VoiceAgentRuntime:

    def __init__(self):

        print("\nInitializing Voice Agent Runtime...")

        self.audio = AudioBuffer()
        self.deepgram = DeepgramFluxStreamer()

        self.coordinator = None
        self.mcp_manager = None

        self.stop_event = self.audio.stop_event

        # Existing worker threads.
        self.vad_thread = None
        self.deepgram_thread = None

        # Event consumer threads.
        self.event_thread = None
        self.deepgram_event_thread = None

        # Background workers.
        self.processing_thread = None
        self.feedback_thread = None

        # Queue of completed user turns.
        self.completed_turn_queue = queue.Queue()

        # Worker lifecycle.
        self.feedback_stop_event = threading.Event()

        # Processing state.
        self.processing_in_progress = threading.Event()
        self.active_processing_turn_id = None

        # User activity state.
        self.user_speech_active = threading.Event()
        self.user_turn_active = threading.Event()

        self.activity_lock = threading.Lock()

        self.last_user_speech_stopped_at = None
        self.last_backchannel_attempt = 0.0

        # Backchannel timing.
        self.backchannel_pause_seconds = 0.85
        self.backchannel_attempt_interval = 0.5

        # Prevent repeated backchannels during one continuous pause.
        self.backchannel_sent_for_pause = False

        # Acknowledgments should be immediately eligible.
        self.acknowledgment_delay = 0.0

        # Coordinates admission of feedback and normal speech.
        self.speech_admission_lock = threading.Lock()

        self.deepgram_error = None

    # ====================================================
    # Deepgram worker
    # ====================================================

    def deepgram_worker(self):

        try:

            asyncio.run(
                self.deepgram.run(
                    self.audio.flux_audio_queue,
                    self.stop_event
                )
            )

        except Exception as error:

            self.deepgram_error = error

            print(
                "\nDeepgram thread error:",
                error
            )

        finally:

            self.deepgram.connection_ready.set()
            self.deepgram.worker_finished.set()

    # ====================================================
    # Gemini feedback callback
    # ====================================================

    def handle_gemini_feedback(
        self,
        event,
        tool_name=None
    ):

        if self.coordinator is None:
            return

        if event == "tool_started":

            turn_id = self.active_processing_turn_id

            if turn_id is None:
                return

            if not self.is_current_turn(turn_id):
                return

            print(
                "\n[FEEDBACK EVENT] MCP tool started:",
                tool_name
            )

            self.coordinator.feedback_manager.notify_tool_started(
                tool_name
            )

        elif event == "processing_started":

            # Processing is started by the runtime so that
            # the actual user transcript can be supplied.
            pass

    # ====================================================
    # User turn lifecycle
    # ====================================================

    def begin_user_turn_if_needed(self):

        if self.coordinator is None:
            return

        manager = self.coordinator.feedback_manager

        status = manager.get_status()

        if (
            not status["turn_active"]
            or status["user_turn_complete"]
        ):

            manager.begin_turn()

            print(
                "\n[RUNTIME] New user conversation turn started."
            )

        else:

            # The user resumed speaking during the same turn.
            # Remove pending feedback from the previous pause.
            manager.cancel_pending_feedback()

        self.user_turn_active.set()

    # ====================================================
    # Speech event consumer
    # ====================================================

    def speech_event_worker(self):

        while True:

            if (
                self.audio.audio_worker_finished.is_set()
                and self.audio.speech_event_queue.empty()
            ):
                break

            try:

                event = self.audio.speech_event_queue.get(
                    timeout=0.1
                )

            except queue.Empty:
                continue

            try:

                event_type = event["type"]

                print(
                    "\n[VAD EVENT]",
                    event_type
                )

                with self.audio.stats_lock:

                    self.audio.consumed_events += 1

                # ========================================
                # User starts speaking
                # ========================================

                if event_type == "SPEECH_STARTED":

                    self.user_speech_active.set()

                    with self.activity_lock:

                        self.last_user_speech_stopped_at = None
                        self.backchannel_sent_for_pause = False

                    self.begin_user_turn_if_needed()

                    if self.coordinator is not None:

                        print(
                            "\n[BARGE-IN] User speech detected."
                        )

                        self.coordinator.stop_speech()

                # ========================================
                # User stops speaking temporarily
                # ========================================

                elif event_type == "SPEECH_STOPPED":

                    self.user_speech_active.clear()

                    with self.activity_lock:

                        self.last_user_speech_stopped_at = (
                            time.monotonic()
                        )

                        self.backchannel_sent_for_pause = False

                    print(
                        "\n[RUNTIME] User speech paused."
                    )

            except Exception as error:

                print(
                    "\nSpeech event processing error:",
                    error
                )

            finally:

                self.audio.speech_event_queue.task_done()

        print("\nSpeech event worker stopped.")

    # ====================================================
    # Enqueue completed user turn
    # ====================================================

    def process_completed_turn(self, transcript):

        transcript = transcript.strip()

        if not transcript:

            print(
                "\nEmpty transcript. Skipping."
            )

            return

        if self.coordinator is None:
            return

        manager = self.coordinator.feedback_manager

        status = manager.get_status()

        if not status["turn_active"]:

            manager.begin_turn()

            status = manager.get_status()

        turn_id = status["turn_id"]

        print(
            "\n[RUNTIME] Completed turn queued:",
            turn_id
        )

        self.completed_turn_queue.put(
            (transcript, turn_id)
        )

    # ====================================================
    # Check whether a turn is still current
    # ====================================================

    def is_current_turn(self, turn_id):

        if self.coordinator is None:
            return False

        status = self.coordinator.feedback_manager.get_status()

        return (
            status["turn_id"] == turn_id
            and status["turn_active"]
        )

    # ====================================================
    # Finish feedback for a specific turn
    # ====================================================

    def finish_feedback_for_turn(self, turn_id):

        if self.coordinator is None:
            return

        manager = self.coordinator.feedback_manager
        status = manager.get_status()

        if status["turn_id"] != turn_id:
            return

        manager.finish_processing()

    # ====================================================
    # Conversation processing worker
    # ====================================================

    def conversation_processing_worker(self):

        print(
            "\n[PROCESSING] Conversation processing worker started."
        )

        while True:

            try:

                item = self.completed_turn_queue.get(
                    timeout=0.1
                )

            except queue.Empty:

                if self.stop_event.is_set():
                    break

                continue

            if item is None:

                self.completed_turn_queue.task_done()
                break

            transcript, turn_id = item

            try:

                if not self.is_current_turn(turn_id):

                    print(
                        "\n[PROCESSING] Discarding stale queued turn:",
                        turn_id
                    )

                    continue

                self.processing_in_progress.set()
                self.active_processing_turn_id = turn_id

                print(
                    "\n========== PROCESSING USER TURN =========="
                )

                print("Turn ID:", turn_id)
                print("User:", transcript)

                # ========================================
                # Start context-aware feedback
                # ========================================

                print(
                    "\n[FEEDBACK] Starting processing feedback..."
                )

                self.coordinator.feedback_manager.notify_processing_started(
                    transcript
                )

                print(
                    "\nGenerating Gemini response..."
                )

                response = self.coordinator.process_transcript(
                    transcript
                )

                if not self.is_current_turn(turn_id):

                    print(
                        "\n[PROCESSING] Response belongs to an "
                        "outdated turn. Discarding it."
                    )

                    self.coordinator.gemini_client.discard_pending_interaction()
                    self.coordinator.gemini_client.cancel_pending_action()

                    continue

                if not response or not response.strip():

                    print(
                        "\nGemini returned an empty response."
                    )

                    continue

                # ========================================
                # Processing has now completed.
                #
                # Clear this BEFORE waiting for the speech
                # admission lock. This immediately invalidates
                # progress feedback that is still being
                # generated or played.
                # ========================================

                self.processing_in_progress.clear()

                print(
                    "\n========== GEMINI RESPONSE =========="
                )

                print(response)

                print(
                    "===================================="
                )

                self.finish_feedback_for_turn(turn_id)

                # ========================================
                # Admit normal assistant speech
                # ========================================

                with self.speech_admission_lock:

                    if not self.is_current_turn(turn_id):

                        print(
                            "\n[PROCESSING] Turn changed before "
                            "speech playback. Discarding response."
                        )

                        self.coordinator.gemini_client.discard_pending_interaction()
                        self.coordinator.gemini_client.cancel_pending_action()

                        continue

                    print(
                        "\nGenerating assistant speech..."
                    )

                    self.coordinator.speak_response(
                        response
                    )

            except Exception as error:

                print(
                    "\nConversation processing error:",
                    error
                )

            finally:

                self.finish_feedback_for_turn(turn_id)

                self.processing_in_progress.clear()

                if self.active_processing_turn_id == turn_id:
                    self.active_processing_turn_id = None

                if self.is_current_turn(turn_id):

                    self.coordinator.feedback_manager.end_turn()

                    self.user_turn_active.clear()

                    with self.activity_lock:

                        self.last_user_speech_stopped_at = None
                        self.backchannel_sent_for_pause = False

                self.completed_turn_queue.task_done()

        print(
            "\n[PROCESSING] Conversation processing worker stopped."
        )

    # ====================================================
    # Backchannel scheduling
    # ====================================================

    def schedule_backchannel_if_appropriate(self):

        if self.coordinator is None:
            return

        if not self.user_turn_active.is_set():
            return

        if self.user_speech_active.is_set():
            return

        if self.processing_in_progress.is_set():
            return

        with self.activity_lock:

            pause_started_at = (
                self.last_user_speech_stopped_at
            )

            last_attempt = self.last_backchannel_attempt

            already_sent = self.backchannel_sent_for_pause

        if pause_started_at is None:
            return

        if already_sent:
            return

        now = time.monotonic()

        if (
            now - pause_started_at
            < self.backchannel_pause_seconds
        ):
            return

        if (
            now - last_attempt
            < self.backchannel_attempt_interval
        ):
            return

        with self.activity_lock:

            self.last_backchannel_attempt = now

        accepted = (
            self.coordinator.feedback_manager.request_backchannel(
                user_turn_active=self.user_turn_active.is_set(),
                user_speech_active=self.user_speech_active.is_set()
            )
        )

        if accepted:

            with self.activity_lock:

                self.backchannel_sent_for_pause = True

            print(
                "\n[BACKCHANNEL] A conversational response "
                "has been queued."
            )

    # ====================================================
    # Feedback validity
    # ====================================================

    def feedback_is_valid(self, message):

        if self.coordinator is None:
            return False

        status = self.coordinator.feedback_manager.get_status()

        if not status["turn_active"]:
            return False

        if status["turn_id"] != message.turn_id:
            return False

        if message.category == "backchannel":

            if status["user_turn_complete"]:
                return False

            if self.user_speech_active.is_set():
                return False

            if not self.user_turn_active.is_set():
                return False

            if self.processing_in_progress.is_set():
                return False

        elif message.category == "acknowledgment":

            # Before admission, an acknowledgment is only eligible
            # while processing is active. Once admitted, its
            # dedicated callback keeps it valid for the current
            # turn even if Gemini finishes during TTS generation.

            if not self.processing_in_progress.is_set():
                return False

        elif message.category == "progress":

            # Progress feedback is valid only while Gemini/MCP
            # processing is actually active.
            if not self.processing_in_progress.is_set():
                return False

        return True

    # ====================================================
    # Delayed acknowledgment handling
    # ====================================================

    def wait_for_acknowledgment_delay(self, message):

        deadline = (
            time.monotonic()
            + self.acknowledgment_delay
        )

        while time.monotonic() < deadline:

            if self.feedback_stop_event.is_set():
                return False

            if not self.feedback_is_valid(message):
                return False

            time.sleep(0.05)

        return self.feedback_is_valid(message)

    # ====================================================
    # Feedback worker
    # ====================================================

    def feedback_worker(self):

        print(
            "\n[FEEDBACK] Feedback worker started."
        )

        while not self.feedback_stop_event.is_set():

            try:

                self.schedule_backchannel_if_appropriate()

                message = (
                    self.coordinator.feedback_manager.get_next_feedback(
                        timeout=0.1
                    )
                )

                if message is None:

                    time.sleep(0.05)

                    continue

                if not self.feedback_is_valid(message):
                    continue

                if message.category == "acknowledgment":

                    if not self.wait_for_acknowledgment_delay(
                        message
                    ):
                        continue

                # This event marks the point at which the
                # acknowledgment has been admitted for playback.
                acknowledgment_admitted = threading.Event()

                def validity_check(
                    current_message=message,
                    admitted_event=acknowledgment_admitted
                ):

                    if (
                        current_message.category == "acknowledgment"
                        and admitted_event.is_set()
                    ):

                        # An admitted acknowledgment may finish
                        # after processing completes, but never
                        # after its turn becomes outdated.
                        return self.is_current_turn(
                            current_message.turn_id
                        )

                    return self.feedback_is_valid(
                        current_message
                    )

                with self.speech_admission_lock:

                    if not self.feedback_is_valid(message):
                        continue

                    if self.feedback_stop_event.is_set():
                        break

                    if message.category == "acknowledgment":

                        acknowledgment_admitted.set()

                    print(
                        "\n[FEEDBACK] Playing:",
                        message.category,
                        message.text
                    )

                    self.coordinator.speak_feedback(
                        message.text,
                        validity_check=validity_check
                    )

            except Exception as error:

                print(
                    "\n[FEEDBACK WORKER ERROR]",
                    error
                )

                time.sleep(0.1)

        print(
            "\n[FEEDBACK] Feedback worker stopped."
        )

    # ====================================================
    # Deepgram event consumer
    # ====================================================

    def deepgram_event_worker(self):

        while True:

            if (
                self.deepgram.worker_finished.is_set()
                and self.deepgram.event_queue.empty()
            ):
                break

            try:

                event = self.deepgram.event_queue.get(
                    timeout=0.1
                )

            except queue.Empty:
                continue

            try:

                event_type = event.get("event")
                message_type = event.get("type")
                transcript = event.get("transcript", "")

                if message_type == "ERROR":

                    print(
                        "\n[DEEPGRAM ERROR]",
                        event.get("error")
                    )

                elif event_type == "EndOfTurn":

                    print(
                        "\n[DEEPGRAM EVENT] EndOfTurn"
                    )

                    print(
                        "\n========== FINAL USER TRANSCRIPT =========="
                    )

                    print(transcript.strip())

                    print(
                        "============================================"
                    )

                    if self.coordinator is not None:

                        self.coordinator.feedback_manager.complete_user_turn()

                    self.user_turn_active.clear()

                    with self.activity_lock:

                        self.last_user_speech_stopped_at = None
                        self.backchannel_sent_for_pause = False

                    self.process_completed_turn(
                        transcript
                    )

                elif event_type and event_type != "Update":

                    print(
                        "\n[DEEPGRAM EVENT]",
                        event_type
                    )

                    if transcript:
                        print("Transcript:", transcript)

                elif transcript:

                    print(
                        "\n[DEEPGRAM TRANSCRIPT UPDATE]",
                        transcript
                    )

            except Exception as error:

                print(
                    "\nDeepgram event processing error:",
                    error
                )

            finally:

                self.deepgram.event_queue.task_done()

        print(
            "\nDeepgram event consumer stopped."
        )

    # ====================================================
    # Start existing audio workers
    # ====================================================

    def start_workers(self):

        print(
            "\nLoading Silero VAD..."
        )

        self.audio.load_vad_model()

        self.vad_thread = threading.Thread(
            target=self.audio.audio_worker,
            name="Silero-VAD-Worker",
            daemon=True
        )

        self.vad_thread.start()

        self.deepgram_thread = threading.Thread(
            target=self.deepgram_worker,
            name="Deepgram-Worker",
            daemon=True
        )

        self.deepgram_thread.start()

        self.event_thread = threading.Thread(
            target=self.speech_event_worker,
            name="Speech-Event-Worker",
            daemon=True
        )

        self.event_thread.start()

        self.deepgram_event_thread = threading.Thread(
            target=self.deepgram_event_worker,
            name="Deepgram-Event-Worker",
            daemon=True
        )

        self.deepgram_event_thread.start()

    # ====================================================
    # Start background workers
    # ====================================================

    def start_background_workers(self):

        print(
            "\nStarting conversation processing worker..."
        )

        self.processing_thread = threading.Thread(
            target=self.conversation_processing_worker,
            name="Conversation-Processing-Worker",
            daemon=True
        )

        self.processing_thread.start()

        print(
            "\nStarting speech feedback worker..."
        )

        self.feedback_thread = threading.Thread(
            target=self.feedback_worker,
            name="Speech-Feedback-Worker",
            daemon=True
        )

        self.feedback_thread.start()

    # ====================================================
    # Wait for Deepgram
    # ====================================================

    def wait_for_deepgram(self, timeout=30):

        print(
            "\nWaiting for Deepgram connection..."
        )

        connected = self.deepgram.connection_ready.wait(
            timeout=timeout
        )

        if not connected:

            raise TimeoutError(
                "Deepgram connection timed out."
            )

        if self.deepgram.worker_finished.is_set():

            raise RuntimeError(
                "Deepgram worker stopped before connecting."
            )

        if self.deepgram_error:

            raise RuntimeError(
                "Deepgram initialization failed."
            ) from self.deepgram_error

        print(
            "\nDeepgram is ready!"
        )

    # ====================================================
    # Initialize MCP manager
    # ====================================================

    def initialize_mcp_manager(self):

        print(
            "\nInitializing MCP Background Manager..."
        )

        self.mcp_manager = MCPBackgroundManager()

        self.mcp_manager.start()

        print(
            "\nMCP Background Manager initialized!"
        )

    # ====================================================
    # Initialize conversation coordinator
    # ====================================================

    def initialize_coordinator(self):

        print(
            "\nInitializing Conversation Coordinator..."
        )

        self.coordinator = ConversationCoordinator(
            mcp_manager=self.mcp_manager
        )

        self.coordinator.gemini_client.feedback_callback = (
            self.handle_gemini_feedback
        )

        print(
            "\nGemini feedback callback connected."
        )

        print(
            "\nConversation Coordinator initialized!"
        )

    # ====================================================
    # Microphone
    # ====================================================

    def start_microphone(self):

        print(
            "\nStarting microphone..."
        )

        with sd.InputStream(
            samplerate=self.audio.sample_rate,
            channels=self.audio.channels,
            blocksize=self.audio.block_size,
            dtype="float32",
            device=self.audio.microphone_device,
            callback=self.audio.audio_callback
        ):

            print(
                "\n========== VOICE AGENT RUNNING =========="
            )

            print(
                "Speak into your microphone."
            )

            print(
                "Pause between sentences."
            )

            print(
                "Wait for the assistant to respond."
            )

            print(
                "Try speaking while the assistant is talking."
            )

            print(
                "Press CTRL+C to stop."
            )

            while not self.stop_event.is_set():

                if self.deepgram.worker_finished.is_set():

                    print(
                        "\nDeepgram worker unexpectedly stopped."
                    )

                    break

                time.sleep(0.1)

    # ====================================================
    # Shutdown
    # ====================================================

    def shutdown(self):

        print(
            "\n\nShutting down Voice Agent Runtime..."
        )

        self.stop_event.set()

        # ================================================
        # 1. Stop producer workers
        # ================================================

        producer_threads = [
            self.vad_thread,
            self.deepgram_thread
        ]

        for thread in producer_threads:

            if thread is not None and thread.is_alive():

                thread.join(timeout=20)

        # ================================================
        # 2. Drain event consumer workers
        # ================================================

        consumer_threads = [
            self.event_thread,
            self.deepgram_event_thread
        ]

        for thread in consumer_threads:

            if thread is not None and thread.is_alive():

                thread.join(timeout=120)

        # ================================================
        # 3. Finish conversation processing
        # ================================================

        if self.processing_thread is not None:

            self.completed_turn_queue.put(None)

            if self.processing_thread.is_alive():

                self.processing_thread.join(
                    timeout=120
                )

        # ================================================
        # 4. Stop feedback worker
        # ================================================

        self.feedback_stop_event.set()

        if self.coordinator is not None:

            try:

                self.coordinator.stop_speech()

            except Exception as error:

                print(
                    "\nSpeech interruption during shutdown failed:",
                    error
                )

        if self.feedback_thread is not None:

            if self.feedback_thread.is_alive():

                self.feedback_thread.join(
                    timeout=30
                )

        # ================================================
        # 5. Close conversation components
        # ================================================

        if self.coordinator is not None:

            try:

                self.coordinator.close()

            except Exception as error:

                print(
                    "\nCoordinator shutdown error:",
                    error
                )

        # ================================================
        # 6. Close MCP manager
        # ================================================

        if self.mcp_manager is not None:

            try:

                self.mcp_manager.close()

            except Exception as error:

                print(
                    "\nMCP manager shutdown error:",
                    error
                )

        # ================================================
        # 7. Final statistics
        # ================================================

        print(
            "\n========== FINAL STATISTICS =========="
        )

        print(
            "VAD processed frames:",
            self.audio.processed_frames
        )

        print(
            "VAD dropped frames:",
            self.audio.dropped_frames
        )

        print(
            "Completed utterances:",
            self.audio.completed_utterances
        )

        print(
            "Generated speech events:",
            self.audio.generated_events
        )

        print(
            "Consumed speech events:",
            self.audio.consumed_events
        )

        print(
            "Deepgram chunks sent:",
            self.deepgram.total_chunks_sent
        )

        print(
            "\nRuntime shutdown complete."
        )

    # ====================================================
    # Main runtime
    # ====================================================

    def run(self):

        try:

            self.start_workers()

            self.wait_for_deepgram()

            self.initialize_mcp_manager()

            self.initialize_coordinator()

            self.start_background_workers()

            self.start_microphone()

        except KeyboardInterrupt:

            print(
                "\nKeyboard interruption received."
            )

        except Exception as error:

            print(
                "\nRuntime error:",
                error
            )

        finally:

            self.shutdown()


# ========================================================
# Application entry point
# ========================================================

def main():

    runtime = VoiceAgentRuntime()

    runtime.run()


if __name__ == "__main__":

    main()
