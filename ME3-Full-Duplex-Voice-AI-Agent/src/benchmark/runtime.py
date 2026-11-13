import asyncio
import queue
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE, override=False)

from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)

from src.agent.runtime import VoiceAgentRuntime
from src.agent.coordinator import ConversationCoordinator
from src.benchmark.audio_adapter import BenchmarkAudioAdapter
from src.benchmark.playback_adapter import BenchmarkPlaybackAdapter
from src.benchmark.metrics import BenchmarkMetrics


class BenchmarkInputBridge(FrameProcessor):

    def __init__(self, audio_adapter):
        super().__init__(name="BenchmarkInputBridge")
        self.audio_adapter = audio_adapter

    async def process_frame(self, frame, direction: FrameDirection):

        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:

            if isinstance(frame, InputAudioRawFrame):

                if frame.sample_rate != 16000:
                    raise RuntimeError(
                        "Benchmark input must be 16 kHz. "
                        f"Received {frame.sample_rate} Hz."
                    )

                if frame.num_channels != 1:
                    raise RuntimeError(
                        "Benchmark input must be mono. "
                        f"Received {frame.num_channels} channels."
                    )

                self.audio_adapter.process_pcm16(frame.audio)

            elif isinstance(frame, EndFrame):
                self.audio_adapter.flush()

        await self.push_frame(frame, direction)


class BenchmarkVoiceAgentRuntime(VoiceAgentRuntime):

    def __init__(self, transport, log_dir: Path):

        super().__init__()

        self.transport = transport
        self.log_dir = Path(log_dir)

        self.metrics = BenchmarkMetrics(self.log_dir)

        self.pipeline_task = None
        self.pipeline_runner = None

        self.benchmark_audio_adapter = None
        self.benchmark_playback_adapter = None

        self.cancel_requested = threading.Event()
        self.shutdown_started = threading.Event()

    # ====================================================
    # Benchmark audio pipeline
    # ====================================================

    def initialize_benchmark_pipeline(self, event_loop):

        print("\nInitializing benchmark audio pipeline...")

        self.benchmark_audio_adapter = BenchmarkAudioAdapter(
            self.audio
        )

        input_bridge = BenchmarkInputBridge(
            self.benchmark_audio_adapter
        )

        pipeline = Pipeline(
            [
                self.transport.input(),
                input_bridge,
                self.transport.output(),
            ]
        )

        pipeline_params = PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
        )

        self.pipeline_task = PipelineTask(
            pipeline,
            params=pipeline_params,
            cancel_on_idle_timeout=False,
            cancel_runner_on_idle_timeout=False,
        )

        self.pipeline_runner = PipelineRunner()

        self.benchmark_playback_adapter = BenchmarkPlaybackAdapter(
            pipeline_task=self.pipeline_task,
            event_loop=event_loop,
            output_sample_rate=16000,
            channels=1,
            metrics=self.metrics,
        )

        print("\nBenchmark audio pipeline initialized.")

    # ====================================================
    # VAD event consumer
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

                print("\n[VAD EVENT]", event_type)

                with self.audio.stats_lock:
                    self.audio.consumed_events += 1

                event_timestamp = event.get("timestamp")

                if event_timestamp is not None:
                    self.metrics.record_at(
                        f"vad_{event_type.lower()}",
                        event_timestamp,
                        source="AudioBuffer",
                    )

                if event_type == "SPEECH_STARTED":

                    self.user_speech_active.set()

                    with self.activity_lock:
                        self.last_user_speech_stopped_at = None
                        self.backchannel_sent_for_pause = False

                    self.begin_user_turn_if_needed()

                    if self.coordinator is not None:

                        print("\n[BARGE-IN] User speech detected.")

                        self.coordinator.stop_speech()

                elif event_type == "SPEECH_STOPPED":

                    self.user_speech_active.clear()

                    with self.activity_lock:
                        self.last_user_speech_stopped_at = (
                            time.monotonic()
                        )

                        self.backchannel_sent_for_pause = False

                    self.metrics.record(
                        "vad_speech_stopped_processed",
                        source_timestamp=event_timestamp,
                    )

                    print("\n[RUNTIME] User speech paused.")

            except Exception as error:

                print(
                    "\nSpeech event processing error:",
                    error,
                )

            finally:
                self.audio.speech_event_queue.task_done()

        print("\nSpeech event worker stopped.")

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
                        event.get("error"),
                    )

                elif event_type == "EndOfTurn":

                    self.metrics.record(
                        "asr_final_event_consumed",
                        transcript_length=len(transcript.strip()),
                    )

                    print("\n[DEEPGRAM EVENT] EndOfTurn")

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

                    self.process_completed_turn(transcript)

                elif event_type and event_type != "Update":

                    print(
                        "\n[DEEPGRAM EVENT]",
                        event_type,
                    )

                    if transcript:
                        print("Transcript:", transcript)

                elif transcript:

                    print(
                        "\n[DEEPGRAM TRANSCRIPT UPDATE]",
                        transcript,
                    )

            except Exception as error:

                print(
                    "\nDeepgram event processing error:",
                    error,
                )

            finally:
                self.deepgram.event_queue.task_done()

        print("\nDeepgram event consumer stopped.")

    # ====================================================
    # Initialize coordinator
    # ====================================================

    def initialize_coordinator(self):

        print(
            "\nInitializing Benchmark Conversation Coordinator..."
        )

        self.coordinator = ConversationCoordinator(
            mcp_manager=self.mcp_manager,
            playback_client=self.benchmark_playback_adapter,
        )

        self.coordinator.benchmark_metrics = self.metrics

        self.coordinator.gemini_client.feedback_callback = (
            self.handle_gemini_feedback
        )

        print("\nGemini feedback callback connected.")
        print("\nBenchmark Conversation Coordinator initialized.")

    # ====================================================
    # Gemini response timing
    # ====================================================

    def install_gemini_metrics(self):

        original_generate_response = (
            self.coordinator.gemini_client.generate_response
        )

        def measured_generate_response(*args, **kwargs):

            started_at = time.monotonic()

            self.metrics.record_at(
                "gemini_request_started",
                started_at,
            )

            succeeded = False

            try:

                result = original_generate_response(
                    *args,
                    **kwargs,
                )

                succeeded = True
                return result

            finally:

                completed_at = time.monotonic()

                self.metrics.record_at(
                    "gemini_response_completed",
                    completed_at,
                    succeeded=succeeded,
                )

                self.metrics.record(
                    "gemini_response_latency",
                    duration_s=completed_at - started_at,
                    succeeded=succeeded,
                )

        self.coordinator.gemini_client.generate_response = (
            measured_generate_response
        )

        print("\n[GEMINI BENCHMARK] Response timing enabled.")

    # ====================================================
    # TTS first-audio timing
    # ====================================================

    def install_tts_metrics(self):

        tts_client = self.coordinator.deepgram_tts_client

        original_generate_audio_stream = (
            tts_client.generate_audio_stream
        )

        def measured_generate_audio_stream(*args, **kwargs):

            callback = kwargs.get("on_audio_chunk")
            callback_in_kwargs = callback is not None

            if callback_in_kwargs:
                original_callback = callback

            elif len(args) >= 2:
                original_callback = args[1]

            else:
                return original_generate_audio_stream(
                    *args,
                    **kwargs,
                )

            if not callable(original_callback):
                return original_generate_audio_stream(
                    *args,
                    **kwargs,
                )

            started_at = time.monotonic()

            self.metrics.record_at(
                "tts_generation_started",
                started_at,
                provider="Deepgram",
            )

            first_chunk_recorded = False

            def measured_audio_callback(audio_bytes):

                nonlocal first_chunk_recorded

                if not first_chunk_recorded:

                    first_chunk_recorded = True

                    first_audio_at = time.monotonic()
                    latency = first_audio_at - started_at

                    self.metrics.record_at(
                        "tts_first_audio_chunk",
                        first_audio_at,
                        provider="Deepgram",
                        latency_s=latency,
                        chunk_bytes=len(audio_bytes),
                    )

                    self.metrics.record(
                        "tts_first_audio_latency",
                        duration_s=latency,
                        provider="Deepgram",
                    )

                return original_callback(audio_bytes)

            if callback_in_kwargs:

                kwargs["on_audio_chunk"] = measured_audio_callback

            else:

                args = list(args)
                args[1] = measured_audio_callback
                args = tuple(args)

            succeeded = False

            try:

                result = original_generate_audio_stream(
                    *args,
                    **kwargs,
                )

                succeeded = True
                return result

            finally:

                completed_at = time.monotonic()

                self.metrics.record_at(
                    "tts_generation_completed",
                    completed_at,
                    provider="Deepgram",
                    succeeded=succeeded,
                    first_chunk_received=first_chunk_recorded,
                )

                self.metrics.record(
                    "tts_generation_latency",
                    duration_s=completed_at - started_at,
                    provider="Deepgram",
                    succeeded=succeeded,
                    first_chunk_received=first_chunk_recorded,
                )

        tts_client.generate_audio_stream = (
            measured_generate_audio_stream
        )

        print("\n[TTS BENCHMARK] First-audio timing enabled.")

    # ====================================================
    # Wait for Deepgram
    # ====================================================

    async def wait_for_deepgram_async(self, timeout=30):

        print("\nWaiting for Deepgram connection...")

        deadline = time.monotonic() + timeout

        while not self.deepgram.connection_ready.is_set():

            if self.cancel_requested.is_set():
                raise asyncio.CancelledError(
                    "Benchmark startup was cancelled."
                )

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Deepgram connection timed out."
                )

            await asyncio.sleep(0.05)

        self.wait_for_deepgram(timeout=0)

    # ====================================================
    # Benchmark runtime
    # ====================================================

    async def run(self):

        print("\n========== BENCHMARK RUNTIME STARTING ==========")

        self.metrics.record("benchmark_runtime_start")

        try:

            event_loop = asyncio.get_running_loop()

            self.initialize_benchmark_pipeline(event_loop)

            self.start_workers()

            await self.wait_for_deepgram_async()

            if self.cancel_requested.is_set():
                return

            self.initialize_mcp_manager()

            self.initialize_coordinator()

            self.install_gemini_metrics()
            self.install_tts_metrics()

            self.start_background_workers()

            if self.cancel_requested.is_set():
                return

            self.metrics.record("benchmark_runtime_ready")

            print("\n========== BENCHMARK ASSISTANT READY ==========")
            print("Audio input: fdbench Pipecat transport")
            print("Audio output: fdbench Pipecat transport")
            print("Microphone: disabled")
            print("Physical speaker: disabled")

            await self.pipeline_runner.run(self.pipeline_task)

            print("\nBenchmark transport pipeline stopped.")

        finally:

            self.metrics.record("benchmark_runtime_stopping")

            print("\nStopping benchmark runtime...")

            await asyncio.to_thread(self.shutdown)

            print("\nBenchmark runtime stopped.")

    # ====================================================
    # Benchmark-specific shutdown
    # ====================================================

    def shutdown(self):

        if self.shutdown_started.is_set():
            return

        self.shutdown_started.set()

        print("\n[SHUTDOWN] Starting benchmark-specific shutdown.")

        # 1. Stop new audio and conversation work.
        self.cancel_requested.set()
        self.stop_event.set()
        self.feedback_stop_event.set()

        # 2. Invalidate the active turn before waiting for
        #    the processing thread. This prevents a late
        #    Gemini response from being admitted for speech.
        if self.coordinator is not None:

            try:

                manager = self.coordinator.feedback_manager
                status = manager.get_status()

                if status.get("turn_active"):
                    manager.end_turn()

            except Exception as error:

                print(
                    "\n[SHUTDOWN] Turn invalidation failed:",
                    error,
                )

            # 3. Interrupt active playback and TTS generation.
            try:

                self.coordinator.stop_speech()

            except Exception as error:

                print(
                    "\n[SHUTDOWN] Speech interruption failed:",
                    error,
                )

        # 4. Discard completed turns that have not started.
        while True:

            try:
                item = self.completed_turn_queue.get_nowait()

            except queue.Empty:
                break

            else:
                self.completed_turn_queue.task_done()

        # 5. Use the existing shutdown for worker joins,
        #    component closure, and MCP cleanup.
        super().shutdown()

        print("\n[SHUTDOWN] Benchmark-specific shutdown complete.")

    # ====================================================
    # Cancellation
    # ====================================================

    async def cancel(self):

        print("\nBenchmark cancellation requested.")

        self.cancel_requested.set()
        self.stop_event.set()

        if self.pipeline_task is not None:
            await self.pipeline_task.cancel(
                reason="Benchmark transport disconnected."
            )