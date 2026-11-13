
import asyncio
import queue
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError

import numpy as np
import soxr

from pipecat.frames.frames import (
    InterruptionFrame,
    OutputAudioRawFrame,
)


class BenchmarkPlaybackAdapter:

    def __init__(
        self,
        pipeline_task,
        event_loop,
        output_sample_rate=16000,
        channels=1,
        metrics=None,
    ):
        self.pipeline_task = pipeline_task
        self.event_loop = event_loop
        self.output_sample_rate = output_sample_rate
        self.channels = channels
        self.metrics = metrics

        self._stop_event = threading.Event()
        self._closed = False
        self._state_lock = threading.Lock()
        self._active_queue = None
        self._stream_first_frame_recorded = False

    def _submit_frame(self, frame):

        if self._closed:
            return

        if not self.event_loop.is_running():
            raise RuntimeError(
                "Pipecat event loop is not running."
            )

        future = asyncio.run_coroutine_threadsafe(
            self.pipeline_task.queue_frame(frame),
            self.event_loop,
        )

        try:
            future.result(timeout=10)

        except FutureTimeoutError:
            future.cancel()

            raise RuntimeError(
                "Timed out while submitting a frame to Pipecat."
            )

    def _send_audio(self, samples):

        if self._stop_event.is_set():
            return

        if samples is None or len(samples) == 0:
            return

        samples = np.asarray(
            samples,
            dtype=np.float32,
        ).reshape(-1)

        samples = np.nan_to_num(
            samples,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        pcm16 = np.clip(
            np.rint(samples * 32768.0),
            -32768,
            32767,
        ).astype("<i2")

        frame = OutputAudioRawFrame(
            audio=pcm16.tobytes(),
            sample_rate=self.output_sample_rate,
            num_channels=self.channels,
        )

        self._submit_frame(frame)

        # Timestamp the first audio frame submitted to
        # the Pipecat output transport for this stream.
        if not self._stream_first_frame_recorded:

            self._stream_first_frame_recorded = True

            submitted_at = time.monotonic()

            if self.metrics is not None:
                self.metrics.record_at(
                    "playback_first_output_frame_submitted",
                    submitted_at,
                    output_sample_rate=self.output_sample_rate,
                    channels=self.channels,
                )

    def play_stream(self, audio_queue, sample_rate=24000):

        if sample_rate <= 0:
            raise ValueError(
                "sample_rate must be positive."
            )

        with self._state_lock:

            if self._closed:
                return

            self._active_queue = audio_queue
            self._stream_first_frame_recorded = False

        if self.metrics is not None:
            self.metrics.record(
                "playback_stream_started",
                input_sample_rate=sample_rate,
                output_sample_rate=self.output_sample_rate,
            )

        resampler = soxr.ResampleStream(
            in_rate=sample_rate,
            out_rate=self.output_sample_rate,
            num_channels=self.channels,
            dtype="float32",
            quality="HQ",
        )

        try:

            while not self._stop_event.is_set():

                try:
                    chunk = audio_queue.get(timeout=0.05)

                except queue.Empty:
                    continue

                try:

                    if chunk is None:

                        if not self._stop_event.is_set():

                            final_samples = resampler.resample_chunk(
                                np.empty(
                                    0,
                                    dtype=np.float32,
                                ),
                                last=True,
                            )

                            self._send_audio(final_samples)

                        break

                    if self._stop_event.is_set():
                        break

                    samples = np.asarray(
                        chunk,
                        dtype=np.float32,
                    ).reshape(-1)

                    if len(samples) == 0:
                        continue

                    resampled = resampler.resample_chunk(
                        samples,
                        last=False,
                    )

                    self._send_audio(resampled)

                finally:
                    audio_queue.task_done()

        finally:

            resampler.clear()

            with self._state_lock:

                if self._active_queue is audio_queue:
                    self._active_queue = None

    def stop(self):

        with self._state_lock:

            if self._stop_event.is_set():
                return

            playback_was_active = (
                self._active_queue is not None
            )

            self._stop_event.set()

        try:

            # Submit the interruption frame first.
            self._submit_frame(
                InterruptionFrame()
            )

            # Only count a barge-in when there was active
            # playback to interrupt.
            if playback_was_active and self.metrics is not None:

                stopped_at = time.monotonic()

                self.metrics.record_at(
                    "barge_in_playback_stop_submitted",
                    stopped_at,
                    measurement="Pipecat interruption submission",
                )

                latency = self.metrics.elapsed(
                    "vad_speech_started",
                    "barge_in_playback_stop_submitted",
                )

                if latency is not None:

                    self.metrics.record(
                        "barge_in_latency",
                        duration_s=latency,
                        measurement="VAD start to interruption submission",
                    )

        except Exception as error:

            print(
                "[BENCHMARK PLAYBACK] "
                "Could not submit interruption:",
                error,
            )

    def reset_interruption(self):

        with self._state_lock:

            if not self._closed:
                self._stop_event.clear()

    def close(self):

        with self._state_lock:

            self._closed = True
            self._stop_event.set()
