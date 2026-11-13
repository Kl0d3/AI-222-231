
import threading
import queue
import time

import numpy as np
import sounddevice as sd


class AudioPlayback:

    def __init__(self, device=None):

        self.device = device

        self._is_playing = False
        self._closed = False

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    # ====================================================
    # Existing complete-audio playback
    # ====================================================

    def play(self, samples, sample_rate):

        if not isinstance(sample_rate, (int, np.integer)):
            raise TypeError("Sample rate must be an integer.")

        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")

        audio = np.asarray(samples, dtype=np.float32)

        if audio.ndim not in (1, 2):
            raise ValueError("Audio must be a 1D or 2D NumPy array.")

        if audio.size == 0:
            raise ValueError("Audio samples cannot be empty.")

        if not np.isfinite(audio).all():
            raise ValueError("Audio contains invalid numerical values.")

        audio = np.ascontiguousarray(audio)

        channels = 1 if audio.ndim == 1 else audio.shape[1]

        with self._lock:

            if self._closed:
                raise RuntimeError("AudioPlayback has been closed.")

            if self._stop_event.is_set():
                print("[AUDIO] Playback skipped: interruption active.")
                return

            if self._is_playing:
                raise RuntimeError("Audio playback is already in progress.")

            self._is_playing = True

        try:

            print("\n[AUDIO] Playing audio...")

            chunk_size = 1024

            with sd.OutputStream(
                device=self.device,
                samplerate=int(sample_rate),
                channels=channels,
                dtype="float32"
            ) as stream:

                for start in range(0, len(audio), chunk_size):

                    if self._stop_event.is_set():
                        break

                    chunk = audio[start:start + chunk_size]

                    stream.write(chunk)

            if self._stop_event.is_set():
                print("[AUDIO] Playback interrupted.")

            else:
                print("[AUDIO] Playback completed.")

        finally:

            with self._lock:
                self._is_playing = False

    # ====================================================
    # Streaming audio playback
    # ====================================================

    def play_stream(self, audio_queue, sample_rate=24000):

        if not isinstance(audio_queue, queue.Queue):
            raise TypeError("audio_queue must be a queue.Queue.")

        if not isinstance(sample_rate, (int, np.integer)):
            raise TypeError("Sample rate must be an integer.")

        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")

        with self._lock:

            if self._closed:
                raise RuntimeError("AudioPlayback has been closed.")

            if self._stop_event.is_set():
                print("[AUDIO] Streaming skipped: interruption active.")
                return

            if self._is_playing:
                raise RuntimeError("Audio playback is already in progress.")

            self._is_playing = True

        # 20 ms of audio per playback frame.
        frame_size = max(1, int(sample_rate * 0.020))

        # Buffer 200 ms before starting playback.
        prebuffer_size = int(sample_rate * 0.200)

        audio_buffer = np.empty(0, dtype=np.float32)
        stream_finished = False

        # ------------------------------------------------
        # Diagnostic measurements
        # ------------------------------------------------

        queue_get_calls = 0
        queue_chunks_received = 0
        queue_empty_count = 0

        queue_wait_total = 0.0
        queue_wait_max = 0.0
        queue_empty_wait_total = 0.0

        frames_written = 0
        write_total = 0.0
        write_max = 0.0
        slow_writes = 0

        playback_start_time = time.perf_counter()

        def receive_audio_chunk():

            nonlocal stream_finished
            nonlocal queue_get_calls
            nonlocal queue_chunks_received
            nonlocal queue_empty_count
            nonlocal queue_wait_total
            nonlocal queue_wait_max
            nonlocal queue_empty_wait_total

            get_start = time.perf_counter()

            try:

                chunk = audio_queue.get(timeout=0.1)

            except queue.Empty:

                wait_time = time.perf_counter() - get_start

                queue_get_calls += 1
                queue_empty_count += 1

                queue_wait_total += wait_time
                queue_empty_wait_total += wait_time
                queue_wait_max = max(
                    queue_wait_max,
                    wait_time
                )

                return False

            wait_time = time.perf_counter() - get_start

            queue_get_calls += 1
            queue_wait_total += wait_time
            queue_wait_max = max(
                queue_wait_max,
                wait_time
            )

            try:

                if chunk is None:
                    stream_finished = True
                    return True

                chunk = np.asarray(chunk, dtype=np.float32)

                if chunk.ndim != 1:
                    raise ValueError(
                        "Streaming audio chunks must be 1D arrays."
                    )

                if chunk.size == 0:
                    return True

                if not np.isfinite(chunk).all():
                    raise ValueError(
                        "Streaming audio contains invalid values."
                    )

                queue_chunks_received += 1

                return chunk

            finally:
                audio_queue.task_done()

        try:

            print("\n[AUDIO] Preparing streaming buffer...")

            # --------------------------------------------
            # Phase 1: Pre-buffer audio
            # --------------------------------------------

            while (
                audio_buffer.size < prebuffer_size
                and not stream_finished
                and not self._stop_event.is_set()
            ):

                chunk = receive_audio_chunk()

                if isinstance(chunk, np.ndarray):

                    audio_buffer = np.concatenate(
                        (audio_buffer, chunk)
                    )

            if self._stop_event.is_set():

                print(
                    "[AUDIO] Streaming interrupted during buffering."
                )

                return

            if audio_buffer.size == 0:

                print("[AUDIO] No audio received.")
                return

            print(
                f"[AUDIO] Initial buffer: "
                f"{audio_buffer.size / sample_rate:.2f} seconds."
            )

            # --------------------------------------------
            # Phase 2: Continuous playback
            # --------------------------------------------

            print("[AUDIO] Starting buffered playback...")

            with sd.OutputStream(
                device=self.device,
                samplerate=int(sample_rate),
                channels=1,
                dtype="float32"
            ) as stream:

                while not self._stop_event.is_set():

                    # Collect enough samples for one frame.
                    while (
                        audio_buffer.size < frame_size
                        and not stream_finished
                        and not self._stop_event.is_set()
                    ):

                        chunk = receive_audio_chunk()

                        if isinstance(chunk, np.ndarray):

                            audio_buffer = np.concatenate(
                                (audio_buffer, chunk)
                            )

                    if self._stop_event.is_set():
                        break

                    if audio_buffer.size == 0 and stream_finished:
                        break

                    # Write one consistent 20 ms frame.
                    if audio_buffer.size >= frame_size:

                        frame = audio_buffer[:frame_size]

                        audio_buffer = audio_buffer[frame_size:]

                    else:

                        # Flush the final incomplete frame.
                        frame = audio_buffer

                        audio_buffer = np.empty(
                            0,
                            dtype=np.float32
                        )

                    # Measure output write duration.
                    write_start = time.perf_counter()

                    stream.write(
                        np.ascontiguousarray(frame)
                    )

                    write_duration = (
                        time.perf_counter() - write_start
                    )

                    frames_written += 1
                    write_total += write_duration

                    write_max = max(
                        write_max,
                        write_duration
                    )

                    if write_duration > 0.020:
                        slow_writes += 1

            if self._stop_event.is_set():

                print("[AUDIO] Streaming playback interrupted.")

            else:

                print("[AUDIO] Streaming playback completed.")

        finally:

            playback_elapsed = (
                time.perf_counter() - playback_start_time
            )

            # --------------------------------------------
            # Diagnostic report
            # --------------------------------------------

            print("\n========== AUDIO PLAYBACK DIAGNOSTICS ==========")

            print(f"Playback worker duration: {playback_elapsed:.3f}s")

            print("\n--- Queue measurements ---")

            print(f"Queue get calls: {queue_get_calls}")
            print(f"Audio chunks received: {queue_chunks_received}")
            print(f"Queue empty timeouts: {queue_empty_count}")

            print(
                f"Total queue waiting time: "
                f"{queue_wait_total * 1000:.2f} ms"
            )

            print(
                f"Total empty-queue waiting time: "
                f"{queue_empty_wait_total * 1000:.2f} ms"
            )

            print(
                f"Maximum queue wait: "
                f"{queue_wait_max * 1000:.2f} ms"
            )

            print("\n--- Output measurements ---")

            print(f"Frames written: {frames_written}")

            if frames_written > 0:

                print(
                    f"Average stream.write duration: "
                    f"{(write_total / frames_written) * 1000:.2f} ms"
                )

                print(
                    f"Maximum stream.write duration: "
                    f"{write_max * 1000:.2f} ms"
                )

                print(
                    f"Writes exceeding 20 ms: {slow_writes}"
                )

            print("=================================================\n")

            with self._lock:
                self._is_playing = False

    # ====================================================
    # Stop playback
    # ====================================================

    def stop(self):

        self._stop_event.set()

        with self._lock:

            if self._is_playing:
                print("\n[AUDIO] Interruption requested.")

    # ====================================================
    # Reset interruption
    # ====================================================

    def reset_interruption(self):

        with self._lock:

            if self._is_playing:
                raise RuntimeError(
                    "Cannot reset interruption during active playback."
                )

            self._stop_event.clear()

    # ====================================================
    # Close playback
    # ====================================================

    def close(self):

        self.stop()

        with self._lock:
            self._closed = True

        print("[AUDIO] AudioPlayback closed.")
