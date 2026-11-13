
import threading

import numpy as np
import sounddevice as sd


class PlaybackEngine:
    """
    Independent music playback engine.

    Supports:
        1. Complete NumPy audio playback.
        2. Progressive PCM streaming through a bounded buffer.
        3. Pause, resume, stop, and playback status.
        4. Music-only volume control.
        5. Smooth volume transitions and audio ducking.

    Audio format:
        float32, shape (frames, channels)
    """

    def __init__(
        self,
        device=None,
        fade_duration=0.25
    ):

        self.device = device
        self.fade_duration = fade_duration

        self._lock = threading.RLock()
        self._buffer_condition = threading.Condition(self._lock)

        self._stream = None
        self._audio = None

        self._sample_rate = None
        self._channels = None

        self._position = 0
        self._total_frames = 0

        self._state = "stopped"

        # Playback mode.
        self._streaming = False
        self._stream_finished = False
        self._has_started_playback = False
        self._state_before_pause = None

        # Circular PCM buffer.
        self._buffer = None
        self._buffer_capacity = 0
        self._buffer_read_index = 0
        self._buffer_write_index = 0
        self._buffered_frames = 0

        self._initial_buffer_frames = 0
        self._stream_frames_received = 0

        # User-selected volume.
        self._user_volume = 0.30

        # Temporary ducking configuration.
        self._duck_factor = 0.20
        self._is_ducked = False

        # Gain transition state.
        self._current_gain = self._user_volume
        self._target_gain = self._user_volume

    # ========================================================
    # VOLUME MANAGEMENT
    # ========================================================

    def _calculate_target_gain(self):

        if self._is_ducked:
            return self._user_volume * self._duck_factor

        return self._user_volume

    def _update_target_gain(self):

        self._target_gain = self._calculate_target_gain()

    def set_volume(self, volume):

        if not 0.0 <= volume <= 1.0:
            raise ValueError(
                "Volume must be between 0.0 and 1.0"
            )

        with self._lock:

            self._user_volume = float(volume)
            self._update_target_gain()

    def get_volume(self):

        with self._lock:
            return self._user_volume

    def duck(self, factor=None):

        if factor is not None:

            if not 0.0 <= factor <= 1.0:
                raise ValueError(
                    "Ducking factor must be between 0.0 and 1.0"
                )

        with self._lock:

            if factor is not None:
                self._duck_factor = float(factor)

            self._is_ducked = True
            self._update_target_gain()

    def unduck(self):

        with self._lock:

            self._is_ducked = False
            self._update_target_gain()

    # ========================================================
    # GAIN PROCESSING
    # ========================================================

    def _apply_gain(self, audio_chunk):

        frames = len(audio_chunk)

        if frames == 0:
            return audio_chunk

        target = self._target_gain
        current = self._current_gain

        ramp_frames = max(
            1,
            int(self._sample_rate * self.fade_duration)
        )

        max_change_per_frame = (
            abs(target - current) / ramp_frames
        )

        if target > current:

            gains = np.minimum(
                current
                + max_change_per_frame
                * np.arange(1, frames + 1),
                target
            )

        elif target < current:

            gains = np.maximum(
                current
                - max_change_per_frame
                * np.arange(1, frames + 1),
                target
            )

        else:

            gains = np.full(
                frames,
                target,
                dtype=np.float32
            )

        self._current_gain = float(gains[-1])

        return audio_chunk * gains[:, np.newaxis]

    # ========================================================
    # COMPLETE AUDIO PLAYBACK CALLBACK
    # ========================================================

    def _play_complete_audio(
        self,
        outdata,
        frames
    ):

        if self._state == "paused":

            outdata.fill(0)
            return

        if self._state != "playing" or self._audio is None:

            outdata.fill(0)
            raise sd.CallbackStop

        remaining = self._total_frames - self._position

        frames_to_copy = min(
            frames,
            remaining
        )

        if frames_to_copy <= 0:

            outdata.fill(0)
            self._state = "finished"

            raise sd.CallbackStop

        start = self._position
        end = start + frames_to_copy

        audio_chunk = self._audio[start:end]

        outdata[:frames_to_copy] = self._apply_gain(
            audio_chunk
        )

        if frames_to_copy < frames:

            outdata[frames_to_copy:].fill(0)

        self._position = end

        if self._position >= self._total_frames:

            self._state = "finished"

            raise sd.CallbackStop

    # ========================================================
    # STREAMING AUDIO CALLBACK
    # ========================================================

    def _play_streaming_audio(
        self,
        outdata,
        frames
    ):

        outdata.fill(0)

        if self._state == "paused":
            return

        if self._state in ("stopped", "finished"):
            raise sd.CallbackStop

        # Before the first playback, wait for the initial
        # buffer threshold or the end of the audio stream.
        if not self._has_started_playback:

            enough_audio = (
                self._buffered_frames
                >= self._initial_buffer_frames
            )

            short_finished_stream = (
                self._stream_finished
                and self._buffered_frames > 0
            )

            if not enough_audio and not short_finished_stream:

                if self._stream_finished:

                    self._state = "finished"
                    raise sd.CallbackStop

                self._state = "buffering"
                return

            self._has_started_playback = True

        if self._buffered_frames == 0:

            if self._stream_finished:

                self._state = "finished"
                raise sd.CallbackStop

            self._state = "buffering"
            return

        frames_to_copy = min(
            frames,
            self._buffered_frames
        )

        first_part = min(
            frames_to_copy,
            self._buffer_capacity - self._buffer_read_index
        )

        second_part = frames_to_copy - first_part

        # Read the first contiguous section.
        audio_chunk = self._buffer[
            self._buffer_read_index:
            self._buffer_read_index + first_part
        ]

        if second_part > 0:

            audio_chunk = np.concatenate(
                (
                    audio_chunk,
                    self._buffer[:second_part]
                ),
                axis=0
            )

        # Apply volume and ducking.
        outdata[:frames_to_copy] = self._apply_gain(
            audio_chunk
        )

        # Move circular buffer read position.
        self._buffer_read_index = (
            self._buffer_read_index + frames_to_copy
        ) % self._buffer_capacity

        self._buffered_frames -= frames_to_copy

        self._position += frames_to_copy

        self._state = "playing"

        # Notify producer that buffer space is available.
        self._buffer_condition.notify_all()

        # If the producer has finished and all buffered
        # samples have been consumed, stop playback.
        if (
            self._stream_finished
            and self._buffered_frames == 0
        ):

            self._state = "finished"

            raise sd.CallbackStop

    # ========================================================
    # SOUNDDEVICE CALLBACK
    # ========================================================

    def _audio_callback(
        self,
        outdata,
        frames,
        time_info,
        status
    ):

        with self._lock:

            if self._streaming:

                self._play_streaming_audio(
                    outdata,
                    frames
                )

            else:

                self._play_complete_audio(
                    outdata,
                    frames
                )

    # ========================================================
    # STREAM MANAGEMENT
    # ========================================================

    def _create_output_stream(self):

        return sd.OutputStream(
            device=self.device,
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            callback=self._audio_callback
        )

    # ========================================================
    # EXISTING COMPLETE AUDIO PLAYBACK
    # ========================================================

    def start(self, audio, sample_rate):

        self.stop()

        audio = np.asarray(
            audio,
            dtype=np.float32
        )

        if audio.ndim == 1:

            audio = audio.reshape(-1, 1)

        if audio.ndim != 2:

            raise ValueError(
                "Audio must have shape (frames, channels)"
            )

        if len(audio) == 0:

            raise ValueError(
                "Audio cannot be empty"
            )

        if sample_rate <= 0:

            raise ValueError(
                "Sample rate must be positive"
            )

        if not np.isfinite(audio).all():

            raise ValueError(
                "Audio contains invalid values"
            )

        audio = np.ascontiguousarray(
            np.clip(audio, -1.0, 1.0),
            dtype=np.float32
        )

        with self._lock:

            self._streaming = False

            self._stream_finished = True
            self._has_started_playback = False

            self._buffer = None
            self._buffer_capacity = 0
            self._buffered_frames = 0

            self._audio = audio

            self._sample_rate = int(sample_rate)
            self._channels = audio.shape[1]

            self._position = 0
            self._total_frames = len(audio)

            self._state = "playing"

            self._current_gain = self._calculate_target_gain()
            self._update_target_gain()

            self._stream = self._create_output_stream()

            stream = self._stream

        try:

            stream.start()

        except Exception:

            with self._lock:

                self._state = "stopped"
                self._stream = None
                self._audio = None

            stream.close()
            raise

    # ========================================================
    # STREAMING PLAYBACK INITIALIZATION
    # ========================================================

    def start_streaming(
        self,
        sample_rate,
        channels=2,
        initial_buffer_seconds=3.0,
        max_buffer_seconds=12.0
    ):
        """
        Initialize progressive playback.

        The output stream starts in buffering state.
        Actual audio playback begins after the initial
        buffer threshold is reached.
        """

        self.stop()

        if sample_rate <= 0:

            raise ValueError(
                "Sample rate must be positive"
            )

        if channels <= 0:

            raise ValueError(
                "Channels must be positive"
            )

        if initial_buffer_seconds <= 0:

            raise ValueError(
                "Initial buffer duration must be positive"
            )

        if max_buffer_seconds < initial_buffer_seconds:

            raise ValueError(
                "Maximum buffer must be at least "
                "the initial buffer duration"
            )

        sample_rate = int(sample_rate)
        channels = int(channels)

        initial_frames = int(
            sample_rate * initial_buffer_seconds
        )

        capacity_frames = int(
            sample_rate * max_buffer_seconds
        )

        with self._lock:

            self._streaming = True
            self._stream_finished = False
            self._has_started_playback = False

            self._sample_rate = sample_rate
            self._channels = channels

            self._buffer_capacity = capacity_frames

            self._buffer = np.zeros(
                (capacity_frames, channels),
                dtype=np.float32
            )

            self._buffer_read_index = 0
            self._buffer_write_index = 0
            self._buffered_frames = 0

            self._initial_buffer_frames = initial_frames
            self._stream_frames_received = 0

            self._position = 0
            self._total_frames = 0

            self._audio = None

            self._state = "buffering"

            self._current_gain = self._calculate_target_gain()
            self._update_target_gain()

            self._stream = self._create_output_stream()

            stream = self._stream

        try:

            stream.start()

        except Exception:

            with self._lock:

                self._state = "stopped"
                self._stream = None

                self._buffer = None
                self._buffer_capacity = 0

            stream.close()
            raise

    # ========================================================
    # ADD DECODED AUDIO CHUNKS
    # ========================================================

    def add_audio_chunk(self, audio_chunk):
        """
        Add decoded PCM audio to the circular buffer.

        This method may wait when the buffer is full.
        The SoundDevice callback never waits for the producer.
        """

        audio_chunk = np.asarray(
            audio_chunk,
            dtype=np.float32
        )

        if audio_chunk.ndim == 1:

            audio_chunk = audio_chunk.reshape(-1, 1)

        if audio_chunk.ndim != 2:

            raise ValueError(
                "Audio chunk must have shape (frames, channels)"
            )

        if len(audio_chunk) == 0:
            return True

        if not np.isfinite(audio_chunk).all():

            raise ValueError(
                "Audio chunk contains invalid values"
            )

        audio_chunk = np.ascontiguousarray(
            np.clip(audio_chunk, -1.0, 1.0),
            dtype=np.float32
        )

        with self._buffer_condition:

            if not self._streaming:

                raise RuntimeError(
                    "Streaming playback has not been initialized"
                )

            if audio_chunk.shape[1] != self._channels:

                raise ValueError(
                    f"Expected {self._channels} channels, "
                    f"received {audio_chunk.shape[1]}"
                )

            if self._stream_finished:

                raise RuntimeError(
                    "Cannot add audio after end of stream"
                )

            source_position = 0
            total_frames = len(audio_chunk)

            while source_position < total_frames:

                while (
                    self._buffered_frames >= self._buffer_capacity
                    and self._state != "stopped"
                ):

                    self._buffer_condition.wait(
                        timeout=0.1
                    )

                if self._state == "stopped":

                    return False

                available_space = (
                    self._buffer_capacity
                    - self._buffered_frames
                )

                frames_to_write = min(
                    total_frames - source_position,
                    available_space
                )

                first_part = min(
                    frames_to_write,
                    self._buffer_capacity
                    - self._buffer_write_index
                )

                second_part = (
                    frames_to_write - first_part
                )

                source_end = (
                    source_position + first_part
                )

                self._buffer[
                    self._buffer_write_index:
                    self._buffer_write_index + first_part
                ] = audio_chunk[
                    source_position:source_end
                ]

                if second_part > 0:

                    self._buffer[:second_part] = (
                        audio_chunk[
                            source_end:
                            source_end + second_part
                        ]
                    )

                self._buffer_write_index = (
                    self._buffer_write_index + frames_to_write
                ) % self._buffer_capacity

                self._buffered_frames += frames_to_write

                self._stream_frames_received += frames_to_write

                source_position += frames_to_write

        return True

    # ========================================================
    # SIGNAL END OF STREAM
    # ========================================================

    def finish_streaming(self):

        with self._buffer_condition:

            if not self._streaming:
                return

            self._stream_finished = True

            self._total_frames = self._stream_frames_received

            self._buffer_condition.notify_all()

    # ========================================================
    # PAUSE / RESUME
    # ========================================================

    def pause(self):

        with self._lock:

            if self._state not in (
                "playing",
                "buffering"
            ):
                return

            self._state_before_pause = self._state
            self._state = "paused"

    def resume(self):

        with self._lock:

            if self._state != "paused":
                return

            self._state = (
                self._state_before_pause or "playing"
            )

            self._state_before_pause = None

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        with self._buffer_condition:

            stream = self._stream

            self._stream = None

            self._state = "stopped"

            self._audio = None

            self._position = 0
            self._total_frames = 0

            self._streaming = False
            self._stream_finished = False
            self._has_started_playback = False

            self._buffer = None
            self._buffer_capacity = 0

            self._buffer_read_index = 0
            self._buffer_write_index = 0
            self._buffered_frames = 0

            self._initial_buffer_frames = 0
            self._stream_frames_received = 0

            self._state_before_pause = None

            self._buffer_condition.notify_all()

        if stream is not None:

            try:

                stream.stop()

            finally:

                stream.close()

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self):

        with self._lock:

            duration = 0.0
            position = 0.0
            buffered_seconds = 0.0
            buffer_capacity_seconds = 0.0

            if self._sample_rate:

                position = (
                    self._position / self._sample_rate
                )

                if not self._streaming or self._stream_finished:

                    duration = (
                        self._total_frames / self._sample_rate
                    )

                buffered_seconds = (
                    self._buffered_frames / self._sample_rate
                )

                if self._streaming:

                    buffer_capacity_seconds = (
                        self._buffer_capacity / self._sample_rate
                    )

            return {
                "state": self._state,
                "position_seconds": round(position, 2),
                "duration_seconds": round(duration, 2),
                "volume": round(self._user_volume, 2),
                "effective_volume": round(
                    self._target_gain,
                    2
                ),
                "is_ducked": self._is_ducked,
                "streaming": self._streaming,
                "buffered_seconds": round(
                    buffered_seconds,
                    2
                ),
                "buffer_capacity_seconds": round(
                    buffer_capacity_seconds,
                    2
                )
            }
