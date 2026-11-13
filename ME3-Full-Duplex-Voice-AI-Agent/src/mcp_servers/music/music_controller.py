
import sys
import threading
import time

from src.mcp_servers.music.audius_client import (
    search_tracks,
    get_track,
    get_track_stream_chunks
)

from src.mcp_servers.music.audio_decoder import AudioDecoder
from src.mcp_servers.music.playback_engine import PlaybackEngine


class MusicController:
    """
    Coordinates Audius progressive streaming, FFmpeg decoding,
    and buffered music playback.

    MP3 downloading and decoding run in a background worker
    so MCP tool calls return after playback begins.

    Supports:
        - Music search and queue management.
        - Progressive MP3 downloading and decoding.
        - Interruptible streaming.
        - Pause, resume, stop, and skip.
        - Music-only volume control.
        - Assistant speech ducking.
    """

    INITIAL_BUFFER_SECONDS = 3.0
    MAX_BUFFER_SECONDS = 12.0
    STREAM_START_TIMEOUT = 45.0

    def __init__(
        self,
        device=None,
        initial_volume=30,
        duck_factor=0.20,
        fade_duration=0.25
    ):

        if not 0 <= initial_volume <= 100:
            raise ValueError(
                "Initial volume must be between 0 and 100."
            )

        self._lock = threading.RLock()

        # Serializes writes into the playback engine.
        self._stream_write_lock = threading.Lock()

        self.decoder = AudioDecoder()

        self.player = PlaybackEngine(
            device=device,
            fade_duration=fade_duration
        )

        self._user_volume = float(initial_volume)
        self._duck_factor = float(duck_factor)
        self._is_muted = False

        self._current_track = None
        self._queue = []
        self._queue_index = -1

        # Streaming worker management.
        self._playback_generation = 0
        self._stream_stop_event = None
        self._stream_thread = None
        self._stream_done_event = None
        self._stream_error = None
        self._stream_stats = None

        self.player.set_volume(
            self._user_volume / 100.0
        )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    @staticmethod
    def _log(message):

        print(
            message,
            file=sys.stderr,
            flush=True
        )

    @staticmethod
    def _log_timing(stage, elapsed):

        print(
            f"[MUSIC TIMING] {stage}: {elapsed:.3f} seconds",
            file=sys.stderr,
            flush=True
        )

    # ========================================================
    # STREAM CANCELLATION
    # ========================================================

    def _cancel_current_stream(self, stop_player=True):
        """
        Invalidate the current streaming operation.

        The generation counter prevents an older worker from
        feeding audio into a newer playback session.
        """

        with self._lock:

            self._playback_generation += 1

            stop_event = self._stream_stop_event

            self._stream_stop_event = None
            self._stream_done_event = None
            self._stream_error = None
            self._stream_stats = None

        if stop_event is not None:
            stop_event.set()

        if stop_player:
            self.player.stop()

    # ========================================================
    # TRACK SEARCH AND PLAYBACK
    # ========================================================

    def search_and_play(self, query, limit=10):

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        total_start = time.perf_counter()

        self._log(
            f"[MUSIC] Searching for: {query}"
        )

        search_start = time.perf_counter()

        tracks = search_tracks(
            query=query,
            limit=limit
        )

        self._log_timing(
            "Audius track search",
            time.perf_counter() - search_start
        )

        if not tracks:
            raise ValueError(
                "No matching Audius tracks were found."
            )

        with self._lock:

            self._queue = tracks
            self._queue_index = -1

        result = self._play_from_queue(0)

        self._log_timing(
            "Total search and playback startup",
            time.perf_counter() - total_start
        )

        return result

    def play_track(self, track_id):

        if not track_id:
            raise ValueError(
                "Track ID cannot be empty."
            )

        track = get_track(track_id)

        with self._lock:

            self._queue = [track]
            self._queue_index = -1

        return self._start_track(track, 0)

    def _play_from_queue(self, start_index):

        with self._lock:
            queue = list(self._queue)

        for index in range(start_index, len(queue)):

            track = queue[index]

            if not track.get("is_streamable", False):

                self._log(
                    f"[MUSIC] Skipping non-streamable track: "
                    f"{track.get('title')}"
                )

                continue

            return self._start_track(track, index)

        raise ValueError(
            "No streamable tracks remain in the search results."
        )

    # ========================================================
    # BACKGROUND STREAMING WORKER
    # ========================================================

    def _stream_worker(
        self,
        track_id,
        generation,
        stop_event,
        done_event
    ):
        """
        Download and decode MP3 progressively.

        Decoded PCM chunks are forwarded to PlaybackEngine.
        """

        worker_start = time.perf_counter()

        mp3_chunks = get_track_stream_chunks(track_id)

        def on_pcm_chunk(audio_chunk):

            if stop_event.is_set():
                return

            # Serialize access to the playback buffer.
            with self._stream_write_lock:

                with self._lock:

                    is_current = (
                        generation == self._playback_generation
                    )

                if not is_current or stop_event.is_set():
                    stop_event.set()
                    return

                accepted = self.player.add_audio_chunk(
                    audio_chunk
                )

                if not accepted:
                    stop_event.set()

        try:

            self._log(
                f"[MUSIC] Progressive streaming started: {track_id}"
            )

            stats = self.decoder.decode_mp3_stream(
                mp3_chunks=mp3_chunks,
                on_pcm_chunk=on_pcm_chunk,
                stop_event=stop_event
            )

            with self._lock:

                is_current = (
                    generation == self._playback_generation
                )

                if is_current:
                    self._stream_stats = stats

            if stats.get("cancelled", False):
                self._log(
                    f"[MUSIC] Streaming cancelled: {track_id}"
                )
                return

            if stats.get("decoded_frames", 0) <= 0:
                raise RuntimeError(
                    "Audius stream produced no decoded audio."
                )

            # Signal end of stream only if this worker still
            # belongs to the active playback session.
            with self._stream_write_lock:

                with self._lock:

                    is_current = (
                        generation == self._playback_generation
                    )

                if is_current and not stop_event.is_set():

                    self.player.finish_streaming()

            self._log(
                f"[MUSIC] Streaming download and decoding completed: "
                f"{track_id}"
            )

            self._log(
                f"[MUSIC TIMING] Decoded frames: "
                f"{stats['decoded_frames']:,}"
            )

            self._log(
                f"[MUSIC TIMING] Decoded duration: "
                f"{stats['decoded_seconds']} seconds"
            )

            self._log_timing(
                "Background streaming worker",
                time.perf_counter() - worker_start
            )

        except Exception as error:

            with self._lock:

                is_current = (
                    generation == self._playback_generation
                )

                if is_current:
                    self._stream_error = error

            if is_current:

                self._log(
                    f"[MUSIC ERROR] Streaming failed: "
                    f"{type(error).__name__}: {error}"
                )

                stop_event.set()

        finally:

            done_event.set()

    # ========================================================
    # PROGRESSIVE TRACK LOADING
    # ========================================================

    def _start_track(self, track, queue_index):

        total_start = time.perf_counter()

        track_id = track.get("id")

        if not track_id:
            raise ValueError(
                "Track has no valid ID."
            )

        if not track.get("is_streamable", False):
            raise ValueError(
                "This Audius track is not streamable."
            )

        self._log(
            f"\n[MUSIC] Loading: "
            f"{track.get('title', 'Unknown title')}"
        )

        self._log(
            f"[MUSIC] Track ID: {track_id}"
        )

        # Cancel any previous streaming operation.
        self._cancel_current_stream()

        stop_event = threading.Event()
        done_event = threading.Event()

        # Create a new playback generation.
        with self._lock:

            self._playback_generation += 1

            generation = self._playback_generation

            self._stream_stop_event = stop_event
            self._stream_done_event = done_event

            self._stream_error = None
            self._stream_stats = None

            self._current_track = dict(track)
            self._queue_index = queue_index

        # Initialize playback before starting the worker.
        playback_start = time.perf_counter()

        try:

            with self._stream_write_lock:

                self.player.start_streaming(
                    sample_rate=self.decoder.sample_rate,
                    channels=self.decoder.channels,
                    initial_buffer_seconds=self.INITIAL_BUFFER_SECONDS,
                    max_buffer_seconds=self.MAX_BUFFER_SECONDS
                )

        except Exception:

            stop_event.set()

            with self._lock:

                if generation == self._playback_generation:
                    self._stream_stop_event = None
                    self._stream_done_event = None

            raise

        self._log_timing(
            "Playback buffer initialization",
            time.perf_counter() - playback_start
        )

        # Launch progressive download and decoding.
        worker = threading.Thread(
            target=self._stream_worker,
            args=(
                track_id,
                generation,
                stop_event,
                done_event
            ),
            name=f"MusicStream-{track_id}",
            daemon=True
        )

        with self._lock:
            self._stream_thread = worker

        worker.start()

        # Wait only until playback begins, not until the
        # entire song has been downloaded or decoded.
        wait_start = time.perf_counter()

        deadline = (
            time.monotonic() + self.STREAM_START_TIMEOUT
        )

        while time.monotonic() < deadline:

            with self._lock:

                error = (
                    self._stream_error
                    if generation == self._playback_generation
                    else None
                )

                is_current = (
                    generation == self._playback_generation
                )

                stats = self._stream_stats

            if not is_current:
                raise RuntimeError(
                    "Track loading was cancelled by another request."
                )

            if error is not None:

                self._cancel_current_stream()

                raise RuntimeError(
                    f"Unable to start track: {error}"
                ) from error

            status = self.player.get_status()

            if status["state"] == "playing":

                self._log(
                    f"[MUSIC] Now playing: "
                    f"{track.get('title', 'Unknown title')}"
                )

                self._log_timing(
                    "Time until playback started",
                    time.perf_counter() - wait_start
                )

                self._log_timing(
                    "Total playback startup",
                    time.perf_counter() - total_start
                )

                return self.get_status()

            if status["state"] == "finished":

                if stats and stats.get("decoded_frames", 0) > 0:

                    self._log(
                        f"[MUSIC] Track finished before status check: "
                        f"{track.get('title')}"
                    )

                    return self.get_status()

                self._cancel_current_stream()

                raise RuntimeError(
                    "Playback finished without producing audio."
                )

            if done_event.is_set():

                with self._lock:
                    error = self._stream_error

                if error is not None:

                    self._cancel_current_stream()

                    raise RuntimeError(
                        f"Unable to start track: {error}"
                    ) from error

                status = self.player.get_status()

                if status["state"] == "finished":
                    return self.get_status()

            time.sleep(0.05)

        # Startup timeout.
        self._cancel_current_stream()

        raise TimeoutError(
            f"Track did not start within "
            f"{self.STREAM_START_TIMEOUT:.0f} seconds."
        )

    # ========================================================
    # PLAYBACK CONTROLS
    # ========================================================

    def pause(self):

        self.player.pause()

        return self.get_status()

    def resume(self):

        self.player.resume()

        return self.get_status()

    def stop(self):

        self._cancel_current_stream()

        return self.get_status()

    def skip_track(self):

        with self._lock:

            next_index = self._queue_index + 1
            queue_length = len(self._queue)

        if next_index >= queue_length:
            raise ValueError(
                "There are no more tracks in the queue."
            )

        return self._play_from_queue(next_index)

    # ========================================================
    # VOLUME MANAGEMENT
    # ========================================================

    def set_volume(self, volume):

        if not isinstance(volume, (int, float)):
            raise ValueError(
                "Volume must be a number."
            )

        if not 0 <= volume <= 100:
            raise ValueError(
                "Volume must be between 0 and 100."
            )

        with self._lock:

            self._user_volume = float(volume)

            effective_volume = (
                0.0
                if self._is_muted
                else self._user_volume / 100.0
            )

            self.player.set_volume(
                effective_volume
            )

        return self.get_volume()

    def adjust_volume(self, change):

        if not isinstance(change, (int, float)):
            raise ValueError(
                "Volume adjustment must be a number."
            )

        with self._lock:

            new_volume = max(
                0.0,
                min(100.0, self._user_volume + change)
            )

        return self.set_volume(new_volume)

    def get_volume(self):

        with self._lock:

            return {
                "volume_percent": round(
                    self._user_volume, 1
                ),
                "is_muted": self._is_muted
            }

    def mute(self):

        with self._lock:

            self._is_muted = True

            self.player.set_volume(0.0)

        return self.get_volume()

    def unmute(self):

        with self._lock:

            self._is_muted = False

            self.player.set_volume(
                self._user_volume / 100.0
            )

        return self.get_volume()

    # ========================================================
    # ASSISTANT SPEECH INTEGRATION
    # ========================================================

    def assistant_speech_started(self):

        self.player.duck(
            factor=self._duck_factor
        )

        return self.get_status()

    def assistant_speech_finished(self):

        self.player.unduck()

        return self.get_status()

    def assistant_speech_interrupted(self):

        self.player.unduck()

        return self.get_status()

    # ========================================================
    # TRACK AND PLAYER STATUS
    # ========================================================

    def get_current_track(self):

        with self._lock:

            if self._current_track is None:
                return None

            return dict(self._current_track)

    def get_status(self):

        with self._lock:

            track = (
                dict(self._current_track)
                if self._current_track is not None
                else None
            )

            queue_position = (
                self._queue_index + 1
                if self._queue_index >= 0
                else 0
            )

            queue_length = len(self._queue)

            is_muted = self._is_muted
            volume = self._user_volume

        playback_status = self.player.get_status()

        return {
            "current_track": track,
            "playback": playback_status,
            "volume_percent": round(volume, 1),
            "is_muted": is_muted,
            "queue_position": queue_position,
            "queue_length": queue_length
        }

    # ========================================================
    # CLEANUP
    # ========================================================

    def close(self):

        self._cancel_current_stream()

        self._log(
            "[MUSIC] Music controller closed."
        )
