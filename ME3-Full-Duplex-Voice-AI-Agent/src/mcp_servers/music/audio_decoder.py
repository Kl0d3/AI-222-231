
import queue
import subprocess
import threading

import numpy as np


class AudioDecoder:
    """
    Decodes MP3 audio into NumPy float32 samples using FFmpeg.

    Output format:
        Sample rate: 24000 Hz
        Channels: 2
        Data type: float32
        Shape: (frames, channels)
    """

    def __init__(
        self,
        sample_rate=24000,
        channels=2,
        ffmpeg_path="ffmpeg"
    ):

        self.sample_rate = sample_rate
        self.channels = channels
        self.ffmpeg_path = ffmpeg_path

    # ========================================================
    # SHARED FFmpeg COMMAND
    # ========================================================

    def _build_ffmpeg_command(self):

        return [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "pipe:1"
        ]

    # ========================================================
    # EXISTING COMPLETE MP3 DECODER
    # ========================================================

    def decode_mp3(self, mp3_data: bytes):
        """
        Decode complete MP3 bytes into PCM audio samples.
        """

        if not mp3_data:
            raise ValueError("MP3 data cannot be empty.")

        try:

            result = subprocess.run(
                self._build_ffmpeg_command(),
                input=mp3_data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=120
            )

        except FileNotFoundError as error:

            raise RuntimeError(
                "FFmpeg executable was not found. "
                "Check your FFmpeg installation and PATH."
            ) from error

        except subprocess.TimeoutExpired as error:

            raise RuntimeError(
                "FFmpeg decoding timed out."
            ) from error

        except subprocess.CalledProcessError as error:

            message = error.stderr.decode(
                errors="replace"
            )

            raise RuntimeError(
                f"FFmpeg decoding failed: {message}"
            ) from error

        if not result.stdout:
            raise RuntimeError(
                "FFmpeg returned empty audio output."
            )

        return self._pcm_bytes_to_array(result.stdout), self.sample_rate

    # ========================================================
    # PCM CONVERSION
    # ========================================================

    def _pcm_bytes_to_array(self, pcm_data):
        """
        Convert raw float32 PCM bytes into a NumPy array.
        """

        bytes_per_frame = 4 * self.channels

        if len(pcm_data) % bytes_per_frame != 0:
            raise RuntimeError(
                "Decoded PCM data contains an incomplete frame."
            )

        audio = np.frombuffer(
            pcm_data,
            dtype="<f4"
        )

        audio = audio.reshape(
            -1,
            self.channels
        )

        if not np.isfinite(audio).all():
            raise RuntimeError(
                "Decoded audio contains invalid values."
            )

        return audio

    # ========================================================
    # STREAMING MP3 DECODER
    # ========================================================

    def decode_mp3_stream(
        self,
        mp3_chunks,
        on_pcm_chunk,
        stop_event=None,
        chunk_frames=4096,
        output_queue_chunks=8
    ):
        """
        Decode incoming MP3 chunks progressively.

        Args:
            mp3_chunks:
                Iterable yielding MP3 byte chunks.

            on_pcm_chunk:
                Callback receiving decoded NumPy float32 arrays.

            stop_event:
                Optional threading.Event used for cancellation.

            chunk_frames:
                Number of audio frames requested per PCM read.

            output_queue_chunks:
                Maximum number of decoded PCM chunks waiting
                for the consumer.

        Returns:
            Dictionary containing decoding statistics.
        """

        if not callable(on_pcm_chunk):
            raise TypeError(
                "on_pcm_chunk must be callable."
            )

        if chunk_frames <= 0:
            raise ValueError(
                "chunk_frames must be positive."
            )

        if output_queue_chunks <= 0:
            raise ValueError(
                "output_queue_chunks must be positive."
            )

        if stop_event is None:
            stop_event = threading.Event()

        if stop_event.is_set():

            return {
                "cancelled": True,
                "decoded_frames": 0,
                "decoded_chunks": 0,
                "decoded_seconds": 0.0
            }

        command = self._build_ffmpeg_command()

        try:

            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )

        except FileNotFoundError as error:

            raise RuntimeError(
                "FFmpeg executable was not found. "
                "Check your FFmpeg installation and PATH."
            ) from error

        # Bounded queue prevents unlimited decoded PCM accumulation.
        output_queue = queue.Queue(
            maxsize=output_queue_chunks
        )

        errors = []
        errors_lock = threading.Lock()

        sentinel = object()

        stats = {
            "decoded_frames": 0,
            "decoded_chunks": 0
        }

        def record_error(error):

            with errors_lock:
                errors.append(error)

        # ----------------------------------------------------
        # QUEUE HELPERS
        # ----------------------------------------------------

        def put_pcm(data):

            while not stop_event.is_set():

                try:

                    output_queue.put(
                        data,
                        timeout=0.1
                    )

                    return True

                except queue.Full:
                    continue

            return False

        def put_sentinel():

            while not stop_event.is_set():

                try:

                    output_queue.put(
                        sentinel,
                        timeout=0.1
                    )

                    return

                except queue.Full:
                    continue

        # ----------------------------------------------------
        # READER THREAD
        # ----------------------------------------------------

        def read_pcm():

            try:

                bytes_per_frame = 4 * self.channels

                read_size = (
                    chunk_frames * bytes_per_frame
                )

                while not stop_event.is_set():

                    pcm_data = process.stdout.read(
                        read_size
                    )

                    if not pcm_data:
                        break

                    if not put_pcm(pcm_data):
                        break

            except Exception as error:

                record_error(error)

            finally:

                put_sentinel()

        # ----------------------------------------------------
        # ERROR READER THREAD
        # ----------------------------------------------------

        stderr_parts = []

        def read_stderr():

            try:

                while True:

                    data = process.stderr.read(4096)

                    if not data:
                        break

                    stderr_parts.append(data)

            except Exception as error:

                record_error(error)

        # ----------------------------------------------------
        # WRITER THREAD
        # ----------------------------------------------------

        def write_mp3():

            try:

                for mp3_chunk in mp3_chunks:

                    if stop_event.is_set():
                        break

                    if not isinstance(
                        mp3_chunk,
                        (bytes, bytearray, memoryview)
                    ):

                        raise TypeError(
                            "MP3 chunks must contain bytes."
                        )

                    if mp3_chunk:

                        process.stdin.write(
                            mp3_chunk
                        )

                        process.stdin.flush()

            except BrokenPipeError:
                # FFmpeg may exit before the input is exhausted.
                pass

            except Exception as error:

                record_error(error)

            finally:

                try:
                    process.stdin.close()
                except Exception:
                    pass

                # Close the generator from the same thread
                # that consumed it.
                close_iterator = getattr(
                    mp3_chunks,
                    "close",
                    None
                )

                if callable(close_iterator):

                    try:
                        close_iterator()
                    except Exception as error:
                        record_error(error)

        # ----------------------------------------------------
        # START WORKER THREADS
        # ----------------------------------------------------

        reader_thread = threading.Thread(
            target=read_pcm,
            name="FFmpegPCMReader",
            daemon=True
        )

        stderr_thread = threading.Thread(
            target=read_stderr,
            name="FFmpegErrorReader",
            daemon=True
        )

        writer_thread = threading.Thread(
            target=write_mp3,
            name="FFmpegMP3Writer",
            daemon=True
        )

        reader_thread.start()
        stderr_thread.start()
        writer_thread.start()

        # ----------------------------------------------------
        # CONSUME DECODED PCM
        # ----------------------------------------------------

        pending_bytes = bytearray()

        bytes_per_frame = 4 * self.channels

        callback_error = None

        try:

            while not stop_event.is_set():

                try:

                    pcm_data = output_queue.get(
                        timeout=0.1
                    )

                except queue.Empty:
                    continue

                if pcm_data is sentinel:
                    break

                pending_bytes.extend(pcm_data)

                usable_bytes = (
                    len(pending_bytes)
                    - len(pending_bytes) % bytes_per_frame
                )

                if usable_bytes == 0:
                    continue

                usable_data = bytes(
                    pending_bytes[:usable_bytes]
                )

                del pending_bytes[:usable_bytes]

                audio = self._pcm_bytes_to_array(
                    usable_data
                )

                if len(audio) == 0:
                    continue

                if stop_event.is_set():
                    break

                on_pcm_chunk(audio)

                stats["decoded_frames"] += len(audio)
                stats["decoded_chunks"] += 1

        except Exception as error:

            callback_error = error

            # Stop workers if the consumer callback fails.
            stop_event.set()

        finally:

            # ------------------------------------------------
            # PROCESS CLEANUP
            # ------------------------------------------------

            if stop_event.is_set() and process.poll() is None:

                try:
                    process.terminate()
                except Exception:
                    pass

            # Wait for the input worker. Audius HTTP reads
            # may take until their configured timeout.
            writer_thread.join(timeout=35)

            # Ensure FFmpeg cannot remain running indefinitely.
            if process.poll() is None:

                try:

                    process.wait(timeout=10)

                except subprocess.TimeoutExpired:

                    process.kill()
                    process.wait()

            # FFmpeg termination should release its output pipes.
            reader_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

            # Final process cleanup if still running.
            if process.poll() is None:

                process.kill()
                process.wait()

            # Close remaining pipe handles.
            for pipe in (
                process.stdin,
                process.stdout,
                process.stderr
            ):

                if pipe is not None:

                    try:
                        pipe.close()
                    except Exception:
                        pass

        # ----------------------------------------------------
        # ERROR HANDLING
        # ----------------------------------------------------

        if callback_error is not None:
            raise callback_error

        if stop_event.is_set():

            return {
                "cancelled": True,
                "decoded_frames": stats["decoded_frames"],
                "decoded_chunks": stats["decoded_chunks"],
                "decoded_seconds": round(
                    stats["decoded_frames"] / self.sample_rate,
                    3
                )
            }

        if writer_thread.is_alive():
            raise RuntimeError(
                "MP3 input worker did not terminate."
            )

        if reader_thread.is_alive():
            raise RuntimeError(
                "FFmpeg PCM reader did not terminate."
            )

        if stderr_thread.is_alive():
            raise RuntimeError(
                "FFmpeg error reader did not terminate."
            )

        if errors:

            raise RuntimeError(
                f"Streaming decoder worker failed: {errors[0]}"
            ) from errors[0]

        return_code = process.returncode

        if return_code != 0:

            message = b"".join(
                stderr_parts
            ).decode(
                errors="replace"
            )

            raise RuntimeError(
                f"FFmpeg streaming decoder failed: {message}"
            )

        return {
            "cancelled": False,
            "decoded_frames": stats["decoded_frames"],
            "decoded_chunks": stats["decoded_chunks"],
            "decoded_seconds": round(
                stats["decoded_frames"] / self.sample_rate,
                3
            )
        }


# ============================================================
# BASIC DECODER TEST
# ============================================================

def main():

    print("=" * 55)
    print("AUDIO DECODER TEST")
    print("=" * 55)

    print("\nGenerating a 5-second test audio signal...")

    sample_rate = 24000
    duration = 5

    time = np.arange(
        sample_rate * duration
    ) / sample_rate

    audio = (
        0.3 * np.sin(2 * np.pi * 440 * time)
    ).astype(np.float32)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "f32le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-i", "pipe:0",
        "-codec:a", "libmp3lame",
        "-b:a", "128k",
        "-f", "mp3",
        "pipe:1"
    ]

    result = subprocess.run(
        command,
        input=audio.tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=30
    )

    mp3_data = result.stdout

    print(f"MP3 generated: {len(mp3_data):,} bytes")

    print("\nTesting complete MP3 decoding...")

    decoder = AudioDecoder(
        sample_rate=sample_rate,
        channels=2
    )

    decoded_audio, decoded_sample_rate = (
        decoder.decode_mp3(mp3_data)
    )

    print("Complete decoding successful.")
    print(f"Sample rate: {decoded_sample_rate} Hz")
    print(f"Shape: {decoded_audio.shape}")

    print("\nTesting streaming MP3 decoding...")

    decoded_chunks = []

    def collect_audio(chunk):
        decoded_chunks.append(chunk)

    def generate_mp3_chunks():

        chunk_size = 4096

        for position in range(
            0,
            len(mp3_data),
            chunk_size
        ):

            yield mp3_data[
                position:position + chunk_size
            ]

    stats = decoder.decode_mp3_stream(
        mp3_chunks=generate_mp3_chunks(),
        on_pcm_chunk=collect_audio
    )

    if not decoded_chunks:
        raise RuntimeError(
            "Streaming decoder produced no audio chunks."
        )

    streamed_audio = np.concatenate(
        decoded_chunks,
        axis=0
    )

    print("\nStreaming decoding successful.")
    print(f"Decoded chunks: {stats['decoded_chunks']}")
    print(f"Decoded frames: {stats['decoded_frames']:,}")
    print(f"Decoded duration: {stats['decoded_seconds']} seconds")
    print(f"Combined shape: {streamed_audio.shape}")
    print(f"Data type: {streamed_audio.dtype}")

    print("\nAudio decoder test completed successfully.")


if __name__ == "__main__":
    main()
