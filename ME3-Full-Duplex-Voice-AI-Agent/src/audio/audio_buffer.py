
import math
import queue
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad


class AudioBuffer:

    def __init__(
        self,
        sample_rate=16000,
        channels=1,
        block_size=512,
        microphone_device=1,
        speaker_device=3,
        queue_max_size=50
    ):

        # Audio configuration
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size

        self.microphone_device = microphone_device
        self.speaker_device = speaker_device

        # Audio queues
        self.audio_queue = queue.Queue(
            maxsize=queue_max_size
        )

        self.flux_audio_queue = queue.Queue(
            maxsize=queue_max_size
        )

        self.speech_event_queue = queue.Queue()
        self.utterance_queue = queue.Queue()

        # Pre-roll buffer
        self.pre_roll_frames = 10

        self.pre_roll_buffer = deque(
            maxlen=self.pre_roll_frames
        )

        # Thread synchronization
        self.stop_event = threading.Event()

        self.audio_worker_finished = threading.Event()
        self.deepgram_worker_finished = threading.Event()

        self.stats_lock = threading.Lock()

        # VAD configuration
        self.start_threshold = 0.60
        self.stop_threshold = 0.35

        self.min_speech_frames = 4
        self.min_silence_duration = 0.5

        self.frame_duration = (
            self.block_size / self.sample_rate
        )

        self.min_silence_frames = math.ceil(
            self.min_silence_duration / self.frame_duration
        )

        # Statistics
        self.processed_frames = 0

        self.dropped_frames = 0
        self.highest_queue_size = 0

        self.flux_dropped_frames = 0
        self.highest_flux_queue_size = 0

        self.generated_events = 0
        self.consumed_events = 0

        self.completed_utterances = 0

        # External components
        self.vad_model = None
        self.deepgram_streamer = None

    # ====================================================
    # Microphone callback
    # ====================================================

    def audio_callback(self, indata, frames, time_info, status):

        if status:
            print("[AUDIO WARNING]", status)

        audio_data = indata.copy()

        # Silero VAD queue
        try:

            self.audio_queue.put_nowait(audio_data)

            with self.stats_lock:

                current_size = self.audio_queue.qsize()

                if current_size > self.highest_queue_size:
                    self.highest_queue_size = current_size

        except queue.Full:

            with self.stats_lock:
                self.dropped_frames += 1

        # Deepgram Flux queue
        try:

            self.flux_audio_queue.put_nowait(audio_data)

            with self.stats_lock:

                current_size = self.flux_audio_queue.qsize()

                if current_size > self.highest_flux_queue_size:
                    self.highest_flux_queue_size = current_size

        except queue.Full:

            with self.stats_lock:
                self.flux_dropped_frames += 1

    # ====================================================
    # Speech event generation
    # ====================================================

    def emit_speech_event(self, event_name):

        event = {
            "type": event_name,
            "timestamp": time.monotonic()
        }

        self.speech_event_queue.put(event)

        with self.stats_lock:
            self.generated_events += 1

    # ====================================================
    # Utterance saving
    # ====================================================

    def save_utterance(self, frames):

        if not frames:
            return

        audio_segment = np.concatenate(
            frames,
            axis=0
        )

        duration = (
            len(audio_segment) / self.sample_rate
        )

        utterance = {
            "audio": audio_segment,
            "sample_rate": self.sample_rate,
            "duration": duration,
            "frames": len(frames)
        }

        self.utterance_queue.put(utterance)

        with self.stats_lock:
            self.completed_utterances += 1

        print("\n[VAD] Utterance saved.")
        print("[VAD] Duration:", round(duration, 2), "seconds")

    # ====================================================
    # Silero VAD worker
    # ====================================================

    def audio_worker(self):

        """
        Continuously processes microphone frames using
        the Silero Voice Activity Detection model.
        """

        try:

            if self.vad_model is None:
                raise RuntimeError(
                    "Silero VAD model has not been initialized."
                )

            is_speaking = False

            speech_frame_count = 0
            silence_frame_count = 0

            utterance_frames = []
            pending_silence_frames = []

            while (
                not self.stop_event.is_set()
                or not self.audio_queue.empty()
            ):

                try:

                    audio_data = self.audio_queue.get(
                        timeout=0.1
                    )

                except queue.Empty:
                    continue

                try:

                    self.pre_roll_buffer.append(
                        audio_data.copy()
                    )

                    audio_tensor = torch.from_numpy(
                        audio_data.flatten()
                    )

                    with torch.no_grad():

                        speech_probability = self.vad_model(
                            audio_tensor,
                            self.sample_rate
                        ).item()

                    # Waiting for speech
                    if not is_speaking:

                        if speech_probability >= self.start_threshold:
                            speech_frame_count += 1
                        else:
                            speech_frame_count = 0

                        if speech_frame_count >= self.min_speech_frames:

                            is_speaking = True

                            speech_frame_count = 0
                            silence_frame_count = 0

                            utterance_frames = [
                                frame.copy()
                                for frame in self.pre_roll_buffer
                            ]

                            pending_silence_frames = []

                            print("\n[VAD] Speech started.")

                            self.emit_speech_event(
                                "SPEECH_STARTED"
                            )

                    # Currently speaking
                    else:

                        if speech_probability < self.stop_threshold:

                            silence_frame_count += 1

                            pending_silence_frames.append(
                                audio_data.copy()
                            )

                        else:

                            utterance_frames.extend(
                                pending_silence_frames
                            )

                            pending_silence_frames.clear()

                            silence_frame_count = 0

                            utterance_frames.append(
                                audio_data.copy()
                            )

                        # End of speech
                        if (
                            silence_frame_count
                            >= self.min_silence_frames
                        ):

                            is_speaking = False

                            speech_frame_count = 0
                            silence_frame_count = 0

                            print("\n[VAD] Speech stopped.")

                            self.save_utterance(
                                utterance_frames
                            )

                            utterance_frames = []
                            pending_silence_frames = []

                            self.emit_speech_event(
                                "SPEECH_STOPPED"
                            )

                    # Update statistics silently.
                    with self.stats_lock:

                        self.processed_frames += 1

                finally:

                    self.audio_queue.task_done()

            # Save unfinished utterance during shutdown
            if is_speaking and utterance_frames:

                utterance_frames.extend(
                    pending_silence_frames
                )

                self.save_utterance(
                    utterance_frames
                )

        except Exception as error:

            print("\n[VAD ERROR]", error)

        finally:

            self.audio_worker_finished.set()

            print("\n[VAD] Audio worker stopped.")

    # ====================================================
    # Silero VAD initialization
    # ====================================================

    def load_vad_model(self):

        torch.set_num_threads(1)

        print("\n[VAD] Loading Silero VAD model...")

        self.vad_model = load_silero_vad()

        self.vad_model.eval()

        print("[VAD] Silero VAD model loaded successfully!")


# ========================================================
# Live microphone + VAD worker test
# ========================================================

def main():

    audio = AudioBuffer()

    print("\nAudioBuffer initialized successfully!")

    audio.load_vad_model()

    vad_thread = threading.Thread(
        target=audio.audio_worker,
        name="Silero-VAD-Worker",
        daemon=True
    )

    vad_thread.start()

    print("\n========== LIVE VAD TEST ==========")
    print("Microphone device:", audio.microphone_device)
    print("Sample rate:", audio.sample_rate)
    print("Block size:", audio.block_size)

    print("\nSpeak into your microphone.")
    print("Pause for at least one second between sentences.")
    print("Press CTRL+C to stop the test.\n")

    try:

        with sd.InputStream(
            samplerate=audio.sample_rate,
            channels=audio.channels,
            blocksize=audio.block_size,
            dtype="float32",
            device=audio.microphone_device,
            callback=audio.audio_callback
        ):

            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:

        print("\n\nStopping microphone capture...")

    except Exception as error:

        print("\nMicrophone error:", error)

    finally:

        audio.stop_event.set()

        vad_thread.join(timeout=5)

        print("\n========== FINAL VAD STATISTICS ==========")

        print(
            "Processed frames:",
            audio.processed_frames
        )

        print(
            "Dropped frames:",
            audio.dropped_frames
        )

        print(
            "Completed utterances:",
            audio.completed_utterances
        )

        print(
            "Generated speech events:",
            audio.generated_events
        )

        print("\nLive VAD test finished.")


if __name__ == "__main__":

    main()
