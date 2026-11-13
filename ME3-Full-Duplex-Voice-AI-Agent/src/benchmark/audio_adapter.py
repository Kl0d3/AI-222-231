import numpy as np

from src.audio.audio_buffer import AudioBuffer


class BenchmarkAudioAdapter:
    """
    Receives benchmark PCM16 audio and forwards it to the
    existing AudioBuffer queues without opening a microphone.
    """

    def __init__(self, audio_buffer: AudioBuffer):
        self.audio_buffer = audio_buffer

        self.sample_rate = 16000
        self.channels = 1
        self.block_size = audio_buffer.block_size

        self._pending_samples = np.empty(0, dtype=np.float32)

    def process_pcm16(self, audio_bytes: bytes) -> None:
        """
        Convert incoming signed PCM16 audio into float32 samples
        and forward complete blocks to the existing audio queues.
        """

        if not audio_bytes:
            return

        # Interpret incoming audio as signed 16-bit PCM.
        pcm16 = np.frombuffer(audio_bytes, dtype="<i2")

        # Convert PCM16 range into float32 range [-1.0, 1.0].
        samples = pcm16.astype(np.float32) / 32768.0

        # Preserve incomplete blocks for the next incoming frame.
        self._pending_samples = np.concatenate(
            (self._pending_samples, samples)
        )

        # Forward complete blocks to the existing AudioBuffer.
        while len(self._pending_samples) >= self.block_size:

            block = self._pending_samples[:self.block_size]

            self._pending_samples = self._pending_samples[
                self.block_size:
            ]

            # Match the shape supplied by SoundDevice:
            # (frames, channels)
            audio_block = block.reshape(-1, 1)

            self.audio_buffer.audio_callback(
                indata=audio_block,
                frames=self.block_size,
                time_info=None,
                status=None
            )

    def flush(self) -> None:
        """
        Forward any remaining samples during stream shutdown.
        """

        if len(self._pending_samples) == 0:
            return

        remaining = self._pending_samples

        self._pending_samples = np.empty(
            0,
            dtype=np.float32
        )

        # Pad the final partial block with silence.
        padded = np.zeros(
            self.block_size,
            dtype=np.float32
        )

        padded[:len(remaining)] = remaining

        self.audio_buffer.audio_callback(
            indata=padded.reshape(-1, 1),
            frames=self.block_size,
            time_info=None,
            status=None
        )