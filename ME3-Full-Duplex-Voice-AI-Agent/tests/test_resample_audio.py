
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


# File paths
AUDIO_DIR = Path(__file__).parent / "audio"

INPUT_FILE = AUDIO_DIR / "sample3.wav"
OUTPUT_FILE = AUDIO_DIR / "sample3_16k.wav"

# Target audio configuration
TARGET_SAMPLE_RATE = 16000
UP = 160
DOWN = 441


print("Reading original audio...")

# Read the original WAV file
with wave.open(str(INPUT_FILE), "rb") as wav:
    channels = wav.getnchannels()
    sample_rate = wav.getframerate()
    sample_width = wav.getsampwidth()
    frames = wav.getnframes()
    audio_bytes = wav.readframes(frames)

# Validate the original audio format
if channels != 1:
    raise ValueError("Expected mono audio.")

if sample_width != 2:
    raise ValueError("Expected 16-bit PCM audio.")

if sample_rate != 44100:
    raise ValueError("Expected an original sample rate of 44100 Hz.")

print(f"Original sample rate: {sample_rate} Hz")
print(f"Original duration: {frames / sample_rate:.2f} seconds")

# Convert raw PCM bytes into NumPy int16 samples
audio_int16 = np.frombuffer(audio_bytes, dtype="<i2")

# Convert integer samples to floating-point values
audio_float = audio_int16.astype(np.float32) / 32768.0

print("\nResampling audio...")

# Resample from 44,100 Hz to 16,000 Hz
resampled_audio = resample_poly(
    audio_float,
    up=UP,
    down=DOWN
)

# Convert back to 16-bit PCM
resampled_audio = np.clip(
    np.rint(resampled_audio * 32768.0),
    -32768,
    32767
).astype("<i2")

# Save the resampled audio as a WAV file
with wave.open(str(OUTPUT_FILE), "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(TARGET_SAMPLE_RATE)
    wav.writeframes(resampled_audio.tobytes())

print("\nResampling completed!")

# Verify the output file
with wave.open(str(OUTPUT_FILE), "rb") as wav:
    output_channels = wav.getnchannels()
    output_rate = wav.getframerate()
    output_width = wav.getsampwidth()
    output_frames = wav.getnframes()

print("\n========== OUTPUT AUDIO ==========")
print(f"File: {OUTPUT_FILE.name}")
print(f"Channels: {output_channels}")
print(f"Sample rate: {output_rate} Hz")
print(f"Sample width: {output_width} bytes")
print(f"Frames: {output_frames}")
print(f"Duration: {output_frames / output_rate:.2f} seconds")
print("==================================")
