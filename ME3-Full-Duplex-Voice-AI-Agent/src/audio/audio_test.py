
import sounddevice as sd
import numpy as np
import wave
from pathlib import Path

# Audio configuration
SAMPLE_RATE = 44100
DURATION = 15

MICROPHONE_DEVICE = 1
SPEAKER_DEVICE = 3

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 2

# Determine project root and output location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "audio"
OUTPUT_FILE = OUTPUT_DIR / "sample3.wav"


def calculate_rms(audio_data):
    """
    Calculate the Root Mean Square amplitude.
    """
    squared_samples = np.square(audio_data)
    mean_squared = np.mean(squared_samples)
    rms = np.sqrt(mean_squared)

    return rms


def save_wav(audio_data, output_path):
    """
    Save float32 audio as a 16-bit PCM WAV file.
    """

    # Convert float32 audio to 16-bit PCM
    audio_int16 = (
        np.clip(audio_data, -1.0, 1.0) * 32767
    ).astype(np.int16)

    # Write WAV file
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(INPUT_CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_int16.tobytes())


def main():

    print("Recording starts in 2 seconds...")
    sd.sleep(2000)

    print("Recording... Speak now!")

    audio_data = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=INPUT_CHANNELS,
        dtype="float32",
        device=MICROPHONE_DEVICE
    )

    sd.wait()

    print("Recording finished!")
    print("Audio shape:", audio_data.shape)

    rms = calculate_rms(audio_data)

    print(f"RMS amplitude: {rms:.6f}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save recording
    save_wav(audio_data, OUTPUT_FILE)

    print(f"WAV file saved successfully: {OUTPUT_FILE}")

    print("Playing recording...")

    sd.play(
        audio_data,
        samplerate=SAMPLE_RATE,
        device=SPEAKER_DEVICE
    )

    sd.wait()

    print("Playback finished!")


if __name__ == "__main__":
    main()
