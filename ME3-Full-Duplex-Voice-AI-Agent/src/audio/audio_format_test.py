
import sounddevice as sd
import numpy as np


# Audio configuration for Silero VAD
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_SIZE = 512

MICROPHONE_DEVICE = 1


def audio_callback(indata, frames, time, status):

    if status:
        print("Audio status:", status)

    # Copy the incoming audio
    audio_data = indata.copy()

    # Calculate RMS
    rms = np.sqrt(np.mean(np.square(audio_data)))

    print(
        f"Frames: {frames} | "
        f"Shape: {audio_data.shape} | "
        f"Data type: {audio_data.dtype} | "
        f"RMS: {rms:.6f}"
    )


def main():

    print("Starting 16 kHz audio capture...")
    print("Speak into your microphone.")
    print("Press CTRL+C to stop.\n")

    try:

        with sd.InputStream(
            device=MICROPHONE_DEVICE,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=FRAME_SIZE,
            callback=audio_callback
        ):

            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\nAudio capture stopped.")


if __name__ == "__main__":
    main()
