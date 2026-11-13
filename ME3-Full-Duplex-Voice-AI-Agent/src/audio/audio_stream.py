
import sounddevice as sd
import numpy as np


# Audio configuration
SAMPLE_RATE = 44100
CHANNELS = 1
BLOCK_SIZE = 1024

MICROPHONE_DEVICE = 1


def audio_callback(indata, frames, time, status):
    """
    Called automatically whenever a new audio block arrives.
    """

    if status:
        print("Audio status:", status)

    # Copy the incoming audio block
    audio_data = indata.copy()

    # Calculate RMS of this audio block
    rms = np.sqrt(np.mean(np.square(audio_data)))

    print(
        f"Frames: {frames} | "
        f"Shape: {audio_data.shape} | "
        f"RMS: {rms:.6f}"
    )


def main():

    print("Starting continuous microphone capture...")
    print("Speak into your microphone.")
    print("Press CTRL+C to stop.\n")

    try:

        with sd.InputStream(
            device=MICROPHONE_DEVICE,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=audio_callback
        ):

            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\nMicrophone capture stopped.")


if __name__ == "__main__":
    main()
