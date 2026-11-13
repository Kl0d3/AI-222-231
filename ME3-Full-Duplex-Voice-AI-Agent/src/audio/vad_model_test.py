
import torch
import sounddevice as sd
from silero_vad import load_silero_vad


# Audio configuration
SAMPLE_RATE = 16000
FRAME_SIZE = 512
CHANNELS = 1

MICROPHONE_DEVICE = 1

# VAD configuration
SPEECH_THRESHOLD = 0.5


def main():

    # Use one CPU thread for inference.
    torch.set_num_threads(1)

    print("Loading Silero VAD model...")

    # Load the pretrained model.
    model = load_silero_vad()
    model.eval()

    print("Silero VAD model loaded successfully!\n")

    print("Starting microphone capture...")
    print("Speak normally, speak quietly, then remain silent.")
    print("Press CTRL+C to stop.\n")

    def audio_callback(indata, frames, time, status):

        if status:
            print("Audio status:", status)

        # Convert microphone audio into a PyTorch tensor.
        audio_data = torch.from_numpy(
            indata.copy().flatten()
        )

        # Run Silero inference.
        with torch.no_grad():
            probability = model(
                audio_data,
                SAMPLE_RATE
            ).item()

        # Classify the audio frame.
        if probability >= SPEECH_THRESHOLD:
            classification = "SPEECH"
        else:
            classification = "NON-SPEECH"

        print(
            f"Probability: {probability:.4f} | "
            f"Detection: {classification}"
        )

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
        print("\nMicrophone capture stopped.")


if __name__ == "__main__":
    main()
