
import torch
import sounddevice as sd
from silero_vad import load_silero_vad


# Audio configuration
SAMPLE_RATE = 16000
FRAME_SIZE = 512
CHANNELS = 1

MICROPHONE_DEVICE = 1


# VAD configuration
START_THRESHOLD = 0.60
STOP_THRESHOLD = 0.35

MIN_SPEECH_FRAMES = 4

# Require 500 ms of continuous silence before stopping.
MIN_SILENCE_DURATION = 0.5

FRAME_DURATION = FRAME_SIZE / SAMPLE_RATE

MIN_SILENCE_FRAMES = int(
    MIN_SILENCE_DURATION / FRAME_DURATION
)


def main():

    torch.set_num_threads(1)

    print("Loading Silero VAD model...")

    model = load_silero_vad()
    model.eval()

    print("Silero VAD model loaded successfully!\n")

    # Speech detection state
    is_speaking = False

    # Consecutive speech frames
    speech_frame_count = 0

    # Consecutive silence frames
    silence_frame_count = 0

    print("Starting microphone capture...")
    print("Speak, pause, say 'uhm', and speak again.")
    print("Press CTRL+C to stop.\n")

    def audio_callback(indata, frames, time, status):

        nonlocal is_speaking
        nonlocal speech_frame_count
        nonlocal silence_frame_count

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

        if not is_speaking:

            if probability >= START_THRESHOLD:
                speech_frame_count += 1
            else:
                speech_frame_count = 0

            # Confirm speech after enough consecutive frames.
            if speech_frame_count >= MIN_SPEECH_FRAMES:

                is_speaking = True
                silence_frame_count = 0

                print("\n>>> SPEECH STARTED <<<")

        else:

            if probability < STOP_THRESHOLD:

                silence_frame_count += 1

            else:

                # Reset silence counter when speech returns.
                silence_frame_count = 0

            # Stop only after sufficient continuous silence.
            if silence_frame_count >= MIN_SILENCE_FRAMES:

                is_speaking = False

                speech_frame_count = 0
                silence_frame_count = 0

                print("\n>>> SPEECH STOPPED <<<")

        state = "SPEAKING" if is_speaking else "NOT SPEAKING"

        print(
            f"Probability: {probability:.4f} | "
            f"State: {state} | "
            f"Silence frames: {silence_frame_count}"
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
