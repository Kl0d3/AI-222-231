
import numpy as np
import sounddevice as sd


DEVICE = 3
SAMPLE_RATE = 24000
DURATION = 2

print("Testing 24 kHz audio output...")
print("Device:", sd.query_devices(DEVICE)["name"])

frequency = 440

time_axis = np.arange(
    SAMPLE_RATE * DURATION
) / SAMPLE_RATE

tone = 0.2 * np.sin(
    2 * np.pi * frequency * time_axis
)

audio = tone.astype(np.float32)

sd.play(
    audio,
    samplerate=SAMPLE_RATE,
    device=DEVICE
)

sd.wait()

print("24 kHz speaker test completed.")
