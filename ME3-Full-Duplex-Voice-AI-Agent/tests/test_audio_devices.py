
import sounddevice as sd


print("DEFAULT AUDIO DEVICE")
print(sd.default.device)

print("\nAVAILABLE OUTPUT DEVICES")

devices = sd.query_devices()

for index, device in enumerate(devices):

    if device["max_output_channels"] > 0:

        print(
            f"\nDevice index: {index}"
        )

        print(
            f"Name: {device['name']}"
        )

        print(
            f"Output channels: {device['max_output_channels']}"
        )

        print(
            f"Default sample rate: {device['default_samplerate']}"
        )
