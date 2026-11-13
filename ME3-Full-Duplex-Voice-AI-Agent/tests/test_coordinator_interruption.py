
import threading
import time

from src.agent.coordinator import ConversationCoordinator


def main():

    coordinator = None
    playback_thread = None

    try:

        print("\nInitializing conversation coordinator...")

        coordinator = ConversationCoordinator()

        print("\nGenerating original response...")

        first_response = coordinator.process_transcript(
            "Explain how a Raspberry Pi is used "
            "in electronics projects. Use three sentences."
        )

        print("\n========== ORIGINAL RESPONSE ==========")
        print(first_response)
        print("=======================================")

        playback_thread = threading.Thread(
            target=coordinator.speak_response,
            args=(first_response,)
        )

        print("\nStarting original speech...")

        playback_thread.start()

        # Wait until Kokoro finishes generating speech
        # and AudioPlayback actually starts playing.
        print("\nWaiting for audio playback to start...")

        playback_started = False

        for _ in range(300):

            if coordinator.playback_client._is_playing:
                playback_started = True
                break

            if not playback_thread.is_alive():
                raise RuntimeError(
                    "Speech thread stopped before playback started."
                )

            time.sleep(0.1)

        if not playback_started:
            raise RuntimeError(
                "Timed out waiting for audio playback to start."
            )

        print("\nPlayback is active!")

        # Allow the assistant to speak for two seconds.
        time.sleep(2)

        print("\nSimulating user interruption...")

        coordinator.stop_speech()

        playback_thread.join(timeout=5)

        if playback_thread.is_alive():
            raise RuntimeError(
                "Coordinator failed to interrupt playback."
            )

        print("\nOriginal response successfully interrupted.")

        print("\nGenerating a new response...")

        second_response = coordinator.process_transcript(
            "Now explain what an MCP server does "
            "in an AI assistant. Use two sentences."
        )

        print("\n========== NEW RESPONSE ==========")
        print(second_response)
        print("==================================")

        print("\nPlaying the new response...")

        coordinator.speak_response(second_response)

        print(
            "\n========== COORDINATOR INTERRUPTION TEST SUCCESS =========="
        )

    except Exception as error:

        print("\nCoordinator interruption test failed:", error)

    finally:

        if coordinator is not None:
            coordinator.close()


if __name__ == "__main__":
    main()
