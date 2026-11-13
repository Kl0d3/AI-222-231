
from src.agent.coordinator import ConversationCoordinator


def main():

    coordinator = None

    try:
        # Initialize the conversation coordinator.
        print("\nInitializing conversation coordinator...")

        coordinator = ConversationCoordinator()

        # Simulate a recognized user transcript.
        transcript = "Explain what a Raspberry Pi is in two sentences."

        print("\n========== USER TRANSCRIPT ==========")
        print(transcript)

        # Generate the response using Gemini.
        print("\nGenerating response using Gemini...")

        response = coordinator.process_transcript(transcript)

        print("\n========== GEMINI RESPONSE ==========")
        print(response)
        print("=====================================")

        # Convert Gemini's response into speech.
        print("\nGenerating speech using Kokoro...")

        audio_path = coordinator.generate_speech(
            response,
            "outputs/conversation_response.wav"
        )

        print("\n========== INTEGRATION SUCCESS ==========")
        print(f"Generated audio: {audio_path}")
        print("=========================================")

    except Exception as error:
        print(f"\nIntegration test failed: {error}")

    finally:
        if coordinator is not None:
            coordinator.close()


if __name__ == "__main__":
    main()
