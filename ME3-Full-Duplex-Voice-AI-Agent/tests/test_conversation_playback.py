
from src.agent.coordinator import ConversationCoordinator


def main():

    coordinator = None

    try:

        print("\nInitializing conversation coordinator...")

        coordinator = ConversationCoordinator()

        prompt = (
            "Explain what a Raspberry Pi is "
            "in two short sentences."
        )

        print("\nSending prompt to Gemini...")

        response = coordinator.process_transcript(
            prompt
        )

        print("\n========== GEMINI RESPONSE ==========")
        print(response)
        print("=====================================")

        print("\nGenerating and playing speech...")

        coordinator.speak_response(response)

        print("\n========== CONVERSATION PLAYBACK TEST SUCCESS ==========")

    except Exception as error:

        print("\nConversation playback test failed:", error)

    finally:

        if coordinator is not None:
            coordinator.close()


if __name__ == "__main__":
    main()
