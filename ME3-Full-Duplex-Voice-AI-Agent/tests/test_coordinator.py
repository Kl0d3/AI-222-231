
from src.agent.coordinator import ConversationCoordinator


def main():

    coordinator = ConversationCoordinator()

    try:
        transcript = "What is a Raspberry Pi?"

        print("\n========== USER TRANSCRIPT ==========")
        print(transcript)

        print("\nSending transcript to coordinator...")

        response = coordinator.process_transcript(transcript)

        print("\n========== GEMINI RESPONSE ==========")
        print(response)
        print("=====================================")

    except Exception as error:
        print(f"\nCoordinator test failed: {error}")

    finally:
        coordinator.close()


if __name__ == "__main__":
    main()
