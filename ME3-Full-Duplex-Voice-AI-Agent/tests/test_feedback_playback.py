
import time

from src.agent.coordinator import ConversationCoordinator


def main():

    coordinator = ConversationCoordinator()

    phrases = [
        "Mm-hmm.",
        "Yeah.",
        "Aha.",
        "Right."
    ]

    try:

        print("\n========== BACKCHANNEL PLAYBACK TEST ==========")

        for index, phrase in enumerate(phrases, start=1):

            print(f"\nTEST {index}/4: {phrase}")

            result = coordinator.speak_feedback(phrase)

            print("Playback result:", result)

            time.sleep(0.5)

        print("\n========== TEST COMPLETED ==========")

    finally:

        coordinator.close()


if __name__ == "__main__":
    main()
