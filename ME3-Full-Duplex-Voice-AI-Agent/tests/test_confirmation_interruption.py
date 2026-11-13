import queue
import threading
from unittest.mock import Mock

from src.agent.coordinator import ConversationCoordinator


def create_coordinator(interrupt=False):

    coordinator = ConversationCoordinator.__new__(
        ConversationCoordinator
    )

    coordinator.gemini_client = Mock()
    coordinator.playback_client = Mock()

    coordinator.audio_queue = queue.Queue()

    coordinator.interruption_event = threading.Event()
    coordinator.speech_lock = threading.Lock()

    coordinator.tts_error = None
    coordinator.playback_error = None

    coordinator.last_chunk_time = None
    coordinator.chunk_intervals = []

    # Simulated TTS generation.
    def fake_generation_worker(sentences):
        coordinator.audio_queue.put(None)

    # Simulated playback.
    def fake_playback_worker():

        while True:

            item = coordinator.audio_queue.get()

            try:

                if item is None:
                    break

            finally:

                coordinator.audio_queue.task_done()

        if interrupt:
            coordinator.interruption_event.set()

    coordinator.tts_generation_worker = fake_generation_worker
    coordinator.audio_playback_worker = fake_playback_worker

    return coordinator


def test_interrupted_confirmation():

    print("\nTEST 1: INTERRUPTED CONFIRMATION QUESTION")

    coordinator = create_coordinator(interrupt=True)

    coordinator.speak_response(
        "Should I create this task?"
    )

    coordinator.gemini_client.discard_pending_interaction.assert_called_once()

    coordinator.gemini_client.cancel_pending_action.assert_called_once()

    coordinator.gemini_client.commit_pending_interaction.assert_not_called()

    print("PASS: Interrupted interaction discarded.")
    print("PASS: Pending operation cancelled.")
    print("PASS: Interaction was not committed.")


def test_completed_confirmation():

    print("\nTEST 2: COMPLETED CONFIRMATION QUESTION")

    coordinator = create_coordinator(interrupt=False)

    coordinator.speak_response(
        "Should I create this task?"
    )

    coordinator.gemini_client.commit_pending_interaction.assert_called_once()

    coordinator.gemini_client.discard_pending_interaction.assert_not_called()

    coordinator.gemini_client.cancel_pending_action.assert_not_called()

    print("PASS: Completed interaction committed.")
    print("PASS: Pending operation remains available.")
    print("PASS: Operation was not cancelled.")


def main():

    print("=" * 60)
    print("CONFIRMATION INTERRUPTION TESTS")
    print("=" * 60)

    test_interrupted_confirmation()
    test_completed_confirmation()

    print("\n" + "=" * 60)
    print("ALL INTERRUPTION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()