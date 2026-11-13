
from unittest.mock import Mock

from src.agent.coordinator import ConversationCoordinator


def create_coordinator():

    coordinator = ConversationCoordinator.__new__(
        ConversationCoordinator
    )

    coordinator.gemini_client = Mock()

    return coordinator


def test_affirmative_confirmation():

    print("\nTEST 1: AFFIRMATIVE CONFIRMATION")

    coordinator = create_coordinator()

    coordinator.gemini_client.pending_action = {
        "tool": "create_task",
        "arguments": {"title": "Review project proposal"}
    }

    coordinator.gemini_client.execute_pending_action.return_value = {
        "success": True,
        "message": "Task created successfully."
    }

    coordinator.gemini_client.generate_response.return_value = (
        "The task has been created successfully."
    )

    response = coordinator.process_transcript("Yes, go ahead.")

    coordinator.gemini_client.execute_pending_action.assert_called_once()

    _, kwargs = coordinator.gemini_client.generate_response.call_args

    assert kwargs["allow_tools"] is False
    assert response == "The task has been created successfully."

    print("PASS: Confirmed operation executed once.")
    print("PASS: Final response generated without tools.")


def test_negative_confirmation():

    print("\nTEST 2: NEGATIVE CONFIRMATION")

    coordinator = create_coordinator()

    coordinator.gemini_client.pending_action = {
        "tool": "delete_task",
        "arguments": {"task_id": "123"}
    }

    coordinator.gemini_client.generate_response.return_value = (
        "Okay, I cancelled that operation."
    )

    coordinator.process_transcript("No")

    coordinator.gemini_client.cancel_pending_action.assert_called_once()
    coordinator.gemini_client.execute_pending_action.assert_not_called()
    coordinator.gemini_client.generate_response.assert_called_once()

    _, kwargs = coordinator.gemini_client.generate_response.call_args

    assert kwargs["allow_tools"] is False

    print("PASS: Pending operation cancelled.")
    print("PASS: Protected operation was not executed.")


def test_unrelated_request():

    print("\nTEST 3: UNRELATED REQUEST")

    coordinator = create_coordinator()

    coordinator.gemini_client.pending_action = {
        "tool": "create_task",
        "arguments": {"title": "Old task"}
    }

    coordinator.gemini_client.generate_response.return_value = (
        "Here is the weather."
    )

    coordinator.process_transcript("What's the weather today?")

    coordinator.gemini_client.cancel_pending_action.assert_called_once()
    coordinator.gemini_client.execute_pending_action.assert_not_called()

    coordinator.gemini_client.generate_response.assert_called_once_with(
        "What's the weather today?"
    )

    print("PASS: Old operation cancelled.")
    print("PASS: New request processed normally.")


def test_ambiguous_confirmation():

    print("\nTEST 4: AMBIGUOUS CONFIRMATION")

    coordinator = create_coordinator()

    result = coordinator.classify_confirmation(
        "Yes, but change the date."
    )

    assert result is None

    print("PASS: Mixed response is not treated as confirmation.")


def test_confirmation_classification():

    print("\nTEST 5: CONFIRMATION CLASSIFICATION")

    coordinator = create_coordinator()

    assert coordinator.classify_confirmation("Yes") == "yes"
    assert coordinator.classify_confirmation("Go ahead") == "yes"
    assert coordinator.classify_confirmation("Confirm") == "yes"

    assert coordinator.classify_confirmation("No") == "no"
    assert coordinator.classify_confirmation("Cancel it") == "no"
    assert coordinator.classify_confirmation("Never mind") == "no"

    assert coordinator.classify_confirmation("Tell me a joke") is None

    print("PASS: Affirmative phrases recognized.")
    print("PASS: Negative phrases recognized.")
    print("PASS: Unrelated phrases rejected.")


def test_execution_failure():

    print("\nTEST 6: EXECUTION FAILURE")

    coordinator = create_coordinator()

    coordinator.gemini_client.pending_action = {
        "tool": "create_task",
        "arguments": {"title": "Test task"}
    }

    coordinator.gemini_client.execute_pending_action.side_effect = (
        RuntimeError("Simulated MCP failure.")
    )

    coordinator.gemini_client.generate_response.return_value = (
        "I couldn't complete that operation."
    )

    coordinator.process_transcript("Yes")

    coordinator.gemini_client.execute_pending_action.assert_called_once()
    coordinator.gemini_client.generate_response.assert_called_once()

    _, kwargs = coordinator.gemini_client.generate_response.call_args

    assert kwargs["allow_tools"] is False

    print("PASS: Execution failure handled.")
    print("PASS: Final response generated without tools.")


def test_confirmed_website_deletion():

    print("\nTEST 7: CONFIRMED WEBSITE DELETION")

    coordinator = create_coordinator()

    coordinator.gemini_client.pending_action = {
        "tool": "delete_website_project",
        "arguments": {
            "project_name": "my-portfolio"
        }
    }

    coordinator.gemini_client.execute_pending_action.return_value = {
        "success": True,
        "message": "Website project deleted successfully."
    }

    coordinator.gemini_client.generate_response.return_value = (
        "The website project has been deleted."
    )

    response = coordinator.process_transcript("Yes, go ahead.")

    coordinator.gemini_client.execute_pending_action.assert_called_once()

    coordinator.gemini_client.generate_response.assert_called_once()

    _, kwargs = coordinator.gemini_client.generate_response.call_args

    assert kwargs["allow_tools"] is False

    assert response == (
        "The website project has been deleted."
    )

    print("PASS: Website deletion confirmation accepted.")
    print("PASS: Pending operation executed exactly once.")
    print("PASS: Final response generated without tools.")


def main():

    print("=" * 60)
    print("COORDINATOR CONFIRMATION TESTS")
    print("=" * 60)

    test_affirmative_confirmation()
    test_negative_confirmation()
    test_unrelated_request()
    test_ambiguous_confirmation()
    test_confirmation_classification()
    test_execution_failure()
    test_confirmed_website_deletion()

    print("\n" + "=" * 60)
    print("ALL 7 COORDINATOR CONFIRMATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
