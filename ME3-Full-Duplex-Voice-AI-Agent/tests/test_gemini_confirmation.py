
import threading
from unittest.mock import Mock

from src.llm.gemini_client import GeminiClient


def create_client():

    client = GeminiClient.__new__(GeminiClient)

    client.mcp_manager = Mock()
    client.pending_action = None
    client.pending_action_lock = threading.Lock()

    return client


def test_stage_protected_action():

    print("\nTEST 1: STAGE PROTECTED ACTION")

    client = create_client()

    function_call = Mock()
    function_call.name = "create_task"
    function_call.arguments = {
        "title": "Test confirmation",
        "notes": "This task must not be created yet."
    }

    result = client.stage_protected_action(function_call)

    assert result["requires_confirmation"] is True

    assert client.pending_action == {
        "tool": "create_task",
        "arguments": {
            "title": "Test confirmation",
            "notes": "This task must not be created yet."
        }
    }

    client.mcp_manager.call_tool.assert_not_called()

    print("PASS: Action staged successfully.")
    print("PASS: MCP operation was not executed.")


def test_prevent_multiple_pending_actions():

    print("\nTEST 2: PREVENT MULTIPLE PENDING ACTIONS")

    client = create_client()

    client.pending_action = {
        "tool": "create_task",
        "arguments": {"title": "Original task"}
    }

    function_call = Mock()
    function_call.name = "delete_task"
    function_call.arguments = {"task_id": "123"}

    result = client.stage_protected_action(function_call)

    assert result["success"] is False
    assert client.pending_action["tool"] == "create_task"

    client.mcp_manager.call_tool.assert_not_called()

    print("PASS: Existing pending action preserved.")
    print("PASS: Additional action was rejected.")


def test_cancel_pending_action():

    print("\nTEST 3: CANCEL PENDING ACTION")

    client = create_client()

    client.pending_action = {
        "tool": "delete_calendar_event",
        "arguments": {"event_id": "test-event"}
    }

    client.cancel_pending_action()

    assert client.pending_action is None

    client.mcp_manager.call_tool.assert_not_called()

    print("PASS: Pending action cancelled.")
    print("PASS: MCP operation was not executed.")


def test_execute_pending_action():

    print("\nTEST 4: EXECUTE CONFIRMED ACTION")

    client = create_client()

    expected_result = {
        "success": True,
        "message": "Task created successfully."
    }

    client.pending_action = {
        "tool": "create_task",
        "arguments": {
            "title": "Confirmed test task"
        }
    }

    client.mcp_manager.call_tool.return_value = expected_result

    result = client.execute_pending_action()

    assert result == expected_result

    client.mcp_manager.call_tool.assert_called_once_with(
        tool_name="create_task",
        arguments={
            "title": "Confirmed test task"
        }
    )

    assert client.pending_action is None

    print("PASS: Exact staged operation executed.")
    print("PASS: Pending action cleared after execution.")


def test_no_pending_action_execution():

    print("\nTEST 5: EXECUTION WITHOUT PENDING ACTION")

    client = create_client()

    try:
        client.execute_pending_action()

        raise AssertionError(
            "Expected RuntimeError was not raised."
        )

    except RuntimeError:
        pass

    client.mcp_manager.call_tool.assert_not_called()

    print("PASS: Execution rejected without pending action.")


def test_concurrent_execution():

    print("\nTEST 6: CONCURRENT EXECUTION PROTECTION")

    client = create_client()

    client.pending_action = {
        "tool": "create_task",
        "arguments": {
            "title": "Concurrent test task"
        }
    }

    expected_result = {
        "success": True,
        "message": "Task created successfully."
    }

    execution_started = threading.Event()
    allow_execution_to_finish = threading.Event()

    def mock_call_tool(tool_name, arguments):

        execution_started.set()

        if not allow_execution_to_finish.wait(timeout=5):
            raise TimeoutError(
                "Mock operation timed out."
            )

        return expected_result

    client.mcp_manager.call_tool.side_effect = mock_call_tool

    results = []
    errors = []

    def execute_action():

        try:
            results.append(
                client.execute_pending_action()
            )

        except Exception as error:
            errors.append(error)

    first_thread = threading.Thread(
        target=execute_action
    )

    second_thread = threading.Thread(
        target=execute_action
    )

    first_thread.start()

    assert execution_started.wait(timeout=5), (
        "First execution did not reach MCP."
    )

    second_thread.start()
    second_thread.join(timeout=5)

    allow_execution_to_finish.set()

    first_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()

    assert len(results) == 1
    assert results[0] == expected_result

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)

    client.mcp_manager.call_tool.assert_called_once_with(
        tool_name="create_task",
        arguments={
            "title": "Concurrent test task"
        }
    )

    assert client.pending_action is None

    print("PASS: First thread claimed the pending operation.")
    print("PASS: Second thread was rejected.")
    print("PASS: MCP operation executed exactly once.")


def test_website_deletion_requires_confirmation():

    print("\nTEST 7: WEBSITE DELETION CONFIRMATION")

    client = create_client()

    function_call = Mock()
    function_call.name = "delete_website_project"
    function_call.arguments = {
        "project_name": "my-portfolio"
    }

    result = client.execute_function_call(function_call)

    assert result["requires_confirmation"] is True

    assert client.pending_action == {
        "tool": "delete_website_project",
        "arguments": {
            "project_name": "my-portfolio"
        }
    }

    client.mcp_manager.call_tool.assert_not_called()

    print("PASS: Website deletion requires confirmation.")
    print("PASS: Project deletion was not executed.")


def main():

    print("=" * 60)
    print("GEMINI CONFIRMATION SAFETY TESTS")
    print("=" * 60)

    test_stage_protected_action()
    test_prevent_multiple_pending_actions()
    test_cancel_pending_action()
    test_execute_pending_action()
    test_no_pending_action_execution()
    test_concurrent_execution()
    test_website_deletion_requires_confirmation()

    print("\n" + "=" * 60)
    print("ALL 7 CONFIRMATION SAFETY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
