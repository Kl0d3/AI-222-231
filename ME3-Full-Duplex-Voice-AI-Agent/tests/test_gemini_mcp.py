import traceback

from src.mcp.background_manager import MCPBackgroundManager
from src.llm.gemini_client import GeminiClient


def main():

    mcp_manager = MCPBackgroundManager()
    gemini_client = None

    try:

        # ================================================
        # TEST 1: START MCP MANAGER
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 1: START MCP MANAGER")
        print("=" * 60)

        mcp_manager.start()

        print("\n[TEST PASSED] MCP manager started.")

        # ================================================
        # TEST 2: INITIALIZE GEMINI
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 2: INITIALIZE GEMINI")
        print("=" * 60)

        gemini_client = GeminiClient(
            mcp_manager=mcp_manager
        )

        print("\n[TEST PASSED] Gemini initialized.")

        # ================================================
        # TEST 3: GEMINI SYSTEM TOOL CALLING
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 3: GEMINI SYSTEM TOOL CALLING")
        print("=" * 60)

        prompt = (
            "Check my computer's current RAM usage. "
            "Use the available system monitoring tool "
            "to retrieve the actual value."
        )

        print("\nUSER:")
        print(prompt)

        response = gemini_client.generate_response(prompt)

        print("\nGEMINI FINAL RESPONSE:")
        print(response)

        assert isinstance(response, str)
        assert response.strip()

        assert (
            gemini_client.pending_interaction_id
            is not None
        )

        print(
            "\n[TEST PASSED] Gemini executed the system "
            "tool and generated a response."
        )

        # ================================================
        # TEST 4: COMMIT SYSTEM INTERACTION
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 4: COMMIT SYSTEM INTERACTION")
        print("=" * 60)

        gemini_client.commit_pending_interaction()

        assert (
            gemini_client.pending_interaction_id
            is None
        )

        assert (
            gemini_client.previous_interaction_id
            is not None
        )

        print("\n[TEST PASSED] System interaction committed.")

        # ================================================
        # TEST 5: GEMINI WEATHER TOOL CALLING
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 5: GEMINI WEATHER TOOL CALLING")
        print("=" * 60)

        weather_prompt = (
            "What's the current weather in Cebu City? "
            "Use the available weather tool to retrieve "
            "the actual current weather conditions."
        )

        print("\nUSER:")
        print(weather_prompt)

        weather_response = gemini_client.generate_response(
            weather_prompt
        )

        print("\nGEMINI WEATHER RESPONSE:")
        print(weather_response)

        assert isinstance(weather_response, str)
        assert weather_response.strip()

        assert (
            gemini_client.pending_interaction_id
            is not None
        )

        print(
            "\n[TEST PASSED] Gemini generated a weather response."
        )

        # ================================================
        # TEST 6: COMMIT WEATHER INTERACTION
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 6: COMMIT WEATHER INTERACTION")
        print("=" * 60)

        gemini_client.commit_pending_interaction()

        assert (
            gemini_client.pending_interaction_id
            is None
        )

        print("\n[TEST PASSED] Weather interaction committed.")

        # ================================================
        # TEST 7: GEMINI WEB SEARCH TOOL CALLING
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 7: GEMINI WEB SEARCH TOOL CALLING")
        print("=" * 60)

        search_prompt = (
            "Search the web for the latest developments "
            "in solid-state batteries. Use the web_search "
            "tool to retrieve actual search results. "
            "Summarize the most important findings and "
            "mention the sources."
        )

        print("\nUSER:")
        print(search_prompt)

        search_response = gemini_client.generate_response(
            search_prompt
        )

        print("\nGEMINI WEB SEARCH RESPONSE:")
        print(search_response)

        assert isinstance(search_response, str)
        assert search_response.strip()

        assert (
            gemini_client.pending_interaction_id
            is not None
        )

        print(
            "\n[TEST PASSED] Gemini generated a web search response."
        )

        # ================================================
        # TEST 8: COMMIT WEB SEARCH INTERACTION
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 8: COMMIT WEB SEARCH INTERACTION")
        print("=" * 60)

        gemini_client.commit_pending_interaction()

        assert (
            gemini_client.pending_interaction_id
            is None
        )

        print("\n[TEST PASSED] Web search interaction committed.")

        # ================================================
        # TEST 9: FOLLOW-UP CONVERSATION
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 9: FOLLOW-UP CONVERSATION")
        print("=" * 60)

        follow_up = (
            "How does the humidity affect how hot "
            "the weather feels?"
        )

        print("\nUSER:")
        print(follow_up)

        follow_up_response = (
            gemini_client.generate_response(follow_up)
        )

        print("\nGEMINI FOLLOW-UP RESPONSE:")
        print(follow_up_response)

        assert isinstance(follow_up_response, str)
        assert follow_up_response.strip()

        print(
            "\n[TEST PASSED] Conversation context "
            "was preserved."
        )

        gemini_client.commit_pending_interaction()

        # ================================================
        # TEST 10: GEMINI GOOGLE CALENDAR TOOL CALLING
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 10: GEMINI GOOGLE CALENDAR TOOL CALLING")
        print("=" * 60)

        calendar_prompt = (
            "Check my Google Calendar for events scheduled "
            "from November 9, 2026 through November 16, 2026. "
            "Use the get_calendar_events tool to retrieve "
            "the actual events. Use the Asia/Manila timezone. "
            "Do not create, modify, or delete any events."
        )

        print("\nUSER:")
        print(calendar_prompt)

        calendar_response = gemini_client.generate_response(
            calendar_prompt
        )

        print("\nGEMINI CALENDAR RESPONSE:")
        print(calendar_response)

        assert isinstance(calendar_response, str)
        assert calendar_response.strip()

        assert (
            gemini_client.pending_interaction_id
            is not None
        )

        print(
            "\n[TEST PASSED] Gemini generated a Calendar response."
        )

        # ================================================
        # TEST 11: COMMIT CALENDAR INTERACTION
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 11: COMMIT CALENDAR INTERACTION")
        print("=" * 60)

        gemini_client.commit_pending_interaction()

        assert (
            gemini_client.pending_interaction_id
            is None
        )

        print("\n[TEST PASSED] Calendar interaction committed.")

        # ================================================
        # TEST 12: GEMINI GOOGLE TASKS TOOL CALLING
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 12: GEMINI GOOGLE TASKS TOOL CALLING")
        print("=" * 60)

        tasks_prompt = (
            "Check my Google Tasks. First retrieve my task "
            "lists using get_task_lists, then retrieve the "
            "existing tasks in my default task list using "
            "get_tasks. Report the task names and their "
            "completion status. Do not create, complete, "
            "or delete any tasks."
        )

        print("\nUSER:")
        print(tasks_prompt)

        tasks_response = gemini_client.generate_response(
            tasks_prompt
        )

        print("\nGEMINI TASKS RESPONSE:")
        print(tasks_response)

        assert isinstance(tasks_response, str)
        assert tasks_response.strip()

        assert (
            gemini_client.pending_interaction_id
            is not None
        )

        print(
            "\n[TEST PASSED] Gemini generated a Google Tasks response."
        )

        # ================================================
        # TEST 13: COMMIT TASKS INTERACTION
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 13: COMMIT TASKS INTERACTION")
        print("=" * 60)

        gemini_client.commit_pending_interaction()

        assert (
            gemini_client.pending_interaction_id
            is None
        )

        print("\n[TEST PASSED] Tasks interaction committed.")

        # ================================================
        # ALL TESTS PASSED
        # ================================================

        print("\n" + "=" * 60)
        print("ALL GEMINI MCP TESTS PASSED")
        print("=" * 60)

    except Exception as error:

        print(
            f"\n[TEST ERROR] {type(error).__name__}: {error}"
        )

        traceback.print_exc()

        raise

    finally:

        print("\n[TEST] Closing Gemini client...")

        if gemini_client is not None:
            gemini_client.close()

        print("\n[TEST] Closing MCP manager...")

        mcp_manager.close()


if __name__ == "__main__":

    main()