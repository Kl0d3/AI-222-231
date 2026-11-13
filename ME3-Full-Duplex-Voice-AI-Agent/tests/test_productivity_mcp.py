import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    print("=" * 60)
    print("PRODUCTIVITY MCP SERVER TEST")
    print("=" * 60)

    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_servers.productivity.server"],
    )

    async with stdio_client(
        server_parameters,
        errlog=sys.stderr,
    ) as (read_stream, write_stream):

        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:

            await session.initialize()

            print("\n[1] MCP CONNECTION SUCCESSFUL")

            # ------------------------------------------------
            # TEST 1: DISCOVER TOOLS
            # ------------------------------------------------

            print("\n[2] DISCOVERING PRODUCTIVITY TOOLS")

            tools_response = await session.list_tools()

            tools = tools_response.tools

            print(f"\nTools discovered: {len(tools)}")

            for tool in tools:
                print(f"- {tool.name}")

            expected_tools = {
                "create_calendar_event",
                "get_calendar_events",
                "delete_calendar_event",
                "create_task",
                "get_tasks",
                "complete_task",
                "delete_task",
                "get_task_lists",
            }

            discovered_tools = {
                tool.name for tool in tools
            }

            missing_tools = expected_tools - discovered_tools
            unexpected_tools = discovered_tools - expected_tools

            if missing_tools:
                raise RuntimeError(
                    f"Missing tools: {missing_tools}"
                )

            if unexpected_tools:
                print(
                    f"\nUnexpected tools: {unexpected_tools}"
                )

            print("\nAll eight expected tools discovered.")

            # ------------------------------------------------
            # TEST 2: GET TASK LISTS
            # ------------------------------------------------

            print("\n" + "=" * 60)
            print("TEST 2: GET GOOGLE TASK LISTS")
            print("=" * 60)

            task_lists_result = await session.call_tool(
                "get_task_lists",
                arguments={},
            )

            print("\nTask lists result:")

            for content in task_lists_result.content:
                if hasattr(content, "text"):
                    print(content.text)

            if task_lists_result.is_error:
                raise RuntimeError(
                    "get_task_lists MCP tool failed."
                )

            # ------------------------------------------------
            # TEST 3: GET CALENDAR EVENTS
            # ------------------------------------------------

            print("\n" + "=" * 60)
            print("TEST 3: GET GOOGLE CALENDAR EVENTS")
            print("=" * 60)

            calendar_result = await session.call_tool(
                "get_calendar_events",
                arguments={
                    "start_datetime": "2026-11-09T00:00:00",
                    "end_datetime": "2026-11-16T00:00:00",
                    "max_results": 20,
                    "timezone": "Asia/Manila",
                },
            )

            print("\nCalendar result:")

            for content in calendar_result.content:
                if hasattr(content, "text"):
                    print(content.text)

            if calendar_result.is_error:
                raise RuntimeError(
                    "get_calendar_events MCP tool failed."
                )

            print("\n" + "=" * 60)
            print("ALL PRODUCTIVITY MCP TESTS PASSED")
            print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())