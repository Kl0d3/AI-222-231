
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PYTHON_EXECUTABLE = sys.executable

SERVER_PARAMETERS = StdioServerParameters(
    command=PYTHON_EXECUTABLE,
    args=["-m", "src.mcp_servers.system.server"],
)


async def call_tool(session, tool_name):
    print(f"\n[MCP CLIENT] Calling {tool_name}...")

    try:
        result = await session.call_tool(
            tool_name,
            arguments={},
        )

        print("\n[MCP RESULT]")

        for item in result.content:
            if hasattr(item, "text"):
                print(item.text)

        if result.is_error:
            print("\n[MCP ERROR] Tool execution failed.")

    except Exception as error:
        print(f"\n[MCP ERROR] {error}")


async def main():
    print("\n[MCP CLIENT] Starting System MCP Server...")

    async with stdio_client(
        SERVER_PARAMETERS,
        errlog=sys.stderr,
    ) as (read_stream, write_stream):

        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:

            await session.initialize()

            print("[MCP CLIENT] Connected successfully.")

            tools_result = await session.list_tools()

            print("\nAvailable MCP tools:")

            for tool in tools_result.tools:
                print(f"- {tool.name}")

            print("\n========================================")
            print(" SYSTEM MCP INTERACTIVE TEST CLIENT")
            print("========================================")
            print("Available commands:")
            print("  ram       - Check RAM usage")
            print("  cpu       - Check CPU usage")
            print("  internet  - Run internet speed test")
            print("  uptime    - Check system uptime")
            print("  all       - Run all four tools")
            print("  quit      - Exit client")

            command_mapping = {
                "ram": "ram_usage",
                "cpu": "cpu_usage",
                "internet": "internet_speed",
                "uptime": "system_uptime",
            }

            while True:
                command = input("\nSystem MCP > ").strip().lower()

                if command in ("quit", "exit"):
                    print("\n[MCP CLIENT] Shutting down...")
                    break

                if command == "all":
                    for tool_name in command_mapping.values():
                        await call_tool(session, tool_name)
                    continue

                tool_name = command_mapping.get(command)

                if tool_name is None:
                    print("[MCP CLIENT] Unknown command.")
                    continue

                await call_tool(session, tool_name)

    print("[MCP CLIENT] Disconnected.")


if __name__ == "__main__":
    asyncio.run(main())
