
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys


async def main():
    print("=" * 55)
    print("SMART HOME MCP SERVER TEST")
    print("=" * 55)

    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_servers.smart_home.server"],
    )

    async with stdio_client(
        server_parameters,
        errlog=sys.stderr,
    ) as (read_stream, write_stream):

        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.list_tools()

            expected_tools = {
                "list_smart_home_devices",
                "get_light_status",
                "control_light",
                "get_thermostat_status",
                "set_thermostat_temperature",
            }

            discovered_tools = {
                tool.name for tool in result.tools
            }

            print(f"\nTools discovered: {len(discovered_tools)}")

            for tool_name in sorted(discovered_tools):
                if tool_name in expected_tools:
                    print(f"PASS: {tool_name}")
                else:
                    print(f"UNEXPECTED: {tool_name}")

            missing_tools = expected_tools - discovered_tools
            unexpected_tools = discovered_tools - expected_tools

            if missing_tools or unexpected_tools:
                raise AssertionError(
                    f"Missing tools: {missing_tools}; "
                    f"Unexpected tools: {unexpected_tools}"
                )

            print("\nALL 5 SMART HOME MCP TOOLS REGISTERED SUCCESSFULLY")

    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
