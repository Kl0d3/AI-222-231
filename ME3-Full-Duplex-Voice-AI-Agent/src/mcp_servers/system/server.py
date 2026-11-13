
from mcp.server.mcpserver import MCPServer

from src.mcp_servers.system.system_monitor import (
    get_ram_usage,
    get_cpu_usage,
    get_internet_speed,
    get_system_uptime,
)


server = MCPServer(
    name="System MCP Server",
    instructions=(
        "Provides system monitoring tools for the machine "
        "running this MCP server. It can retrieve RAM usage, "
        "CPU usage, internet speed, and system uptime. "
        "All measurements describe the local host machine."
    ),
)


@server.tool()
def ram_usage() -> dict:
    """Get total, used, and available RAM and memory usage percentage."""
    return get_ram_usage()


@server.tool()
def cpu_usage() -> dict:
    """Get CPU utilization percentage and physical and logical core counts."""
    return get_cpu_usage()


@server.tool()
def internet_speed() -> dict:
    """Measure local host internet download speed, upload speed, and ping."""
    return get_internet_speed()


@server.tool()
def system_uptime() -> dict:
    """Get how long the local host has been running since its last boot."""
    return get_system_uptime()


async def main():
    await server.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
