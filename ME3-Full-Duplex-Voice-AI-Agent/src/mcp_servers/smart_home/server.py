
from mcp.server.mcpserver import MCPServer

from src.mcp_servers.smart_home.smart_home_service import (
    smart_home_service,
)


server = MCPServer(
    name="Smart Home MCP Server",
    instructions=(
        "Provides simulated smart home controls for lights and "
        "thermostats. This server does not control physical devices. "
        "Available devices are three simulated lights and two "
        "simulated thermostats. Use the device names returned by "
        "list_smart_home_devices. Temperatures are in Celsius and "
        "brightness is expressed as a percentage."
    ),
)


@server.tool()
def list_smart_home_devices() -> dict:
    """List all available simulated smart lights and thermostats."""
    return smart_home_service.list_smart_home_devices()


@server.tool()
def get_light_status(device_name: str) -> dict:
    """Get the on/off state and brightness of a simulated light."""
    return smart_home_service.get_light_status(device_name)


@server.tool()
def control_light(
    device_name: str,
    action: str,
    brightness: int | None = None,
) -> dict:
    """
    Control a simulated light.

    Actions:
    - turn_on: Turn the light on.
    - turn_off: Turn the light off.
    - set_brightness: Set brightness from 0 to 100 percent.
      A brightness of 0 turns the light off; any value above 0
      turns it on.
    """
    return smart_home_service.control_light(
        device_name=device_name,
        action=action,
        brightness=brightness,
    )


@server.tool()
def get_thermostat_status(device_name: str) -> dict:
    """Get the current and target temperatures of a simulated thermostat."""
    return smart_home_service.get_thermostat_status(device_name)


@server.tool()
def set_thermostat_temperature(
    device_name: str,
    temperature: float,
) -> dict:
    """
    Set a simulated thermostat's target temperature.

    Temperature must be between 10 and 35 degrees Celsius.
    This changes only the simulated target temperature.
    """
    return smart_home_service.set_thermostat_temperature(
        device_name=device_name,
        temperature=temperature,
    )


async def main():
    await server.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
