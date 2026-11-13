
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# CONFIGURATION
# ============================================================

PYTHON_EXECUTABLE = sys.executable

SERVER_ARGUMENTS = [
    "-m",
    "src.mcp_servers.music.server"
]


# ============================================================
# MCP CLIENT
# ============================================================

async def main():

    server_parameters = StdioServerParameters(
        command=PYTHON_EXECUTABLE,
        args=SERVER_ARGUMENTS,
        cwd=None
    )

    print("\n[MCP CLIENT] Starting Music MCP Server...")

    async with stdio_client(
        server_parameters,
        errlog=sys.stderr
    ) as (
        read_stream,
        write_stream
    ):

        async with ClientSession(
            read_stream,
            write_stream
        ) as session:

            await session.initialize()

            print("[MCP CLIENT] Connected successfully.")

            # ------------------------------------------------
            # LIST AVAILABLE TOOLS
            # ------------------------------------------------

            tools_result = await session.list_tools()

            print("\nAvailable MCP tools:")

            for tool in tools_result.tools:
                print(f"- {tool.name}")

            # ------------------------------------------------
            # INTERACTIVE COMMANDS
            # ------------------------------------------------

            print("\n========================================")
            print(" MUSIC MCP INTERACTIVE TEST CLIENT")
            print("========================================")

            print("Playback:")
            print("  play <query>")
            print("  pause")
            print("  resume")
            print("  stop")
            print("  skip")
            print("  current")
            print("  status")

            print("\nVolume:")
            print("  volume <0-100>")
            print("  increase <amount>")
            print("  decrease <amount>")
            print("  volume_status")
            print("  mute")
            print("  unmute")

            print("\nOther:")
            print("  exit")

            # ------------------------------------------------
            # INTERACTIVE TOOL LOOP
            # ------------------------------------------------

            while True:

                command = await asyncio.to_thread(
                    input,
                    "\nMusic MCP > "
                )

                command = command.strip()

                if not command:
                    continue

                if command.lower() == "exit":

                    print("\n[MCP CLIENT] Closing session...")
                    break

                parts = command.split(maxsplit=1)

                action = parts[0].lower()

                tool_name = None
                arguments = {}

                # --------------------------------------------
                # PLAYBACK COMMANDS
                # --------------------------------------------

                if action == "play":

                    if len(parts) < 2:
                        print("Please provide a search query.")
                        continue

                    tool_name = "play_music"

                    arguments = {
                        "query": parts[1]
                    }

                elif action == "pause":

                    tool_name = "pause_music"

                elif action == "resume":

                    tool_name = "resume_music"

                elif action == "stop":

                    tool_name = "stop_music"

                elif action == "skip":

                    tool_name = "skip_track"

                elif action == "current":

                    tool_name = "get_current_track"

                elif action == "status":

                    tool_name = "get_music_status"

                # --------------------------------------------
                # VOLUME COMMANDS
                # --------------------------------------------

                elif action == "volume":

                    if len(parts) < 2:
                        print("Usage: volume <0-100>")
                        continue

                    try:
                        volume = float(parts[1])
                    except ValueError:
                        print("Volume must be a number.")
                        continue

                    tool_name = "set_music_volume"

                    arguments = {
                        "volume": volume
                    }

                elif action == "increase":

                    if len(parts) < 2:
                        print("Usage: increase <amount>")
                        continue

                    try:
                        amount = float(parts[1])
                    except ValueError:
                        print("Amount must be a number.")
                        continue

                    tool_name = "adjust_music_volume"

                    arguments = {
                        "change": amount
                    }

                elif action == "decrease":

                    if len(parts) < 2:
                        print("Usage: decrease <amount>")
                        continue

                    try:
                        amount = float(parts[1])
                    except ValueError:
                        print("Amount must be a number.")
                        continue

                    tool_name = "adjust_music_volume"

                    arguments = {
                        "change": -amount
                    }

                elif action == "volume_status":

                    tool_name = "get_music_volume"

                elif action == "mute":

                    tool_name = "mute_music"

                elif action == "unmute":

                    tool_name = "unmute_music"

                else:

                    print("Unknown command.")
                    continue

                # --------------------------------------------
                # CALL MCP TOOL
                # --------------------------------------------

                print(f"\n[MCP CLIENT] Calling {tool_name}...")

                try:

                    result = await session.call_tool(
                        tool_name,
                        arguments=arguments
                    )

                    print("\n[MCP RESULT]")

                    for content in result.content:

                        if hasattr(content, "text"):
                            print(content.text)
                        else:
                            print(content)

                    if result.is_error:
                        print("\n[MCP CLIENT] Tool returned an error.")

                except Exception as error:

                    print(
                        f"\n[MCP CLIENT] Tool call failed: {error}"
                    )

    print("\n[MCP CLIENT] Disconnected.")


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\n[MCP CLIENT] Interrupted by user.")
