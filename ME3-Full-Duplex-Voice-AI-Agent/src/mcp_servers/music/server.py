
import asyncio
import json
import sys

from mcp.server.mcpserver import MCPServer

from src.mcp_servers.music.audius_client import search_tracks
from src.mcp_servers.music.music_controller import MusicController


# ============================================================
# MCP SERVER
# ============================================================

server = MCPServer(
    name="Music MCP Server",
    instructions=(
        "Provides music search, playback, and volume control "
        "using the Audius music platform. "
        "Can search, play, pause, resume, stop, skip tracks, "
        "retrieve current track information and playback status, "
        "and control music volume."
    )
)


# ============================================================
# MUSIC CONTROLLER
# ============================================================

# One persistent controller for the lifetime of the server.
music_controller = MusicController(
    initial_volume=30,
    duck_factor=0.20
)


# ============================================================
# RESPONSE HELPER
# ============================================================

def format_response(data):
    return json.dumps(
        {
            "success": True,
            "data": data
        },
        indent=2,
        default=str
    )


# ============================================================
# MUSIC SEARCH
# ============================================================

@server.tool(
    name="search_music",
    description=(
        "Search Audius for music tracks using a song title, "
        "artist name, or keyword. Returns matching tracks "
        "and their streaming availability."
    )
)
def search_music(query: str, limit: int = 5) -> str:

    tracks = search_tracks(
        query=query,
        limit=limit
    )

    return format_response(
        {
            "query": query,
            "count": len(tracks),
            "tracks": tracks
        }
    )


# ============================================================
# MUSIC PLAYBACK
# ============================================================

@server.tool(
    name="play_music",
    description=(
        "Search Audius for music matching the user's request "
        "and play the first available streamable track."
    )
)
def play_music(query: str) -> str:

    result = music_controller.search_and_play(
        query=query
    )

    return format_response(result)


@server.tool(
    name="pause_music",
    description="Pause the currently playing music."
)
def pause_music() -> str:

    result = music_controller.pause()

    return format_response(result)


@server.tool(
    name="resume_music",
    description="Resume previously paused music."
)
def resume_music() -> str:

    result = music_controller.resume()

    return format_response(result)


@server.tool(
    name="stop_music",
    description="Stop music playback."
)
def stop_music() -> str:

    result = music_controller.stop()

    return format_response(result)


@server.tool(
    name="skip_track",
    description=(
        "Skip the currently playing track and play the next "
        "streamable track in the existing music queue."
    )
)
def skip_track() -> str:

    result = music_controller.skip_track()

    return format_response(result)


@server.tool(
    name="get_current_track",
    description=(
        "Get information about the currently selected music "
        "track, including its title, artist, duration, and ID. "
        "Returns null if no track has been selected."
    )
)
def get_current_track() -> str:

    result = music_controller.get_current_track()

    return format_response(result)


@server.tool(
    name="get_music_status",
    description=(
        "Get the current music track, playback state, "
        "volume, and queue information."
    )
)
def get_music_status() -> str:

    result = music_controller.get_status()

    return format_response(result)


# ============================================================
# MUSIC VOLUME CONTROL
# ============================================================

@server.tool(
    name="set_music_volume",
    description=(
        "Set the music playback volume to a specific percentage "
        "between 0 and 100. This changes music volume only, "
        "not the computer's system volume."
    )
)
def set_music_volume(volume: float) -> str:

    result = music_controller.set_volume(volume)

    return format_response(result)


@server.tool(
    name="adjust_music_volume",
    description=(
        "Increase or decrease music volume by a specified "
        "percentage. Use a positive number to increase and "
        "a negative number to decrease. For example, 10 "
        "increases volume by 10 percentage points."
    )
)
def adjust_music_volume(change: float) -> str:

    result = music_controller.adjust_volume(change)

    return format_response(result)


@server.tool(
    name="get_music_volume",
    description=(
        "Get the current selected music volume percentage "
        "and whether music is muted."
    )
)
def get_music_volume() -> str:

    result = music_controller.get_volume()

    return format_response(result)


@server.tool(
    name="mute_music",
    description=(
        "Mute music playback while preserving the selected "
        "music volume for later restoration."
    )
)
def mute_music() -> str:

    result = music_controller.mute()

    return format_response(result)


@server.tool(
    name="unmute_music",
    description=(
        "Unmute music playback and restore the previously "
        "selected music volume."
    )
)
def unmute_music() -> str:

    result = music_controller.unmute()

    return format_response(result)


# ============================================================
# SERVER STARTUP
# ============================================================

async def main():

    print(
        "[MUSIC MCP] Starting Music MCP Server...",
        file=sys.stderr
    )

    try:
        await server.run_stdio_async()

    finally:
        music_controller.close()


if __name__ == "__main__":
    asyncio.run(main())
