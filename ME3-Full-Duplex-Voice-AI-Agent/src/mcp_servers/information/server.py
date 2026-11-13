
import sys
import traceback

from mcp.server.mcpserver import MCPServer

from src.mcp_servers.information.weather_service import (
    get_weather,
)

from src.mcp_servers.information.web_search_service import (
    web_search as execute_web_search,
)


# ============================================================
# MCP SERVER
# ============================================================

server = MCPServer(
    name="Information MCP Server",
    instructions=(
        "Provides external information retrieval tools. "
        "It supports retrieving current weather conditions "
        "using Open-Meteo and searching the web using Tavily. "
        "Use the current weather tool when users ask about "
        "current weather conditions. "
        "Use the web search tool when users request current, "
        "recent, or online information that requires external "
        "sources. "
        "For general knowledge, explanations, and reasoning "
        "that do not require fresh online information, "
        "answer directly using Gemini instead."
    ),
)


# ============================================================
# WEATHER
# ============================================================

@server.tool(
    name="get_current_weather",
    description=(
        "Retrieve actual current weather conditions for a "
        "city or location worldwide. Use this tool when the "
        "user asks about current weather."
    )
)
def get_current_weather(location: str) -> dict:

    try:
        return get_weather(location)

    except Exception:
        print(
            "\n[INFORMATION MCP] WEATHER TOOL EXCEPTION:",
            file=sys.stderr,
        )

        traceback.print_exc(file=sys.stderr)

        raise


# ============================================================
# WEB SEARCH
# ============================================================

@server.tool(
    name="web_search",
    description=(
        "Search the internet using Tavily for current, recent, "
        "or online information. Use this tool when the user "
        "asks for the latest information, recent developments, "
        "current events, or information that requires online "
        "sources. Do not use it for ordinary general knowledge "
        "or explanations that Gemini can answer directly. "
        "Returns search result titles, URLs, content snippets, "
        "and relevance scores."
    )
)
def web_search(
    query: str,
    max_results: int = 5
) -> dict:

    return execute_web_search(
        query=query,
        max_results=max_results
    )


# ============================================================
# SERVER STARTUP
# ============================================================

async def main():

    await server.run_stdio_async()


if __name__ == "__main__":

    import asyncio

    asyncio.run(main())
