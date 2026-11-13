
import os

from dotenv import load_dotenv
from tavily import TavilyClient


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

DEFAULT_MAX_RESULTS = 5
MAX_ALLOWED_RESULTS = 10

DEFAULT_SEARCH_DEPTH = "basic"


# ============================================================
# TAVILY CLIENT
# ============================================================

if not TAVILY_API_KEY:
    raise RuntimeError(
        "TAVILY_API_KEY was not found in the environment."
    )

tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)


# ============================================================
# WEB SEARCH
# ============================================================

def web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> dict:
    """
    Search the web using Tavily.

    Args:
        query:
            The user's search query.

        max_results:
            Maximum number of search results to return.

    Returns:
        A dictionary containing the query and search results.
    """

    # --------------------------------------------------------
    # VALIDATE QUERY
    # --------------------------------------------------------

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "Search query must be a non-empty string."
        )

    query = query.strip()

    # --------------------------------------------------------
    # VALIDATE RESULT COUNT
    # --------------------------------------------------------

    if isinstance(max_results, bool) or not isinstance(
        max_results, int
    ):
        raise ValueError(
            "max_results must be an integer."
        )

    if not 1 <= max_results <= MAX_ALLOWED_RESULTS:
        raise ValueError(
            f"max_results must be between 1 and "
            f"{MAX_ALLOWED_RESULTS}."
        )

    # --------------------------------------------------------
    # EXECUTE TAVILY SEARCH
    # --------------------------------------------------------

    try:

        response = tavily_client.search(
            query=query,
            search_depth=DEFAULT_SEARCH_DEPTH,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )

    except Exception as error:

        raise RuntimeError(
            f"Tavily web search failed: {error}"
        ) from error

    # --------------------------------------------------------
    # NORMALIZE SEARCH RESULTS
    # --------------------------------------------------------

    results = []

    for item in response.get("results", []):

        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
            }
        )

    # --------------------------------------------------------
    # RETURN STRUCTURED RESPONSE
    # --------------------------------------------------------

    return {
        "success": True,
        "query": query,
        "result_count": len(results),
        "search_depth": DEFAULT_SEARCH_DEPTH,
        "results": results,
        "source": "Tavily",
    }
