
import os

import requests
from dotenv import load_dotenv


# Load environment variables from the project-root .env file.
load_dotenv()

AUDIUS_API_KEY = os.getenv("AUDIUS_API_KEY")

BASE_URL = "https://api.audius.co/v1"


def search_tracks(query: str, limit: int = 5):
    """
    Search public Audius tracks.

    Args:
        query: Track title, artist, or keyword.
        limit: Maximum number of results.

    Returns:
        A list of matching track dictionaries.
    """

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    params = {
        "query": query,
        "limit": limit,
    }

    # API key is optional for public read-only searches,
    # but we use yours to access the higher rate limit.
    if AUDIUS_API_KEY:
        params["api_key"] = AUDIUS_API_KEY

    url = f"{BASE_URL}/tracks/search"

    response = requests.get(
        url,
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    result = response.json()

    return result.get("data", [])


def main():
    print("=" * 50)
    print("AUDIUS TRACK SEARCH TEST")
    print("=" * 50)

    if not AUDIUS_API_KEY:
        print("API key not found. Continuing without it.")
    else:
        print("Audius API key loaded successfully.")

    query = "lofi"

    print(f"\nSearching for: {query}")

    try:
        tracks = search_tracks(query, limit=5)

        if not tracks:
            print("\nNo tracks found.")
            return

        print(f"\nFound {len(tracks)} tracks:\n")

        for index, track in enumerate(tracks, start=1):
            title = track.get("title", "Unknown title")

            user = track.get("user") or {}
            artist = user.get("name", "Unknown artist")

            track_id = track.get("id", "Unknown ID")
            duration = track.get("duration", 0)

            print(f"{index}. {title}")
            print(f"   Artist: {artist}")
            print(f"   Track ID: {track_id}")
            print(f"   Duration: {duration} seconds")
            print()

        print("Audius search test completed.")

    except requests.exceptions.RequestException as error:
        print("\nAudius API request failed.")
        print(f"Error: {error}")

    except (ValueError, KeyError) as error:
        print("\nCould not process the Audius response.")
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
