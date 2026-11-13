
import os

import requests
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

AUDIUS_API_KEY = os.getenv("AUDIUS_API_KEY")

BASE_URL = "https://api.audius.co/v1"

REQUEST_TIMEOUT = 20
STREAM_TIMEOUT = 30
STREAM_CHUNK_SIZE = 64 * 1024


# ============================================================
# API HELPERS
# ============================================================

def _get_request_params():
    """
    Build common Audius API request parameters.
    """

    params = {}

    if AUDIUS_API_KEY:
        params["api_key"] = AUDIUS_API_KEY

    return params


def _make_request(endpoint, params=None, timeout=REQUEST_TIMEOUT):
    """
    Send a regular GET request to the Audius API.

    Returns:
        requests.Response
    """

    url = f"{BASE_URL}{endpoint}"

    request_params = _get_request_params()

    if params:
        request_params.update(params)

    response = requests.get(
        url,
        params=request_params,
        timeout=timeout,
        allow_redirects=True
    )

    response.raise_for_status()

    return response


# ============================================================
# TRACK SEARCH
# ============================================================

def search_tracks(query: str, limit: int = 5):
    """
    Search public Audius tracks.

    Returns:
        A list of normalized track dictionaries.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not isinstance(limit, int) or limit < 1:
        raise ValueError("Limit must be a positive integer.")

    response = _make_request(
        "/tracks/search",
        params={
            "query": query.strip(),
            "limit": limit
        }
    )

    try:
        result = response.json()
    finally:
        response.close()

    tracks = result.get("data", [])

    return [
        normalize_track(track)
        for track in tracks
    ]


# ============================================================
# TRACK INFORMATION
# ============================================================

def get_track(track_id: str):
    """
    Retrieve metadata for a specific Audius track.
    """

    if not track_id:
        raise ValueError("Track ID cannot be empty.")

    response = _make_request(
        f"/tracks/{track_id}"
    )

    try:
        result = response.json()
    finally:
        response.close()

    track = result.get("data")

    if not track:
        raise ValueError("Track was not found.")

    return normalize_track(track)


def normalize_track(track):
    """
    Convert an Audius track response into a
    consistent application-level dictionary.
    """

    user = track.get("user") or {}

    return {
        "id": track.get("id"),
        "title": track.get("title", "Unknown title"),
        "artist": user.get("name", "Unknown artist"),
        "duration": track.get("duration", 0),
        "genre": track.get("genre"),
        "is_streamable": track.get("is_streamable", False),
        "is_downloadable": track.get("is_downloadable", False)
    }


# ============================================================
# TRACK STREAM — EXISTING COMPLETE RESPONSE
# ============================================================

def get_track_stream(track_id: str):
    """
    Retrieve the playable MP3 stream response.

    Returns:
        A requests.Response object.

    The caller is responsible for closing the response.

    Note:
        This retains the existing non-progressive behavior.
    """

    if not track_id:
        raise ValueError("Track ID cannot be empty.")

    response = _make_request(
        f"/tracks/{track_id}/stream",
        timeout=STREAM_TIMEOUT
    )

    content_type = response.headers.get(
        "Content-Type",
        ""
    )

    if not content_type.lower().startswith("audio/"):
        response.close()

        raise ValueError(
            f"Unexpected stream content type: {content_type}"
        )

    return response


# ============================================================
# TRACK STREAM — PROGRESSIVE CHUNKS
# ============================================================

def get_track_stream_chunks(
    track_id: str,
    chunk_size: int = STREAM_CHUNK_SIZE
):
    """
    Download an Audius MP3 stream progressively.

    Yields:
        bytes: MP3 data chunks as they arrive.

    The HTTP response is closed automatically when:
        - The stream finishes.
        - An exception occurs.
        - The generator is explicitly closed.

    Args:
        track_id: Audius track ID.
        chunk_size: Maximum size of each downloaded chunk.
    """

    if not track_id:
        raise ValueError("Track ID cannot be empty.")

    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("Chunk size must be a positive integer.")

    url = f"{BASE_URL}/tracks/{track_id}/stream"

    response = None

    try:

        response = requests.get(
            url,
            params=_get_request_params(),
            timeout=STREAM_TIMEOUT,
            allow_redirects=True,
            stream=True
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            ""
        )

        if not content_type.lower().startswith("audio/"):
            raise ValueError(
                f"Unexpected stream content type: {content_type}"
            )

        for chunk in response.iter_content(
            chunk_size=chunk_size
        ):

            if chunk:
                yield chunk

    finally:

        if response is not None:
            response.close()


# ============================================================
# STREAM AVAILABILITY
# ============================================================

def is_track_streamable(track_id: str):
    """
    Check whether a track supports streaming.
    """

    track = get_track(track_id)

    return track["is_streamable"]


# ============================================================
# BASIC TEST
# ============================================================

def main():

    print("=" * 55)
    print("AUDIUS CLIENT TEST")
    print("=" * 55)

    if AUDIUS_API_KEY:
        print("Audius API key loaded successfully.")
    else:
        print("No API key found. Using public access.")

    try:

        print("\n1. Searching for lofi tracks...")

        tracks = search_tracks("lofi", limit=5)

        for index, track in enumerate(tracks, start=1):

            print(f"\n{index}. {track['title']}")
            print(f"   Artist: {track['artist']}")
            print(f"   ID: {track['id']}")
            print(f"   Duration: {track['duration']} seconds")
            print(f"   Streamable: {track['is_streamable']}")

        if not tracks:
            print("\nNo tracks found.")
            return

        print("\n2. Checking the first track...")

        track_id = tracks[0]["id"]

        track = get_track(track_id)

        print(f"Title: {track['title']}")
        print(f"Streamable: {track['is_streamable']}")

        print("\nAudius client test completed.")

    except requests.RequestException as error:

        print("\nAudius API request failed.")
        print(f"Error type: {type(error).__name__}")

    except Exception as error:

        print("\nAudius client test failed.")
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
