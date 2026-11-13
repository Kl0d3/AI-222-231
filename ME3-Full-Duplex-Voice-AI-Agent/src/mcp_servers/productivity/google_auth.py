import os
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


# Resolve the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Google OAuth credential files.
CREDENTIALS_DIR = PROJECT_ROOT / "credentials"
CLIENT_SECRET_FILE = CREDENTIALS_DIR / "google_oauth_client.json"
TOKEN_FILE = CREDENTIALS_DIR / "google_token.json"

# Permissions required by Productivity MCP.
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/tasks",
]


def get_google_credentials() -> Credentials:
    """
    Authenticate with Google and return valid OAuth credentials.

    The first execution opens a browser for Google authorization.
    Subsequent executions reuse the saved token and refresh it when needed.
    """

    load_dotenv(PROJECT_ROOT / ".env")

    if not CLIENT_SECRET_FILE.exists():
        raise FileNotFoundError(
            f"Google OAuth client credentials not found: "
            f"{CLIENT_SECRET_FILE}"
        )

    credentials = None

    # Reuse previously saved credentials.
    if TOKEN_FILE.exists():
        credentials = Credentials.from_authorized_user_file(
            str(TOKEN_FILE),
            SCOPES,
        )

    # Refresh expired credentials when possible.
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())

    # Start browser-based OAuth if no valid credentials exist.
    if not credentials or not credentials.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CLIENT_SECRET_FILE),
            SCOPES,
        )

        credentials = flow.run_local_server(
            port=0,
            access_type="offline",
            prompt="consent",
        )

    # Ensure the credentials directory exists.
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the token for future use.
    TOKEN_FILE.write_text(
        credentials.to_json(),
        encoding="utf-8",
    )

    return credentials