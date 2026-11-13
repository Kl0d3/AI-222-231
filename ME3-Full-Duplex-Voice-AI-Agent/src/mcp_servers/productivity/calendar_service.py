from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.mcp_servers.productivity.google_auth import get_google_credentials


DEFAULT_TIMEZONE = "Asia/Manila"


def get_calendar_service():
    """Build and return an authenticated Google Calendar service."""

    credentials = get_google_credentials()

    return build(
        "calendar",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )


def create_calendar_event(
    title: str,
    start_datetime: str,
    end_datetime: str,
    description: str = "",
    location: str = "",
    timezone: str = DEFAULT_TIMEZONE,
) -> dict:
    """
    Create a Google Calendar event.

    Datetime format: YYYY-MM-DDTHH:MM:SS
    Example: 2026-11-10T14:00:00
    """

    try:
        if not title or not title.strip():
            return {
                "success": False,
                "error": "Event title cannot be empty.",
            }

        tz = ZoneInfo(timezone)

        start = datetime.fromisoformat(start_datetime).replace(
            tzinfo=tz
        )

        end = datetime.fromisoformat(end_datetime).replace(
            tzinfo=tz
        )

        if end <= start:
            return {
                "success": False,
                "error": "Event end time must be after its start time.",
            }

        event_body = {
            "summary": title.strip(),
            "description": description,
            "location": location,
            "start": {
                "dateTime": start.isoformat(),
                "timeZone": timezone,
            },
            "end": {
                "dateTime": end.isoformat(),
                "timeZone": timezone,
            },
        }

        service = get_calendar_service()

        event = (
            service.events()
            .insert(
                calendarId="primary",
                body=event_body,
            )
            .execute()
        )

        return {
            "success": True,
            "message": "Calendar event created successfully.",
            "event": {
                "id": event.get("id"),
                "title": event.get("summary"),
                "start": event.get("start", {}).get("dateTime"),
                "end": event.get("end", {}).get("dateTime"),
                "timezone": timezone,
                "html_link": event.get("htmlLink"),
            },
        }

    except (ValueError, KeyError) as error:
        return {
            "success": False,
            "error": f"Invalid event input: {error}",
        }

    except HttpError as error:
        return {
            "success": False,
            "error": f"Google Calendar API error: {error}",
        }

    except Exception as error:
        return {
            "success": False,
            "error": f"Unexpected error: {error}",
        }


def get_calendar_events(
    start_datetime: str,
    end_datetime: str,
    max_results: int = 20,
    timezone: str = DEFAULT_TIMEZONE,
) -> dict:
    """
    Retrieve calendar events within a specified time range.

    Datetime format: YYYY-MM-DDTHH:MM:SS
    """

    try:
        tz = ZoneInfo(timezone)

        start = datetime.fromisoformat(start_datetime).replace(
            tzinfo=tz
        )

        end = datetime.fromisoformat(end_datetime).replace(
            tzinfo=tz
        )

        if end <= start:
            return {
                "success": False,
                "error": "End datetime must be after start datetime.",
            }

        if not 1 <= max_results <= 250:
            return {
                "success": False,
                "error": "max_results must be between 1 and 250.",
            }

        service = get_calendar_service()

        result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=start.isoformat(),
                timeMax=end.isoformat(),
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        events = []

        for event in result.get("items", []):
            event_start = event.get("start", {})
            event_end = event.get("end", {})

            events.append({
                "id": event.get("id"),
                "title": event.get("summary", "(No title)"),
                "description": event.get("description", ""),
                "location": event.get("location", ""),
                "start": (
                    event_start.get("dateTime")
                    or event_start.get("date")
                ),
                "end": (
                    event_end.get("dateTime")
                    or event_end.get("date")
                ),
                "html_link": event.get("htmlLink"),
                "status": event.get("status"),
            })

        return {
            "success": True,
            "event_count": len(events),
            "timezone": timezone,
            "events": events,
        }

    except (ValueError, KeyError) as error:
        return {
            "success": False,
            "error": f"Invalid date/time input: {error}",
        }

    except HttpError as error:
        return {
            "success": False,
            "error": f"Google Calendar API error: {error}",
        }

    except Exception as error:
        return {
            "success": False,
            "error": f"Unexpected error: {error}",
        }


def delete_calendar_event(event_id: str) -> dict:
    """Delete a Google Calendar event using its event ID."""

    try:
        if not event_id or not event_id.strip():
            return {
                "success": False,
                "error": "Event ID cannot be empty.",
            }

        service = get_calendar_service()

        service.events().delete(
            calendarId="primary",
            eventId=event_id.strip(),
        ).execute()

        return {
            "success": True,
            "message": "Calendar event deleted successfully.",
            "event_id": event_id,
        }

    except HttpError as error:
        return {
            "success": False,
            "error": f"Google Calendar API error: {error}",
        }

    except Exception as error:
        return {
            "success": False,
            "error": f"Unexpected error: {error}",
        }