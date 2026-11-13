from datetime import datetime, timezone

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.mcp_servers.productivity.google_auth import get_google_credentials


def get_tasks_service():
    """Build and return an authenticated Google Tasks service."""

    credentials = get_google_credentials()

    return build(
        "tasks",
        "v1",
        credentials=credentials,
        cache_discovery=False,
    )


def get_task_lists() -> dict:
    """Retrieve the task lists available in the user's Google account."""

    try:
        service = get_tasks_service()

        result = (
            service.tasklists()
            .list(maxResults=100)
            .execute()
        )

        task_lists = []

        for task_list in result.get("items", []):
            task_lists.append({
                "id": task_list.get("id"),
                "title": task_list.get("title"),
                "updated": task_list.get("updated"),
            })

        return {
            "success": True,
            "task_list_count": len(task_lists),
            "task_lists": task_lists,
        }

    except HttpError as error:
        return {
            "success": False,
            "error": f"Google Tasks API error: {error}",
        }

    except Exception as error:
        return {
            "success": False,
            "error": f"Unexpected error: {error}",
        }


def create_task(
    title: str,
    notes: str = "",
    due_date: str = None,
    task_list_id: str = "@default",
) -> dict:
    """
    Create a Google Task.

    due_date format: YYYY-MM-DD
    Example: 2026-11-12

    task_list_id defaults to @default (My Tasks).
    """

    try:
        if not title or not title.strip():
            return {
                "success": False,
                "error": "Task title cannot be empty.",
            }

        task_body = {
            "title": title.strip(),
        }

        if notes:
            task_body["notes"] = notes

        if due_date:
            due = datetime.strptime(
                due_date,
                "%Y-%m-%d",
            ).replace(tzinfo=timezone.utc)

            task_body["due"] = (
                due.strftime("%Y-%m-%dT00:00:00.000Z")
            )

        service = get_tasks_service()

        task = (
            service.tasks()
            .insert(
                tasklist=task_list_id,
                body=task_body,
            )
            .execute()
        )

        return {
            "success": True,
            "message": "Task created successfully.",
            "task": {
                "id": task.get("id"),
                "title": task.get("title"),
                "notes": task.get("notes", ""),
                "due": task.get("due"),
                "status": task.get("status"),
                "task_list_id": task_list_id,
            },
        }

    except ValueError as error:
        return {
            "success": False,
            "error": f"Invalid due date: {error}",
        }

    except HttpError as error:
        return {
            "success": False,
            "error": f"Google Tasks API error: {error}",
        }

    except Exception as error:
        return {
            "success": False,
            "error": f"Unexpected error: {error}",
        }


def get_tasks(
    task_list_id: str = "@default",
    show_completed: bool = False,
    max_results: int = 100,
) -> dict:
    """Retrieve tasks from a specified Google Tasks list."""

    try:
        if not 1 <= max_results <= 100:
            return {
                "success": False,
                "error": "max_results must be between 1 and 100.",
            }

        service = get_tasks_service()

        result = (
            service.tasks()
            .list(
                tasklist=task_list_id,
                showCompleted=show_completed,
                showHidden=False,
                maxResults=max_results,
            )
            .execute()
        )

        tasks = []

        for task in result.get("items", []):
            tasks.append({
                "id": task.get("id"),
                "title": task.get("title"),
                "notes": task.get("notes", ""),
                "due": task.get("due"),
                "status": task.get("status"),
                "completed": task.get("completed"),
                "task_list_id": task_list_id,
            })

        return {
            "success": True,
            "task_count": len(tasks),
            "task_list_id": task_list_id,
            "tasks": tasks,
        }

    except HttpError as error:
        return {
            "success": False,
            "error": f"Google Tasks API error: {error}",
        }

    except Exception as error:
        return {
            "success": False,
            "error": f"Unexpected error: {error}",
        }


def complete_task(
    task_id: str,
    task_list_id: str = "@default",
) -> dict:
    """Mark an existing Google Task as completed."""

    try:
        if not task_id or not task_id.strip():
            return {
                "success": False,
                "error": "Task ID cannot be empty.",
            }

        service = get_tasks_service()

        task = (
            service.tasks()
            .get(
                tasklist=task_list_id,
                task=task_id.strip(),
            )
            .execute()
        )

        task["status"] = "completed"

        updated_task = (
            service.tasks()
            .update(
                tasklist=task_list_id,
                task=task_id.strip(),
                body=task,
            )
            .execute()
        )

        return {
            "success": True,
            "message": "Task marked as completed.",
            "task": {
                "id": updated_task.get("id"),
                "title": updated_task.get("title"),
                "status": updated_task.get("status"),
                "completed": updated_task.get("completed"),
            },
        }

    except HttpError as error:
        return {
            "success": False,
            "error": f"Google Tasks API error: {error}",
        }

    except Exception as error:
        return {
            "success": False,
            "error": f"Unexpected error: {error}",
        }


def delete_task(
    task_id: str,
    task_list_id: str = "@default",
) -> dict:
    """Delete an existing Google Task."""

    try:
        if not task_id or not task_id.strip():
            return {
                "success": False,
                "error": "Task ID cannot be empty.",
            }

        service = get_tasks_service()

        service.tasks().delete(
            tasklist=task_list_id,
            task=task_id.strip(),
        ).execute()

        return {
            "success": True,
            "message": "Task deleted successfully.",
            "task_id": task_id,
            "task_list_id": task_list_id,
        }

    except HttpError as error:
        return {
            "success": False,
            "error": f"Google Tasks API error: {error}",
        }

    except Exception as error:
        return {
            "success": False,
            "error": f"Unexpected error: {error}",
        }