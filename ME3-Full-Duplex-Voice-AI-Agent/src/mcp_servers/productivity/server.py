import asyncio
import sys
import traceback

from mcp.server.mcpserver import MCPServer

from src.mcp_servers.productivity.calendar_service import (
    create_calendar_event as calendar_create_event,
    get_calendar_events as calendar_get_events,
    delete_calendar_event as calendar_delete_event,
)

from src.mcp_servers.productivity.tasks_service import (
    create_task as tasks_create_task,
    get_tasks as tasks_get_tasks,
    complete_task as tasks_complete_task,
    delete_task as tasks_delete_task,
    get_task_lists as tasks_get_task_lists,
)


server = MCPServer(
    name="Productivity MCP Server",
    instructions=(
        "Provides productivity tools using Google Calendar and Google Tasks. "
        "Use Google Calendar for appointments, meetings, and scheduled events. "
        "Use Google Tasks for to-do items, deadlines, and task completion. "
        "Use the appropriate tool for the user's request. "
        "Before deleting any calendar event or task, obtain explicit "
        "confirmation from the user. Never claim an operation succeeded "
        "unless the tool returns success=true."
    ),
)


# ============================================================
# GOOGLE CALENDAR TOOLS
# ============================================================

@server.tool()
def create_calendar_event(
    title: str,
    start_datetime: str,
    end_datetime: str,
    description: str = "",
    location: str = "",
    timezone: str = "Asia/Manila",
) -> dict:
    """Create a Google Calendar event."""

    return calendar_create_event(
        title=title,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        description=description,
        location=location,
        timezone=timezone,
    )


@server.tool()
def get_calendar_events(
    start_datetime: str,
    end_datetime: str,
    max_results: int = 20,
    timezone: str = "Asia/Manila",
) -> dict:
    """Retrieve Google Calendar events within a date/time range."""

    return calendar_get_events(
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        max_results=max_results,
        timezone=timezone,
    )


@server.tool()
def delete_calendar_event(event_id: str) -> dict:
    """Delete a calendar event. Requires explicit user confirmation."""

    return calendar_delete_event(event_id=event_id)


# ============================================================
# GOOGLE TASKS TOOLS
# ============================================================

@server.tool()
def create_task(
    title: str,
    notes: str = "",
    due_date: str = None,
    task_list_id: str = "@default",
) -> dict:
    """Create a Google Task with an optional due date."""

    return tasks_create_task(
        title=title,
        notes=notes,
        due_date=due_date,
        task_list_id=task_list_id,
    )


@server.tool()
def get_tasks(
    task_list_id: str = "@default",
    show_completed: bool = False,
    max_results: int = 100,
) -> dict:
    """Retrieve tasks from a Google Tasks list."""

    return tasks_get_tasks(
        task_list_id=task_list_id,
        show_completed=show_completed,
        max_results=max_results,
    )


@server.tool()
def complete_task(
    task_id: str,
    task_list_id: str = "@default",
) -> dict:
    """Mark a Google Task as completed."""

    return tasks_complete_task(
        task_id=task_id,
        task_list_id=task_list_id,
    )


@server.tool()
def delete_task(
    task_id: str,
    task_list_id: str = "@default",
) -> dict:
    """Delete a task. Requires explicit user confirmation."""

    return tasks_delete_task(
        task_id=task_id,
        task_list_id=task_list_id,
    )


@server.tool()
def get_task_lists() -> dict:
    """Retrieve available Google Tasks lists."""

    return tasks_get_task_lists()


# ============================================================
# SERVER STARTUP
# ============================================================

async def main():
    await server.run_stdio_async()


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except Exception:
        print(
            "[PRODUCTIVITY MCP] Server startup failed.",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        raise