
from mcp.server.mcpserver import MCPServer

from src.mcp_servers.website.website_manager import (
    create_website_project as manager_create_website_project,
    list_website_projects as manager_list_website_projects,
    read_website_file as manager_read_website_file,
    write_website_file as manager_write_website_file,
    delete_website_project as manager_delete_website_project,
    download_website_asset as manager_download_website_asset,
)

from src.mcp_servers.website.github_service import (
    publish_website as github_publish_website,
    update_published_website as github_update_published_website,
    get_website_status as github_get_website_status,
)


server = MCPServer(
    name="Website MCP Server",
    instructions=(
        "Manages static website projects locally and publishes "
        "them through GitHub Pages. "
        "Gemini is responsible for planning website content and "
        "generating HTML, CSS, and JavaScript. "
        "Use website project tools to create, inspect, read, "
        "and update local website files. "
        "Use download_website_asset to download permitted images "
        "into a website project's assets directory. "
        "Use the returned relative path in HTML and CSS. "
        "Only supported image formats can be downloaded. "
        "Use GitHub publishing tools to create new repositories "
        "and publish or update agent-managed websites. "
        "Never claim that a website was published or updated "
        "unless the corresponding tool confirms success. "
        "Never delete a website project without explicit "
        "user confirmation."
    ),
)


# ============================================================
# LOCAL WEBSITE PROJECT TOOLS
# ============================================================

@server.tool()
def create_website_project(
    project_name: str,
    description: str = "",
) -> dict:
    """Create a website project with standard starter files."""
    return manager_create_website_project(
        project_name,
        description
    )


@server.tool()
def list_website_projects() -> dict:
    """List locally managed website projects."""
    return manager_list_website_projects()


@server.tool()
def read_website_file(
    project_name: str,
    relative_path: str,
) -> dict:
    """Read a text file from a website project."""
    return manager_read_website_file(
        project_name,
        relative_path
    )


@server.tool()
def write_website_file(
    project_name: str,
    relative_path: str,
    content: str,
) -> dict:
    """Create or update a text file inside a website project."""
    return manager_write_website_file(
        project_name,
        relative_path,
        content
    )


@server.tool()
def delete_website_project(project_name: str) -> dict:
    """Permanently delete a local website project.

    Requires explicit user confirmation before execution.
    """
    return manager_delete_website_project(
        project_name
    )


@server.tool()
def download_website_asset(
    project_name: str,
    url: str,
    filename: str = "image",
) -> dict:
    """Download a validated image into a website project's assets directory.

    Supports PNG, JPEG, WebP, and GIF images.
    Maximum download size is 5 MB.
    Existing files are never overwritten.
    """
    return manager_download_website_asset(
        project_name,
        url,
        filename
    )


# ============================================================
# GITHUB PUBLISHING TOOLS
# ============================================================

@server.tool()
def publish_website(
    project_name: str,
    repository_name: str = "",
    description: str = "Website created by AI-231 Voice Assistant",
) -> dict:
    """Publish a website to a new public GitHub repository.

    Publishes all supported files in one Git commit and
    configures GitHub Pages.
    """
    return github_publish_website(
        project_name=project_name,
        repository_name=repository_name or None,
        description=description,
        private=False,
    )


@server.tool()
def update_published_website(project_name: str) -> dict:
    """Update a website in its registered agent-managed repository."""
    return github_update_published_website(
        project_name
    )


@server.tool()
def get_website_status(project_name: str) -> dict:
    """Check repository and GitHub Pages deployment status."""
    return github_get_website_status(
        project_name
    )


# ============================================================
# SERVER ENTRY POINT
# ============================================================

async def main():
    await server.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
