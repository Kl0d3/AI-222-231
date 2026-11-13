
from src.mcp_servers.website.server import server


def main():

    print("=" * 60)
    print("WEBSITE MCP SERVER REGISTRATION TEST")
    print("=" * 60)

    expected_tools = {
        "create_website_project",
        "list_website_projects",
        "read_website_file",
        "write_website_file",
        "delete_website_project",
        "download_website_asset",
        "publish_website",
        "update_published_website",
        "get_website_status",
    }

    tools = server._tool_manager.list_tools()

    discovered_tools = {
        tool.name for tool in tools
    }

    print(f"\nTools discovered: {len(discovered_tools)}")

    for name in sorted(discovered_tools):
        status = "PASS" if name in expected_tools else "UNEXPECTED"
        print(f"{status}: {name}")

    missing_tools = expected_tools - discovered_tools
    unexpected_tools = discovered_tools - expected_tools

    print("\n" + "=" * 60)

    if not missing_tools and not unexpected_tools:
        print("ALL 9 WEBSITE MCP TOOLS REGISTERED SUCCESSFULLY")
    else:
        if missing_tools:
            print("MISSING TOOLS:", sorted(missing_tools))

        if unexpected_tools:
            print("UNEXPECTED TOOLS:", sorted(unexpected_tools))

        raise AssertionError("Website MCP registration test failed.")

    print("=" * 60)


if __name__ == "__main__":
    main()
