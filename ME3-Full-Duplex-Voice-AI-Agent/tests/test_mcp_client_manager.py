
import asyncio
import json

from src.mcp.client_manager import MCPClientManager


async def main():

    manager = MCPClientManager()

    try:
        print("\n[TEST] Connecting to MCP servers...")
        await manager.connect_all()

        print("\n[TEST] Discovering MCP tools...")
        await manager.discover_tools()

        # --------------------------------------------------
        # TEST 1: SYSTEM MCP TOOL ROUTING
        # --------------------------------------------------

        print("\n" + "=" * 60)
        print("TEST 1: SYSTEM MCP TOOL ROUTING")
        print("=" * 60)

        ram_result = await manager.call_tool(
            tool_name="ram_usage",
            arguments={},
        )

        print(json.dumps(ram_result, indent=2))

        assert ram_result["success"] is True
        assert ram_result["server"] == "system"
        assert isinstance(ram_result["result"], dict)

        print("\n[TEST PASSED] System tool routing works.")

        # --------------------------------------------------
        # TEST 2: MUSIC MCP TOOL ROUTING
        # --------------------------------------------------

        print("\n" + "=" * 60)
        print("TEST 2: MUSIC MCP TOOL ROUTING")
        print("=" * 60)

        music_result = await manager.call_tool(
            tool_name="get_music_volume",
            arguments={},
        )

        print(json.dumps(music_result, indent=2))

        print("\nRAW NORMALIZED MUSIC RESULT:")
        print(repr(music_result["result"]))

        assert music_result["success"] is True
        assert music_result["server"] == "music"
        assert music_result["tool"] == "get_music_volume"

        print("\n[TEST PASSED] Music tool routing works.")

        # --------------------------------------------------
        # TEST 3: UNKNOWN TOOL HANDLING
        # --------------------------------------------------

        print("\n" + "=" * 60)
        print("TEST 3: UNKNOWN TOOL HANDLING")
        print("=" * 60)

        try:
            await manager.call_tool(
                tool_name="unknown_tool",
                arguments={},
            )

        except ValueError as error:
            print(f"[EXPECTED ERROR] {error}")

        else:
            raise AssertionError(
                "Unknown tool was not rejected."
            )

        print("\n[TEST PASSED] Unknown tools are rejected.")

        print("\n" + "=" * 60)
        print("ALL ROUTING TESTS PASSED")
        print("=" * 60)

    except Exception as error:
        print(f"\n[TEST ERROR] {type(error).__name__}: {error}")
        raise

    finally:
        print("\n[TEST] Closing MCP connections...")
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
