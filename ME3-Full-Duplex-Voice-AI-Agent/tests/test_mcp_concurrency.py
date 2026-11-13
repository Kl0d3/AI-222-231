
import json
import threading
import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.mcp.background_manager import MCPBackgroundManager


def main():

    manager = MCPBackgroundManager()

    try:

        # ================================================
        # TEST 1: START MANAGER
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 1: START MCP BACKGROUND MANAGER")
        print("=" * 60)

        manager.start()

        assert manager.started is True

        print("\n[TEST PASSED] Manager started.")

        # ================================================
        # TEST 2: REPEATED REQUESTS
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 2: REPEATED TOOL REQUESTS")
        print("=" * 60)

        for index in range(5):

            result = manager.call_tool(
                tool_name="ram_usage",
                arguments={}
            )

            assert result["success"] is True

            print(
                f"Request {index + 1}: "
                f"{result['result']['usage_percent']}% RAM usage"
            )

        print("\n[TEST PASSED] Repeated requests succeeded.")

        # ================================================
        # TEST 3: DIFFERENT MCP SERVERS
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 3: DIFFERENT MCP SERVER REQUESTS")
        print("=" * 60)

        system_result = manager.call_tool(
            tool_name="cpu_usage",
            arguments={}
        )

        music_result = manager.call_tool(
            tool_name="get_music_volume",
            arguments={}
        )

        assert system_result["success"] is True
        assert system_result["server"] == "system"

        assert music_result["success"] is True
        assert music_result["server"] == "music"

        print("\nSYSTEM RESULT:")
        print(json.dumps(system_result, indent=2))

        print("\nMUSIC RESULT:")
        print(json.dumps(music_result, indent=2))

        print("\n[TEST PASSED] Both MCP servers responded.")

        # ================================================
        # TEST 4: CONCURRENT REQUESTS
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 4: CONCURRENT MCP REQUESTS")
        print("=" * 60)

        requests = [
            ("ram_usage", {}),
            ("cpu_usage", {}),
            ("system_uptime", {}),
            ("get_music_volume", {}),
            ("get_music_status", {}),
            ("ram_usage", {}),
            ("cpu_usage", {}),
            ("system_uptime", {}),
        ]

        results = []

        def execute_request(request):

            tool_name, arguments = request

            thread_name = threading.current_thread().name

            print(
                f"\n[{thread_name}] Submitting {tool_name}"
            )

            result = manager.call_tool(
                tool_name=tool_name,
                arguments=arguments
            )

            return result

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=4) as executor:

            futures = [
                executor.submit(execute_request, request)
                for request in requests
            ]

            for future in as_completed(futures):

                result = future.result()

                results.append(result)

                print(
                    f"[COMPLETED] {result['server']}: "
                    f"{result['tool']}"
                )

        elapsed = time.perf_counter() - start_time

        assert len(results) == len(requests)

        assert all(
            result["success"] is True
            for result in results
        )

        print(f"\nCompleted requests: {len(results)}")
        print(f"Elapsed time: {elapsed:.2f} seconds")

        print("\n[TEST PASSED] Concurrent requests succeeded.")

        # ================================================
        # TEST 5: CONNECTION STABILITY
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 5: CONNECTION STABILITY")
        print("=" * 60)

        assert manager.started is True
        assert manager.loop.is_running()
        assert manager.thread.is_alive()

        final_result = manager.call_tool(
            tool_name="ram_usage",
            arguments={}
        )

        assert final_result["success"] is True

        print("\n[TEST PASSED] Manager remains operational.")

        # ================================================
        # ALL TESTS PASSED
        # ================================================

        print("\n" + "=" * 60)
        print("ALL MCP CONCURRENCY TESTS PASSED")
        print("=" * 60)

    except Exception as error:

        print(
            f"\n[TEST ERROR] {type(error).__name__}: {error}"
        )

        raise

    finally:

        print("\n[TEST] Closing MCP background manager...")

        manager.close()


if __name__ == "__main__":

    main()
