
import json

from src.mcp.background_manager import MCPBackgroundManager


def main():

    manager = MCPBackgroundManager()

    try:

        # ================================================
        # TEST 1: START BACKGROUND MANAGER
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 1: START BACKGROUND MANAGER")
        print("=" * 60)

        manager.start()

        assert manager.started is True
        assert manager.loop.is_running()
        assert manager.thread.is_alive()

        print("\n[TEST PASSED] Background manager started.")

        # ================================================
        # TEST 2: DISCOVER GEMINI TOOLS
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 2: DISCOVER GEMINI TOOLS")
        print("=" * 60)

        tools = manager.get_gemini_tools()

        print(f"\nDiscovered tools: {len(tools)}")

        for tool in tools:
            print(f"- {tool['name']}")

        assert len(tools) == 41

        print("\n[TEST PASSED] All 41 tools discovered.")

        # ================================================
        # TEST 3: SYSTEM TOOL EXECUTION
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 3: SYSTEM TOOL EXECUTION")
        print("=" * 60)

        ram_result = manager.call_tool(
            tool_name="ram_usage",
            arguments={}
        )

        print(json.dumps(ram_result, indent=2))

        assert ram_result["success"] is True
        assert ram_result["server"] == "system"

        print("\n[TEST PASSED] System tool executed.")

        # ================================================
        # TEST 4: MUSIC TOOL EXECUTION
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 4: MUSIC TOOL EXECUTION")
        print("=" * 60)

        music_result = manager.call_tool(
            tool_name="get_music_volume",
            arguments={}
        )

        print(json.dumps(music_result, indent=2))

        assert music_result["success"] is True
        assert music_result["server"] == "music"

        print("\n[TEST PASSED] Music tool executed.")

        # ================================================
        # TEST 5: WEATHER TOOL EXECUTION
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 5: WEATHER TOOL EXECUTION")
        print("=" * 60)

        weather_result = manager.call_tool(
            tool_name="get_current_weather",
            arguments={
                "location": "Cebu City"
            }
        )

        print(json.dumps(weather_result, indent=2))

        assert weather_result["success"] is True
        assert weather_result["server"] == "information"

        weather_data = weather_result["result"]

        assert isinstance(weather_data, dict)
        assert weather_data["location"] == "Cebu City"
        assert "temperature_c" in weather_data
        assert "weather_description" in weather_data
        assert "humidity_percent" in weather_data

        print("\n[TEST PASSED] Weather tool executed.")
        print("[TEST PASSED] Weather data received successfully.")

        # ================================================
        # TEST 6: WEB SEARCH TOOL EXECUTION
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 6: WEB SEARCH TOOL EXECUTION")
        print("=" * 60)

        search_result = manager.call_tool(
            tool_name="web_search",
            arguments={
                "query": "latest developments in solid-state batteries",
                "max_results": 3
            }
        )

        print(json.dumps(search_result, indent=2))

        assert search_result["success"] is True
        assert search_result["server"] == "information"

        search_data = search_result["result"]

        assert isinstance(search_data, dict)
        assert search_data["success"] is True
        assert search_data["query"] == (
            "latest developments in solid-state batteries"
        )
        assert isinstance(search_data["results"], list)
        assert len(search_data["results"]) > 0
        assert search_data["result_count"] == len(
            search_data["results"]
        )

        for result in search_data["results"]:
            assert isinstance(result, dict)
            assert "title" in result
            assert "url" in result
            assert "content" in result

        print("\n[TEST PASSED] Web search tool executed.")
        print("[TEST PASSED] Search results received successfully.")

        # ================================================
        # TEST 7: CONNECTION REUSE
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 7: CONNECTION REUSE")
        print("=" * 60)

        original_loop = manager.loop
        original_thread = manager.thread

        manager.start()

        assert manager.loop is original_loop
        assert manager.thread is original_thread
        assert manager.started is True

        print("\n[TEST PASSED] Existing connections are reused.")

        # ================================================
        # TEST 8: WEBSITE PROJECT LISTING
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 8: WEBSITE PROJECT LISTING")
        print("=" * 60)

        website_list_result = manager.call_tool(
            tool_name="list_website_projects",
            arguments={}
        )

        print(json.dumps(website_list_result, indent=2))

        assert website_list_result["success"] is True
        assert website_list_result["server"] == "website"

        website_list_text = json.dumps(
            website_list_result["result"]
        )

        assert "website-manager-test" in website_list_text

        print("\n[TEST PASSED] Website project listing executed.")
        print("[TEST PASSED] Existing test project was found.")

        # ================================================
        # TEST 9: WEBSITE FILE READING
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 9: WEBSITE FILE READING")
        print("=" * 60)

        website_read_result = manager.call_tool(
            tool_name="read_website_file",
            arguments={
                "project_name": "website-manager-test",
                "relative_path": "index.html"
            }
        )

        print(json.dumps(website_read_result, indent=2))

        assert website_read_result["success"] is True
        assert website_read_result["server"] == "website"

        website_read_data = website_read_result["result"]

        assert isinstance(website_read_data, dict)
        assert "content" in website_read_data
        assert "Website manager test passed." in (
            website_read_data["content"]
        )

        print("\n[TEST PASSED] Website file reading executed.")
        print("[TEST PASSED] index.html content received.")

        # ================================================
        # TEST 10: WEBSITE FILE WRITING
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 10: WEBSITE FILE WRITING")
        print("=" * 60)

        test_content = (
            "Website MCP integration test passed.\n"
            "File writing through the central manager works."
        )

        website_write_result = manager.call_tool(
            tool_name="write_website_file",
            arguments={
                "project_name": "website-manager-test",
                "relative_path": "mcp-integration-test.txt",
                "content": test_content
            }
        )

        print(json.dumps(website_write_result, indent=2))

        assert website_write_result["success"] is True
        assert website_write_result["server"] == "website"

        print("\n[TEST PASSED] Website file writing executed.")

        # ================================================
        # TEST 11: VERIFY WRITTEN FILE
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 11: VERIFY WRITTEN FILE")
        print("=" * 60)

        website_verify_result = manager.call_tool(
            tool_name="read_website_file",
            arguments={
                "project_name": "website-manager-test",
                "relative_path": "mcp-integration-test.txt"
            }
        )

        print(json.dumps(website_verify_result, indent=2))

        assert website_verify_result["success"] is True
        assert website_verify_result["server"] == "website"

        verified_data = website_verify_result["result"]

        assert isinstance(verified_data, dict)
        assert verified_data["content"] == test_content

        print("\n[TEST PASSED] Written file was read successfully.")
        print("[TEST PASSED] File contents match exactly.")

        # ================================================
        # TEST 12: LIST SMART HOME DEVICES
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 12: SMART HOME DEVICE LISTING")
        print("=" * 60)

        devices_result = manager.call_tool(
            tool_name="list_smart_home_devices",
            arguments={}
        )

        print(json.dumps(devices_result, indent=2))

        assert devices_result["success"] is True
        assert devices_result["server"] == "smart_home"

        devices_data = devices_result["result"]

        assert isinstance(devices_data, dict)
        assert devices_data["success"] is True

        print("\n[TEST PASSED] Smart Home devices listed.")

        # ================================================
        # TEST 13: TURN ON LIVING ROOM LIGHT
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 13: CONTROL SMART LIGHT")
        print("=" * 60)

        light_control_result = manager.call_tool(
            tool_name="control_light",
            arguments={
                "device_name": "Living Room Light",
                "action": "turn_on"
            }
        )

        print(json.dumps(light_control_result, indent=2))

        assert light_control_result["success"] is True
        assert light_control_result["server"] == "smart_home"

        print("\n[TEST PASSED] Living Room Light control executed.")

        # ================================================
        # TEST 14: VERIFY LIGHT STATUS
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 14: VERIFY SMART LIGHT STATUS")
        print("=" * 60)

        light_status_result = manager.call_tool(
            tool_name="get_light_status",
            arguments={
                "device_name": "Living Room Light"
            }
        )

        print(json.dumps(light_status_result, indent=2))

        assert light_status_result["success"] is True
        assert light_status_result["server"] == "smart_home"

        light_data = light_status_result["result"]

        assert isinstance(light_data, dict)
        assert light_data["success"] is True

        print("\n[TEST PASSED] Smart light status retrieved.")

        # ================================================
        # TEST 15: SET THERMOSTAT TEMPERATURE
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 15: SET THERMOSTAT TEMPERATURE")
        print("=" * 60)

        thermostat_set_result = manager.call_tool(
            tool_name="set_thermostat_temperature",
            arguments={
                "device_name": "Living Room Thermostat",
                "temperature": 22
            }
        )

        print(json.dumps(thermostat_set_result, indent=2))

        assert thermostat_set_result["success"] is True
        assert thermostat_set_result["server"] == "smart_home"

        print("\n[TEST PASSED] Thermostat temperature updated.")

        # ================================================
        # TEST 16: VERIFY THERMOSTAT TEMPERATURE
        # ================================================

        print("\n" + "=" * 60)
        print("TEST 16: VERIFY THERMOSTAT TEMPERATURE")
        print("=" * 60)

        thermostat_status_result = manager.call_tool(
            tool_name="get_thermostat_status",
            arguments={
                "device_name": "Living Room Thermostat"
            }
        )

        print(json.dumps(thermostat_status_result, indent=2))

        assert thermostat_status_result["success"] is True
        assert thermostat_status_result["server"] == "smart_home"

        thermostat_data = thermostat_status_result["result"]

        assert isinstance(thermostat_data, dict)
        assert thermostat_data["success"] is True

        assert float(
            thermostat_data["target_temperature"]
        ) == 22.0

        print("\n[TEST PASSED] Thermostat target verified at 22°C.")

        # ================================================
        # ALL TESTS PASSED
        # ================================================

        print("\n" + "=" * 60)
        print("ALL BACKGROUND MANAGER TESTS PASSED")
        print("=" * 60)

    except Exception as error:

        print(
            f"\n[TEST ERROR] {type(error).__name__}: {error}"
        )

        raise

    finally:

        print("\n[TEST] Closing background manager...")

        manager.close()


if __name__ == "__main__":

    main()
