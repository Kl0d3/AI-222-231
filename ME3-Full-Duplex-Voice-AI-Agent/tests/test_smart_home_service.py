
from src.mcp_servers.smart_home.smart_home_service import (
    smart_home_service,
)


def run_test(name, condition):
    if condition:
        print(f"PASS: {name}")
    else:
        raise AssertionError(f"FAIL: {name}")


def main():
    print("=" * 55)
    print("SMART HOME SERVICE TESTS")
    print("=" * 55)

    # Test 1: List devices
    result = smart_home_service.list_smart_home_devices()

    run_test("Device listing succeeds", result["success"])
    run_test("Five devices registered", result["device_count"] == 5)

    # Test 2: Initial light state
    result = smart_home_service.get_light_status("Living Room Light")

    run_test("Initial light is off", result["is_on"] is False)
    run_test("Initial brightness is 100%", result["brightness"] == 100)

    # Test 3: Turn light on
    result = smart_home_service.control_light(
        "Living Room Light",
        "turn_on",
    )

    run_test("Turn light on", result["success"] and result["is_on"])

    # Test 4: Adjust brightness
    result = smart_home_service.control_light(
        "Living Room Light",
        "set_brightness",
        50,
    )

    run_test(
        "Set brightness to 50%",
        result["success"] and result["brightness"] == 50,
    )

    # Test 5: Brightness zero turns light off
    result = smart_home_service.control_light(
        "Living Room Light",
        "set_brightness",
        0,
    )

    run_test(
        "Zero brightness turns light off",
        result["success"] and result["is_on"] is False,
    )

    # Test 6: Turn light off
    smart_home_service.control_light(
        "Bedroom Light",
        "turn_on",
    )

    result = smart_home_service.control_light(
        "Bedroom Light",
        "turn_off",
    )

    run_test("Turn light off", result["success"] and not result["is_on"])

    # Test 7: Case-insensitive device names
    result = smart_home_service.get_light_status("kitchen light")

    run_test("Case-insensitive device lookup", result["success"])

    # Test 8: Invalid brightness
    result = smart_home_service.control_light(
        "Kitchen Light",
        "set_brightness",
        150,
    )

    run_test("Reject brightness above 100%", not result["success"])

    # Test 9: Unknown light
    result = smart_home_service.get_light_status("Garage Light")

    run_test("Reject unknown light", not result["success"])

    # Test 10: Initial thermostat state
    result = smart_home_service.get_thermostat_status(
        "Living Room Thermostat"
    )

    run_test("Thermostat status succeeds", result["success"])
    run_test(
        "Initial current temperature is 24°C",
        result["current_temperature"] == 24.0,
    )

    # Test 11: Set target temperature
    result = smart_home_service.set_thermostat_temperature(
        "Living Room Thermostat",
        22,
    )

    run_test(
        "Set target temperature to 22°C",
        result["success"] and result["target_temperature"] == 22.0,
    )

    # Test 12: Invalid thermostat temperature
    result = smart_home_service.set_thermostat_temperature(
        "Living Room Thermostat",
        40,
    )

    run_test("Reject temperature above 35°C", not result["success"])

    # Test 13: Unknown thermostat
    result = smart_home_service.get_thermostat_status("Office Thermostat")

    run_test("Reject unknown thermostat", not result["success"])

    print("=" * 55)
    print("ALL SMART HOME SERVICE TESTS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    main()
