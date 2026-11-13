
import threading


class SmartHomeService:
    """Manages simulated smart lights and thermostats."""

    def __init__(self):
        self._lock = threading.RLock()

        self._lights = {
            "Living Room Light": {
                "type": "light",
                "is_on": False,
                "brightness": 100,
            },
            "Bedroom Light": {
                "type": "light",
                "is_on": False,
                "brightness": 100,
            },
            "Kitchen Light": {
                "type": "light",
                "is_on": False,
                "brightness": 100,
            },
        }

        self._thermostats = {
            "Living Room Thermostat": {
                "type": "thermostat",
                "current_temperature": 24.0,
                "target_temperature": 24.0,
            },
            "Bedroom Thermostat": {
                "type": "thermostat",
                "current_temperature": 24.0,
                "target_temperature": 24.0,
            },
        }

    def list_smart_home_devices(self):
        """Return all available simulated devices."""

        with self._lock:
            devices = []

            for name, state in self._lights.items():
                devices.append({
                    "name": name,
                    "type": state["type"],
                })

            for name, state in self._thermostats.items():
                devices.append({
                    "name": name,
                    "type": state["type"],
                })

            return {
                "success": True,
                "devices": devices,
                "device_count": len(devices),
                "message": "Simulated smart home devices retrieved.",
            }

    def get_light_status(self, device_name):
        """Return the current state of a simulated light."""

        with self._lock:
            name = self._find_device(device_name, self._lights)

            if name is None:
                return self._unknown_device(
                    device_name,
                    self._lights,
                    "light",
                )

            state = self._lights[name]

            return {
                "success": True,
                "device": name,
                "is_on": state["is_on"],
                "brightness": state["brightness"],
                "brightness_unit": "percent",
                "message": f"{name} is {'on' if state['is_on'] else 'off'} "
                           f"at {state['brightness']}% brightness.",
            }

    def control_light(self, device_name, action, brightness=None):
        """
        Control a simulated light.

        Supported actions:
        - turn_on
        - turn_off
        - set_brightness

        Setting brightness to 0 turns the light off.
        Setting brightness above 0 turns it on.
        """

        with self._lock:
            name = self._find_device(device_name, self._lights)

            if name is None:
                return self._unknown_device(
                    device_name,
                    self._lights,
                    "light",
                )

            if action not in ("turn_on", "turn_off", "set_brightness"):
                return {
                    "success": False,
                    "error": (
                        "Unsupported action. Use turn_on, turn_off, "
                        "or set_brightness."
                    ),
                }

            state = self._lights[name]

            if action == "turn_on":
                state["is_on"] = True

            elif action == "turn_off":
                state["is_on"] = False

            elif action == "set_brightness":
                if brightness is None:
                    return {
                        "success": False,
                        "error": "Brightness is required for set_brightness.",
                    }

                if isinstance(brightness, bool) or not isinstance(
                    brightness, (int, float)
                ):
                    return {
                        "success": False,
                        "error": "Brightness must be a number from 0 to 100.",
                    }

                if not 0 <= brightness <= 100:
                    return {
                        "success": False,
                        "error": "Brightness must be between 0 and 100.",
                    }

                state["brightness"] = int(brightness)
                state["is_on"] = brightness > 0

            return {
                "success": True,
                "device": name,
                "is_on": state["is_on"],
                "brightness": state["brightness"],
                "message": (
                    f"{name} is now "
                    f"{'on' if state['is_on'] else 'off'} "
                    f"at {state['brightness']}% brightness."
                ),
            }

    def get_thermostat_status(self, device_name):
        """Return the current and target temperatures."""

        with self._lock:
            name = self._find_device(device_name, self._thermostats)

            if name is None:
                return self._unknown_device(
                    device_name,
                    self._thermostats,
                    "thermostat",
                )

            state = self._thermostats[name]

            return {
                "success": True,
                "device": name,
                "current_temperature": state["current_temperature"],
                "target_temperature": state["target_temperature"],
                "temperature_unit": "Celsius",
                "message": (
                    f"{name}: current temperature is "
                    f"{state['current_temperature']}°C, "
                    f"target temperature is "
                    f"{state['target_temperature']}°C."
                ),
            }

    def set_thermostat_temperature(self, device_name, temperature):
        """Change a simulated thermostat's target temperature."""

        with self._lock:
            name = self._find_device(device_name, self._thermostats)

            if name is None:
                return self._unknown_device(
                    device_name,
                    self._thermostats,
                    "thermostat",
                )

            if isinstance(temperature, bool) or not isinstance(
                temperature, (int, float)
            ):
                return {
                    "success": False,
                    "error": "Temperature must be a number.",
                }

            if not 10 <= temperature <= 35:
                return {
                    "success": False,
                    "error": (
                        "Temperature must be between 10°C and 35°C."
                    ),
                }

            state = self._thermostats[name]
            state["target_temperature"] = float(temperature)

            return {
                "success": True,
                "device": name,
                "current_temperature": state["current_temperature"],
                "target_temperature": state["target_temperature"],
                "temperature_unit": "Celsius",
                "message": (
                    f"{name} target temperature set to "
                    f"{temperature}°C."
                ),
            }

    @staticmethod
    def _find_device(device_name, devices):
        """Find a device using a case-insensitive name match."""

        if not isinstance(device_name, str):
            return None

        requested_name = device_name.strip().casefold()

        for name in devices:
            if name.casefold() == requested_name:
                return name

        return None

    @staticmethod
    def _unknown_device(device_name, devices, device_type):
        """Build a helpful response for an unknown device."""

        available = list(devices.keys())

        return {
            "success": False,
            "error": f"Unknown {device_type}: {device_name}.",
            "available_devices": available,
        }


# One shared service instance for the MCP server.
smart_home_service = SmartHomeService()
