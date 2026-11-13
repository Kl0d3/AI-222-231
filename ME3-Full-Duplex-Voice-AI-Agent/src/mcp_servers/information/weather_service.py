
import requests


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

REQUEST_TIMEOUT = 10


WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def get_coordinates(location: str) -> dict:
    """
    Convert a location name into geographic coordinates
    using the Open-Meteo Geocoding API.
    """

    if not isinstance(location, str) or not location.strip():
        raise ValueError("Location must be a non-empty string.")

    try:
        response = requests.get(
            GEOCODING_URL,
            params={
                "name": location.strip(),
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()
        data = response.json()

    except requests.RequestException as error:
        raise RuntimeError(
            f"Location lookup failed: {error}"
        ) from error

    except ValueError as error:
        raise RuntimeError(
            "Location service returned invalid JSON."
        ) from error

    results = data.get("results", [])

    if not results:
        raise ValueError(
            f"Could not find a location matching '{location}'."
        )

    result = results[0]

    return {
        "name": result.get("name"),
        "country": result.get("country"),
        "admin1": result.get("admin1"),
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "timezone": result.get("timezone", "auto"),
    }


def get_weather(location: str) -> dict:
    """
    Retrieve current weather conditions for a location
    using the Open-Meteo Forecast API.
    """

    coordinates = get_coordinates(location)

    try:
        response = requests.get(
            WEATHER_URL,
            params={
                "latitude": coordinates["latitude"],
                "longitude": coordinates["longitude"],
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "apparent_temperature,"
                    "is_day,"
                    "precipitation,"
                    "weather_code,"
                    "wind_speed_10m"
                ),
                "timezone": coordinates["timezone"],
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()
        data = response.json()

    except requests.RequestException as error:
        raise RuntimeError(
            f"Weather retrieval failed: {error}"
        ) from error

    except ValueError as error:
        raise RuntimeError(
            "Weather service returned invalid JSON."
        ) from error

    current = data.get("current")

    if not current:
        raise RuntimeError(
            "Weather service did not return current conditions."
        )

    weather_code = current.get("weather_code")

    return {
        "location": coordinates["name"],
        "region": coordinates["admin1"],
        "country": coordinates["country"],
        "temperature_c": current.get("temperature_2m"),
        "feels_like_c": current.get("apparent_temperature"),
        "humidity_percent": current.get("relative_humidity_2m"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "precipitation_mm": current.get("precipitation"),
        "weather_code": weather_code,
        "weather_description": WEATHER_CODES.get(
            weather_code,
            "Unknown weather condition",
        ),
        "is_day": bool(current.get("is_day")),
        "observation_time": current.get("time"),
        "timezone": data.get("timezone"),
        "temperature_unit": data.get("current_units", {}).get(
            "temperature_2m"
        ),
        "wind_speed_unit": data.get("current_units", {}).get(
            "wind_speed_10m"
        ),
        "source": "Open-Meteo",
    }
