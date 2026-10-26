"""
src/tools/weather_tool.py

Fetches current weather data from the OpenWeatherMap API with dual Celsius & Fahrenheit metrics.
Falls back to a structured mock tool when no API key is configured.
"""

import requests

try:
    from langchain_core.tools import Tool
except ImportError:
    from langchain.tools import Tool  # type: ignore


def create_weather_tool(openweathermap_api_key: str) -> Tool:
    """
    Build a LangChain Tool that returns current weather from OpenWeatherMap.
    """

    def get_weather(city: str) -> str:
        """
        Call the OpenWeatherMap endpoint for a given city and return summary.
        """
        city = city.strip()
        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={openweathermap_api_key}&units=metric"
        )

        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 404:
                return (
                    f"Could not find weather for '{city}'. "
                    "Please check the city name (e.g., 'London', 'Tokyo', 'New York')."
                )

            response.raise_for_status()
            data = response.json()

            temp_c = data["main"]["temp"]
            temp_f = (temp_c * 9 / 5) + 32
            feels_like_c = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            raw_desc = data["weather"][0]["description"].lower()
            description = data["weather"][0]["description"].capitalize()
            wind_speed_ms = data["wind"]["speed"]
            wind_kmh = wind_speed_ms * 3.6
            city_name = data.get("name", city)
            country = data.get("sys", {}).get("country", "")
            location = f"{city_name}, {country}" if country else city_name

            emoji = "🌤️"
            if "rain" in raw_desc or "drizzle" in raw_desc:
                emoji = "🌧️"
            elif "thunder" in raw_desc:
                emoji = "🌩️"
            elif "snow" in raw_desc:
                emoji = "❄️"
            elif "clear" in raw_desc or "sun" in raw_desc:
                emoji = "☀️"
            elif "cloud" in raw_desc:
                emoji = "☁️"
            elif "fog" in raw_desc or "mist" in raw_desc:
                emoji = "🌫️"

            return (
                f"{emoji} Weather in {location}: "
                f"{temp_c:.1f}°C / {temp_f:.1f}°F (feels like {feels_like_c:.1f}°C), "
                f"{description}, "
                f"Humidity: {humidity}%, "
                f"Wind: {wind_kmh:.1f} km/h"
            )

        except requests.exceptions.Timeout:
            return f"Weather service timed out for '{city}'. Please try again."
        except Exception as exc:
            return f"Could not retrieve weather for '{city}'. Error: {exc}"

    return Tool(
        name="get_weather",
        func=get_weather,
        description=(
            "Get current weather and meteorological conditions for any city. "
            "Input: city name (e.g., 'London', 'Tokyo', or 'New York, US'). "
            "Returns temperature (°C/°F), condition description, humidity, and wind speed."
        ),
    )


def create_mock_weather_tool() -> Tool:
    """
    Return a mock weather tool used when no OpenWeatherMap API key is set.
    """

    def mock_weather(city: str) -> str:
        city = city.strip()
        return (
            f"[MOCK DATA — configure OPENWEATHERMAP_API_KEY for live data] "
            f"🌤️ Weather in {city}: 19.5°C / 67.1°F (feels like 18.5°C), "
            f"Partly cloudy, Humidity: 62%, Wind: 14.5 km/h"
        )

    return Tool(
        name="get_weather",
        func=mock_weather,
        description=(
            "Get current weather and meteorological conditions for any city. "
            "NOTE: Running in mock mode (no API key configured). "
            "Input: city name (e.g., 'London' or 'Tokyo')."
        ),
    )
