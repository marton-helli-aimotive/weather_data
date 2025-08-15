"""Test WeatherAPI integration."""

import asyncio
from datetime import datetime, timezone
import os

from src.weather_pipeline.api import WeatherClientFactory
from src.weather_pipeline.models.weather import WeatherProvider, Coordinates


async def test_weatherapi():
    """Test WeatherAPI integration."""
    print("🌤️  Testing WeatherAPI integration...")
    
    # Test coordinates (London)
    coordinates = Coordinates(latitude=51.5074, longitude=-0.1278)
    
    try:
        # Load API key from environment variable
        api_key = os.environ.get("WEATHERAPI_KEY")
        if not api_key:
            raise ValueError("WEATHERAPI_KEY environment variable not set.")

        # Create WeatherAPI client
        client = WeatherClientFactory.create_client(
            WeatherProvider.WEATHERAPI, 
            api_key=api_key
        )
        
        print(f"   ✅ WeatherAPI client created")
        print(f"   Circuit breaker state: {client.get_circuit_breaker_state()}")
        
        # Test current weather
        weather_data = await client.get_current_weather(coordinates, "London", "UK")
        
        if weather_data:
            point = weather_data[0]
            print(f"   ✅ Current weather retrieved!")
            print(f"   City: {point.city}, {point.country}")
            print(f"   Temperature: {point.temperature}°C")
            print(f"   Humidity: {point.humidity}%")
            print(f"   Pressure: {point.pressure} hPa")
            print(f"   Wind Speed: {point.wind_speed} m/s")
            print(f"   Provider: {point.provider.value}")
            print(f"   Timestamp: {point.timestamp}")
            print(f"   Confidence: {point.confidence_score}")
        else:
            print("   ❌ No weather data returned")
            
    except Exception as e:
        print(f"   ❌ WeatherAPI test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_weatherapi())
