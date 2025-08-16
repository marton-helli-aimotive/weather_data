"""Integration tests for WeatherAPI provider."""

import pytest
import asyncio
import os
from datetime import datetime, timezone
from unittest.mock import patch

from weather_pipeline.api import WeatherClientFactory
from weather_pipeline.models.weather import WeatherProvider, Coordinates


class TestWeatherAPIIntegration:
    """Integration tests for WeatherAPI client."""

    @pytest.fixture
    def london_coordinates(self):
        """London coordinates for testing."""
        return Coordinates(latitude=51.5074, longitude=-0.1278)

    @pytest.fixture
    def weatherapi_key(self):
        """Get WeatherAPI key from environment."""
        return os.environ.get("WEATHERAPI_KEY")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_weatherapi_client_creation(self, weatherapi_key):
        """Test WeatherAPI client creation."""
        if not weatherapi_key:
            pytest.skip("WEATHERAPI_KEY environment variable not set")
        
        client = WeatherClientFactory.create_client(
            WeatherProvider.WEATHERAPI, 
            api_key=weatherapi_key
        )
        
        assert client is not None
        assert client.get_circuit_breaker_state() in ["CLOSED", "OPEN", "HALF_OPEN"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow  # Mark as slow test for optional execution
    async def test_weatherapi_current_weather(self, london_coordinates, weatherapi_key):
        """Test WeatherAPI current weather retrieval."""
        if not weatherapi_key:
            pytest.skip("WEATHERAPI_KEY environment variable not set")
        
        client = WeatherClientFactory.create_client(
            WeatherProvider.WEATHERAPI, 
            api_key=weatherapi_key
        )
        
        try:
            weather_data = await client.get_current_weather(
                london_coordinates, "London", "UK"
            )
            
            assert weather_data is not None
            assert len(weather_data) > 0
            
            point = weather_data[0]
            assert point.city == "London"
            assert point.provider == WeatherProvider.WEATHERAPI
            assert point.temperature is not None
            assert point.humidity is not None
            assert point.pressure is not None
            assert isinstance(point.timestamp, datetime)
            assert point.coordinates.latitude == london_coordinates.latitude
            assert point.coordinates.longitude == london_coordinates.longitude
            
        except Exception as e:
            pytest.fail(f"WeatherAPI integration test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_weatherapi_error_handling(self, london_coordinates):
        """Test WeatherAPI error handling with invalid key."""
        client = WeatherClientFactory.create_client(
            WeatherProvider.WEATHERAPI, 
            api_key="invalid_key"
        )
        
        with pytest.raises(Exception):  # Should raise an authentication error
            await client.get_current_weather(london_coordinates, "London", "UK")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_weatherapi_circuit_breaker_metrics(self, weatherapi_key):
        """Test WeatherAPI circuit breaker and rate limiting metrics."""
        if not weatherapi_key:
            pytest.skip("WEATHERAPI_KEY environment variable not set")
        
        client = WeatherClientFactory.create_client(
            WeatherProvider.WEATHERAPI, 
            api_key=weatherapi_key
        )
        
        metrics = client.get_metrics()
        
        assert "circuit_breaker_state" in metrics
        assert "available_tokens" in metrics
        assert metrics["circuit_breaker_state"] in ["CLOSED", "OPEN", "HALF_OPEN"]
        assert isinstance(metrics["available_tokens"], (int, float))
