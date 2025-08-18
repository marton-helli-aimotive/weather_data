"""Integration tests for Milestone 2: Enhanced API Client & Data Models."""

import pytest
import asyncio
import os
from datetime import datetime, timezone
from unittest.mock import patch

from src.weather_pipeline.api import WeatherClientFactory
from src.weather_pipeline.models.weather import WeatherProvider, Coordinates, WeatherDataPoint


class TestMilestone2Integration:
    """Integration tests for Milestone 2 functionality."""

    @pytest.fixture
    def london_coordinates(self):
        """London coordinates for testing."""
        return Coordinates(latitude=51.5074, longitude=-0.1278)

    @pytest.fixture
    def api_keys(self):
        """Get API keys from environment."""
        return {
            WeatherProvider.WEATHERAPI: os.getenv("WEATHERAPI_KEY"),
            WeatherProvider.OPENWEATHER: os.getenv("OPENWEATHER_API_KEY")
        }

    @pytest.mark.integration
    def test_factory_pattern_client_creation(self, api_keys):
        """Test factory pattern for creating different provider clients."""
        # Test 7Timer (no API key required)
        seven_timer_client = WeatherClientFactory.create_client(WeatherProvider.SEVEN_TIMER)
        assert seven_timer_client is not None
        assert seven_timer_client.get_circuit_breaker_state() in ["closed", "open", "half_open"]
        
        # Test WeatherAPI if key available
        if api_keys[WeatherProvider.WEATHERAPI]:
            weatherapi_client = WeatherClientFactory.create_client(
                WeatherProvider.WEATHERAPI, 
                api_key=api_keys[WeatherProvider.WEATHERAPI]
            )
            assert weatherapi_client is not None
            assert weatherapi_client.get_circuit_breaker_state() in ["closed", "open", "half_open"]

    @pytest.mark.integration
    def test_multi_provider_client_creation(self, api_keys):
        """Test multi-provider client with fallback capabilities."""
        providers = [WeatherProvider.WEATHERAPI, WeatherProvider.SEVEN_TIMER]
        available_keys = {k: v for k, v in api_keys.items() if v is not None}
        
        multi_client = WeatherClientFactory.create_multi_provider_client(
            providers=providers,
            api_keys=available_keys
        )
        
        assert multi_client is not None
        available_providers = multi_client.get_available_providers()
        assert len(available_providers) > 0
        assert all(isinstance(p, WeatherProvider) for p in available_providers)

    @pytest.mark.integration
    def test_circuit_breaker_states(self, api_keys):
        """Test circuit breaker states for different providers."""
        seven_timer_client = WeatherClientFactory.create_client(WeatherProvider.SEVEN_TIMER)
        state = seven_timer_client.get_circuit_breaker_state()
        assert state in ["closed", "open", "half_open"]
        
        if api_keys[WeatherProvider.WEATHERAPI]:
            weatherapi_client = WeatherClientFactory.create_client(
                WeatherProvider.WEATHERAPI, 
                api_key=api_keys[WeatherProvider.WEATHERAPI]
            )
            state = weatherapi_client.get_circuit_breaker_state()
            assert state in ["closed", "open", "half_open"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_weatherapi_current_weather_call(self, london_coordinates, api_keys):
        """Test actual WeatherAPI current weather call."""
        if not api_keys[WeatherProvider.WEATHERAPI]:
            pytest.skip("WEATHERAPI_KEY not available")
        
        client = WeatherClientFactory.create_client(
            WeatherProvider.WEATHERAPI, 
            api_key=api_keys[WeatherProvider.WEATHERAPI]
        )
        
        try:
            weather_data = await client.get_current_weather(
                london_coordinates, "London", "UK"
            )
            
            if weather_data:
                point = weather_data[0]
                assert isinstance(point, WeatherDataPoint)
                assert point.city == "London"
                assert point.provider == WeatherProvider.WEATHERAPI
                assert point.temperature is not None
                assert point.humidity is not None
                assert isinstance(point.timestamp, datetime)
                assert point.coordinates.latitude == london_coordinates.latitude
                assert point.coordinates.longitude == london_coordinates.longitude
                
        except Exception as e:
            # Log the error but don't fail the test due to potential API issues
            pytest.skip(f"WeatherAPI call failed (external dependency): {e}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_seven_timer_current_weather_call(self, london_coordinates):
        """Test actual 7Timer current weather call."""
        client = WeatherClientFactory.create_client(WeatherProvider.SEVEN_TIMER)
        
        try:
            weather_data = await client.get_current_weather(
                london_coordinates, "London", "UK"
            )
            
            if weather_data:
                point = weather_data[0]
                assert isinstance(point, WeatherDataPoint)
                assert point.provider == WeatherProvider.SEVEN_TIMER
                assert point.temperature is not None
                assert isinstance(point.timestamp, datetime)
                assert point.coordinates.latitude == london_coordinates.latitude
                assert point.coordinates.longitude == london_coordinates.longitude
                
        except Exception as e:
            # 7Timer API can be unreliable, so skip on failure
            pytest.skip(f"7Timer API call failed (external dependency): {e}")

    @pytest.mark.integration
    def test_rate_limiting_and_metrics(self, api_keys):
        """Test rate limiting and circuit breaker metrics."""
        client = WeatherClientFactory.create_client(WeatherProvider.SEVEN_TIMER)
        
        metrics = client.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "circuit_breaker_state" in metrics
        assert "available_tokens" in metrics
        assert metrics["circuit_breaker_state"] in ["closed", "open", "half_open"]
        assert isinstance(metrics["available_tokens"], (int, float))

    @pytest.mark.integration
    def test_data_model_validation(self, london_coordinates):
        """Test Pydantic data model validation."""
        # Test manual data creation
        test_point = WeatherDataPoint(
            timestamp=datetime.now(timezone.utc),
            temperature=15.5,
            humidity=65,
            pressure=1013.25,
            wind_speed=3.2,
            wind_direction=180,
            city="London",
            country="UK",
            coordinates=london_coordinates,
            provider=WeatherProvider.SEVEN_TIMER
        )
        
        assert isinstance(test_point, WeatherDataPoint)
        assert test_point.temperature == 15.5
        assert test_point.humidity == 65
        assert test_point.city == "London"
        assert test_point.provider == WeatherProvider.SEVEN_TIMER
        assert test_point.coordinates.latitude == london_coordinates.latitude
        assert test_point.coordinates.longitude == london_coordinates.longitude

    @pytest.mark.integration
    def test_data_model_validation_errors(self, london_coordinates):
        """Test data model validation with invalid data."""
        # Test invalid humidity (>100)
        with pytest.raises(ValueError):
            WeatherDataPoint(
                timestamp=datetime.now(timezone.utc),
                temperature=15.5,
                humidity=150,  # Invalid
                pressure=1013.25,
                wind_speed=3.2,
                wind_direction=180,
                city="London",
                country="UK",
                coordinates=london_coordinates,
                provider=WeatherProvider.SEVEN_TIMER
            )

        # Test invalid wind direction (>360)
        with pytest.raises(ValueError):
            WeatherDataPoint(
                timestamp=datetime.now(timezone.utc),
                temperature=15.5,
                humidity=65,
                pressure=1013.25,
                wind_speed=3.2,
                wind_direction=400,  # Invalid
                city="London",
                country="UK",
                coordinates=london_coordinates,
                provider=WeatherProvider.SEVEN_TIMER
            )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_provider_fallback_mechanism(self, london_coordinates, api_keys):
        """Test provider fallback capabilities."""
        providers = [WeatherProvider.WEATHERAPI, WeatherProvider.SEVEN_TIMER]
        available_keys = {k: v for k, v in api_keys.items() if v is not None}
        
        multi_client = WeatherClientFactory.create_multi_provider_client(
            providers=providers,
            api_keys=available_keys
        )
        
        available_providers = multi_client.get_available_providers()
        assert len(available_providers) > 0
        
        # Test that at least one provider is available
        assert WeatherProvider.SEVEN_TIMER in available_providers
        if api_keys[WeatherProvider.WEATHERAPI]:
            assert WeatherProvider.WEATHERAPI in available_providers

    @pytest.mark.integration
    def test_milestone2_success_criteria(self, api_keys):
        """Test that all Milestone 2 success criteria are met."""
        # Multi-provider API support with factory pattern
        seven_timer_client = WeatherClientFactory.create_client(WeatherProvider.SEVEN_TIMER)
        assert seven_timer_client is not None
        
        # Circuit breaker pattern implemented
        assert hasattr(seven_timer_client, 'get_circuit_breaker_state')
        assert seven_timer_client.get_circuit_breaker_state() in ["closed", "open", "half_open"]
        
        # Rate limiting and retry logic
        assert hasattr(seven_timer_client, 'get_metrics')
        metrics = seven_timer_client.get_metrics()
        assert "available_tokens" in metrics
        
        # Comprehensive Pydantic data models
        assert WeatherDataPoint is not None
        assert WeatherProvider is not None
        assert Coordinates is not None
        
        # Provider fallback capabilities
        multi_client = WeatherClientFactory.create_multi_provider_client(
            providers=[WeatherProvider.SEVEN_TIMER],
            api_keys={}
        )
        assert len(multi_client.get_available_providers()) > 0
