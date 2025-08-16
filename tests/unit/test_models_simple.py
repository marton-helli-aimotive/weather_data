"""Simplified unit tests for models."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from weather_pipeline.models.weather import WeatherDataPoint, Coordinates, WeatherProvider


class TestWeatherDataPoint:
    """Test WeatherDataPoint model."""

    def test_weather_data_creation(self):
        """Test creating a WeatherDataPoint instance."""
        data = WeatherDataPoint(
            timestamp=datetime.now(timezone.utc),
            temperature=25.0,
            humidity=65,
            pressure=1013.25,
            wind_speed=5.0,
            wind_direction=180,
            city="TestCity",
            coordinates=Coordinates(latitude=50.0, longitude=10.0),
            provider=WeatherProvider.WEATHERAPI
        )
        
        assert data.temperature == 25.0
        assert data.humidity == 65
        assert data.city == "TestCity"
        assert data.coordinates.latitude == 50.0
        assert data.provider == WeatherProvider.WEATHERAPI

    def test_coordinates_validation(self):
        """Test Coordinates validation."""
        # Valid coordinates
        coords = Coordinates(latitude=50.0, longitude=10.0)
        assert coords.latitude == 50.0
        assert coords.longitude == 10.0
        
        # Invalid latitude
        with pytest.raises(ValueError):
            Coordinates(latitude=95.0, longitude=10.0)  # > 90
            
        # Invalid longitude
        with pytest.raises(ValueError):
            Coordinates(latitude=50.0, longitude=185.0)  # > 180

    def test_weather_provider_enum(self):
        """Test WeatherProvider enum values."""
        assert WeatherProvider.SEVEN_TIMER == "7timer"
        assert WeatherProvider.OPENWEATHER == "openweather"
        assert WeatherProvider.WEATHERAPI == "weatherapi"

    def test_weather_data_to_dict(self):
        """Test converting WeatherDataPoint to dictionary."""
        data = WeatherDataPoint(
            timestamp=datetime.now(timezone.utc),
            temperature=25.0,
            humidity=65,
            pressure=1013.25,
            wind_speed=5.0,
            wind_direction=180,
            city="TestCity",
            coordinates=Coordinates(latitude=50.0, longitude=10.0),
            provider=WeatherProvider.WEATHERAPI
        )
        
        data_dict = data.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["temperature"] == 25.0
        assert data_dict["city"] == "TestCity"
        assert data_dict["provider"] == "weatherapi"
