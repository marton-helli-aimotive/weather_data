"""Basic tests to verify the foundation is working."""

import pytest
from pathlib import Path
from weather_pipeline.config import get_settings, Settings
from weather_pipeline.core import get_container, DIContainer
from weather_pipeline.models import WeatherProvider, Coordinates


def test_settings_loading():
    """Test that settings can be loaded."""
    get_settings.cache_clear()  # Clear any cached settings first
    settings = get_settings()
    assert isinstance(settings, Settings)
    # App name might be overridden by environment variables
    assert isinstance(settings.app_name, str)
    assert len(settings.app_name) > 0
    assert settings.app_version == "0.1.0"


def test_dependency_injection_container():
    """Test basic dependency injection functionality."""
    container = get_container()
    assert isinstance(container, DIContainer)
    
    # Test singleton registration
    test_value = "test_singleton"
    container.register_singleton(str, test_value)
    
    # Test retrieval
    retrieved = container.get(str)
    assert retrieved == test_value
    
    # Clean up
    container.clear()


def test_weather_provider_enum():
    """Test WeatherProvider enum."""
    assert WeatherProvider.SEVEN_TIMER == "7timer"
    assert WeatherProvider.OPENWEATHER == "openweather"
    assert WeatherProvider.WEATHERAPI == "weatherapi"


def test_coordinates_validation():
    """Test Coordinates model validation."""
    # Valid coordinates
    coords = Coordinates(latitude=47.4979, longitude=19.0402)
    assert coords.latitude == 47.4979
    assert coords.longitude == 19.0402
    
    # Test validation - invalid latitude
    with pytest.raises(ValueError):
        Coordinates(latitude=91.0, longitude=0.0)
    
    # Test validation - invalid longitude  
    with pytest.raises(ValueError):
        Coordinates(latitude=0.0, longitude=181.0)
