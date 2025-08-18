"""Test configuration and fixtures."""

import asyncio
import pytest
import pytest_asyncio
from pathlib import Path
from typing import Dict, Any, List, AsyncGenerator
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Import from our application
from src.weather_pipeline.config import get_settings, Settings
from src.weather_pipeline.core import get_container
from src.weather_pipeline.models import (
    WeatherProvider, Coordinates, WeatherDataPoint,
    WeatherAPIResponse, OpenWeatherMapResponse, SevenTimerResponse
)
from src.weather_pipeline.api import (
    WeatherAPIClient, OpenWeatherMapClient, SevenTimerClient,
    WeatherClientFactory
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory for testing."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_coordinates():
    """Sample coordinates for testing."""
    return {
        "budapest": {"latitude": 47.4979, "longitude": 19.0402},
        "london": {"latitude": 51.5074, "longitude": -0.1278},
        "new_york": {"latitude": 40.7128, "longitude": -74.0060},
        "tokyo": {"latitude": 35.6762, "longitude": 139.6503},
        "invalid_lat": {"latitude": 91.0, "longitude": 0.0},
        "invalid_lon": {"latitude": 0.0, "longitude": 181.0},
    }


@pytest.fixture
def test_settings() -> Settings:
    """Test settings with mocked values."""
    with patch.dict('os.environ', {
        'WEATHER_API_WEATHERAPI_KEY': 'test_weatherapi_key',
        'WEATHER_API_OPENWEATHER_API_KEY': 'test_openweather_key',
        'DB_HOST': 'localhost',
        'LOG_LEVEL': 'DEBUG'
    }):
        return get_settings()


@pytest.fixture
def mock_weather_data() -> List[Dict[str, Any]]:
    """Mock weather data for testing."""
    return [
        {
            "timestamp": datetime.now(timezone.utc),
            "temperature": 20.5,
            "humidity": 65,
            "pressure": 1013.25,
            "wind_speed": 5.2,
            "wind_direction": 180,
            "precipitation": 0.0,
            "visibility": 10.0,
            "cloud_cover": 25,
            "uv_index": 5.0,
            "city": "London",
            "country": "UK",
            "coordinates": {"latitude": 51.5074, "longitude": -0.1278},
            "provider": "weatherapi",
            "is_forecast": True,
            "confidence_score": 0.95
        },
        {
            "timestamp": datetime.now(timezone.utc) + timedelta(hours=1),
            "temperature": 21.0,
            "humidity": 63,
            "pressure": 1013.00,
            "wind_speed": 5.5,
            "wind_direction": 185,
            "precipitation": 0.2,
            "visibility": 9.8,
            "cloud_cover": 30,
            "uv_index": 5.2,
            "city": "London",
            "country": "UK",
            "coordinates": {"latitude": 51.5074, "longitude": -0.1278},
            "provider": "weatherapi",
            "is_forecast": True,
            "confidence_score": 0.93
        }
    ]


@pytest.fixture
def sample_weather_dataframe() -> pd.DataFrame:
    """Sample weather data as pandas DataFrame."""
    np.random.seed(42)
    n_records = 100
    
    timestamps = pd.date_range('2024-01-01', periods=n_records, freq='H')
    cities = ['New York', 'London', 'Tokyo']
    
    data = {
        'timestamp': timestamps,
        'city': np.random.choice(cities, n_records),
        'temperature': np.random.normal(20, 5, n_records),
        'humidity': np.random.randint(30, 90, n_records),
        'pressure': np.random.normal(1013, 20, n_records),
        'wind_speed': np.random.exponential(5, n_records),
        'wind_direction': np.random.randint(0, 360, n_records),
        'precipitation': np.random.exponential(0.5, n_records),
        'latitude': np.random.uniform(40, 45, n_records),
        'longitude': np.random.uniform(-75, -70, n_records)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_api_responses():
    """Mock API response data for different providers."""
    return {
        "weatherapi": {
            "location": {
                "name": "London",
                "region": "City of London, Greater London",
                "country": "United Kingdom",
                "lat": 51.52,
                "lon": -0.11,
                "tz_id": "Europe/London",
                "localtime": "2024-01-01 12:00"
            },
            "current": {
                "temp_c": 20.5,
                "temp_f": 68.9,
                "condition": {
                    "text": "Partly cloudy",
                    "icon": "//cdn.weatherapi.com/weather/64x64/day/116.png",
                    "code": 1003
                },
                "wind_mph": 3.2,
                "wind_kph": 5.2,
                "wind_degree": 180,
                "wind_dir": "S",
                "pressure_mb": 1013.0,
                "pressure_in": 29.91,
                "precip_mm": 0.0,
                "precip_in": 0.0,
                "humidity": 65,
                "cloud": 25,
                "feelslike_c": 20.5,
                "feelslike_f": 68.9,
                "vis_km": 10.0,
                "vis_miles": 6.2,
                "uv": 5.0,
                "gust_mph": 4.5,
                "gust_kph": 7.2
            }
        },
        "openweather": {
            "coord": {"lon": -0.1278, "lat": 51.5074},
            "weather": [
                {
                    "id": 803,
                    "main": "Clouds",
                    "description": "broken clouds",
                    "icon": "04d"
                }
            ],
            "base": "stations",
            "main": {
                "temp": 293.65,
                "feels_like": 293.65,
                "temp_min": 292.04,
                "temp_max": 295.37,
                "pressure": 1013,
                "humidity": 65
            },
            "visibility": 10000,
            "wind": {
                "speed": 5.2,
                "deg": 180
            },
            "clouds": {
                "all": 75
            },
            "dt": 1704110400,
            "sys": {
                "type": 2,
                "id": 2019646,
                "country": "GB",
                "sunrise": 1704097567,
                "sunset": 1704123421
            },
            "timezone": 0,
            "id": 2643743,
            "name": "London",
            "cod": 200
        },
        "7timer": {
            "product": "civil",
            "init": "2024010112",
            "dataseries": [
                {
                    "timepoint": 3,
                    "cloudcover": 3,
                    "seeing": 5,
                    "transparency": 5,
                    "lifted_index": 10,
                    "rh2m": 65,
                    "wind10m_direction": "S",
                    "wind10m_speed": 2,
                    "temp2m": 20,
                    "prec_type": "none"
                }
            ]
        }
    }


@pytest.fixture
async def mock_weather_clients(test_settings) -> Dict[str, Mock]:
    """Mock weather client instances."""
    mock_clients = {}
    
    for provider in ["weatherapi", "openweather", "7timer"]:
        mock_client = AsyncMock()
        mock_client.provider = WeatherProvider(provider)
        mock_client.get_current_weather = AsyncMock()
        mock_client.get_forecast = AsyncMock()
        mock_client.health_check = AsyncMock(return_value=True)
        mock_clients[provider] = mock_client
    
    return mock_clients


@pytest.fixture
def mock_redis():
    """Mock Redis client for caching tests."""
    mock_redis = Mock()
    mock_redis.get = Mock(return_value=None)
    mock_redis.set = Mock(return_value=True)
    mock_redis.delete = Mock(return_value=1)
    mock_redis.exists = Mock(return_value=False)
    mock_redis.expire = Mock(return_value=True)
    return mock_redis


@pytest.fixture(autouse=True)
def clean_container():
    """Clean the DI container before each test."""
    container = get_container()
    container.clear()
    yield
    container.clear()


@pytest.fixture
def benchmark_data() -> pd.DataFrame:
    """Large dataset for performance benchmarking."""
    np.random.seed(42)
    n_records = 10000
    
    timestamps = pd.date_range('2024-01-01', periods=n_records, freq='min')
    cities = ['New York', 'London', 'Tokyo', 'Paris', 'Berlin', 'Sydney']
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'city': np.random.choice(cities, n_records),
        'temperature': np.random.normal(20, 10, n_records),
        'humidity': np.random.randint(0, 100, n_records),
        'pressure': np.random.normal(1013, 50, n_records),
        'wind_speed': np.random.exponential(8, n_records),
        'wind_direction': np.random.randint(0, 360, n_records),
        'precipitation': np.random.exponential(1, n_records),
        'latitude': np.random.uniform(-90, 90, n_records),
        'longitude': np.random.uniform(-180, 180, n_records)
    })
