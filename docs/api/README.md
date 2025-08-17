# Weather Pipeline API Documentation

## Overview

The Weather Pipeline provides a comprehensive Python API for collecting, processing, and analyzing weather data from multiple providers. This documentation covers all public APIs, data models, and usage examples.

## Table of Contents

- [Quick Start](#quick-start)
- [API Clients](#api-clients)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
import asyncio
from weather_pipeline.api import WeatherAPIClient
from weather_pipeline.models import Coordinates

async def get_weather():
    client = WeatherAPIClient(api_key="your_api_key")
    
    location = Coordinates(latitude=40.7128, longitude=-74.0060)  # New York
    weather_data = await client.get_current_weather(location)
    
    print(f"Temperature: {weather_data.temperature}°C")
    print(f"Humidity: {weather_data.humidity}%")

# Run the async function
asyncio.run(get_weather())
```

## API Clients

### WeatherAPIClient

Primary client for WeatherAPI.com integration.

```python
from weather_pipeline.api import WeatherAPIClient

client = WeatherAPIClient(
    api_key="your_api_key",
    timeout=30,
    rate_limit_calls=100,
    rate_limit_period=60
)
```

#### Methods

##### `get_current_weather(location: Coordinates) -> WeatherDataPoint`

Retrieves current weather conditions for a specific location.

**Parameters:**
- `location` (Coordinates): Geographic coordinates

**Returns:**
- `WeatherDataPoint`: Current weather data

**Example:**
```python
location = Coordinates(latitude=51.5074, longitude=-0.1278)  # London
current_weather = await client.get_current_weather(location)
```

##### `get_forecast(location: Coordinates, days: int = 5) -> List[WeatherDataPoint]`

Retrieves weather forecast for multiple days.

**Parameters:**
- `location` (Coordinates): Geographic coordinates
- `days` (int): Number of forecast days (1-10)

**Returns:**
- `List[WeatherDataPoint]`: List of forecast data points

**Example:**
```python
forecast = await client.get_forecast(location, days=7)
for day in forecast:
    print(f"Date: {day.timestamp}, Temp: {day.temperature}°C")
```

### MultiProviderClient

Intelligent client that uses multiple weather providers with fallback support.

```python
from weather_pipeline.api import MultiProviderClient

client = MultiProviderClient(
    primary_provider="weatherapi",
    fallback_providers=["7timer", "openweather"],
    circuit_breaker_threshold=5
)
```

#### Methods

##### `get_weather_data(location: Coordinates, prefer_provider: str = None) -> WeatherDataPoint`

Gets weather data with automatic provider fallback.

**Parameters:**
- `location` (Coordinates): Geographic coordinates
- `prefer_provider` (str, optional): Preferred provider name

**Returns:**
- `WeatherDataPoint`: Weather data from available provider

**Example:**
```python
# Will try WeatherAPI first, fallback to 7Timer if needed
weather_data = await client.get_weather_data(location)

# Force specific provider
weather_data = await client.get_weather_data(location, prefer_provider="7timer")
```

### SevenTimerClient

Client for 7Timer.info weather service (no API key required).

```python
from weather_pipeline.api import SevenTimerClient

client = SevenTimerClient(timeout=30)
```

### OpenWeatherMapClient

Client for OpenWeatherMap API service.

```python
from weather_pipeline.api import OpenWeatherMapClient

client = OpenWeatherMapClient(
    api_key="your_openweather_key",
    units="metric"  # metric, imperial, or kelvin
)
```

## Data Models

### WeatherDataPoint

Core data structure representing weather conditions at a specific time and location.

```python
from weather_pipeline.models import WeatherDataPoint
from datetime import datetime

weather_point = WeatherDataPoint(
    timestamp=datetime.now(),
    temperature=22.5,
    humidity=65,
    pressure=1013.25,
    wind_speed=5.2,
    wind_direction=180,
    precipitation=0.0,
    visibility=10000,
    uv_index=5,
    condition="Partly Cloudy",
    city="London",
    country="UK",
    coordinates=Coordinates(latitude=51.5074, longitude=-0.1278),
    data_source="weatherapi"
)
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | When the data was recorded |
| `temperature` | float | Temperature in Celsius |
| `humidity` | int | Relative humidity percentage (0-100) |
| `pressure` | float | Atmospheric pressure in hPa |
| `wind_speed` | float | Wind speed in m/s |
| `wind_direction` | int | Wind direction in degrees (0-360) |
| `precipitation` | float | Precipitation amount in mm |
| `visibility` | int | Visibility distance in meters |
| `uv_index` | float | UV index value |
| `condition` | str | Weather condition description |
| `city` | str | City name |
| `country` | str | Country code |
| `coordinates` | Coordinates | Geographic location |
| `data_source` | str | Source provider name |

### Coordinates

Geographic coordinate representation.

```python
from weather_pipeline.models import Coordinates

location = Coordinates(
    latitude=40.7128,
    longitude=-74.0060
)
```

### LocationInfo

Extended location information with geocoding data.

```python
from weather_pipeline.models import LocationInfo

location_info = LocationInfo(
    city="New York",
    country="US",
    timezone="America/New_York",
    coordinates=Coordinates(latitude=40.7128, longitude=-74.0060)
)
```

### DataQualityMetrics

Quality assessment metrics for weather data.

```python
from weather_pipeline.models import DataQualityMetrics

quality_metrics = DataQualityMetrics(
    completeness_score=0.95,
    accuracy_score=0.98,
    timeliness_score=0.90,
    consistency_score=0.97,
    missing_fields=["wind_direction"],
    outlier_count=2,
    data_age_minutes=5
)
```

### WeatherAlert

Weather alert and warning information.

```python
from weather_pipeline.models import WeatherAlert

alert = WeatherAlert(
    alert_type="warning",
    severity="moderate",
    title="Heavy Rain Warning",
    description="Heavy rainfall expected in the next 6 hours",
    start_time=datetime.now(),
    end_time=datetime.now() + timedelta(hours=6),
    affected_areas=["London", "Birmingham"]
)
```

## Error Handling

### Exception Types

#### APIError

Base exception for all API-related errors.

```python
from weather_pipeline.api.exceptions import APIError

try:
    weather_data = await client.get_current_weather(location)
except APIError as e:
    print(f"API Error: {e.message}")
    print(f"Error Code: {e.error_code}")
    print(f"Provider: {e.provider}")
```

#### RateLimitError

Raised when API rate limits are exceeded.

```python
from weather_pipeline.api.exceptions import RateLimitError

try:
    weather_data = await client.get_current_weather(location)
except RateLimitError as e:
    print(f"Rate limit exceeded. Retry after: {e.retry_after} seconds")
```

#### AuthenticationError

Raised for invalid API keys or authentication issues.

```python
from weather_pipeline.api.exceptions import AuthenticationError

try:
    weather_data = await client.get_current_weather(location)
except AuthenticationError as e:
    print(f"Authentication failed: {e.message}")
```

#### ValidationError

Raised for invalid input parameters.

```python
from weather_pipeline.api.exceptions import ValidationError

try:
    # Invalid coordinates
    location = Coordinates(latitude=200, longitude=400)
    weather_data = await client.get_current_weather(location)
except ValidationError as e:
    print(f"Validation error: {e.message}")
```

### Error Handling Best Practices

```python
import asyncio
from weather_pipeline.api import MultiProviderClient
from weather_pipeline.api.exceptions import APIError, RateLimitError
from weather_pipeline.models import Coordinates

async def robust_weather_fetch(location: Coordinates):
    client = MultiProviderClient()
    
    try:
        # Primary attempt
        weather_data = await client.get_weather_data(location)
        return weather_data
        
    except RateLimitError as e:
        # Wait and retry
        print(f"Rate limited, waiting {e.retry_after} seconds...")
        await asyncio.sleep(e.retry_after)
        return await client.get_weather_data(location)
        
    except APIError as e:
        # Log error and try different provider
        print(f"Provider {e.provider} failed: {e.message}")
        return await client.get_weather_data(location, prefer_provider="7timer")
        
    except Exception as e:
        # Handle unexpected errors
        print(f"Unexpected error: {e}")
        return None
```

## Best Practices

### 1. Async Context Management

Always use async context managers for proper resource cleanup:

```python
from weather_pipeline.api import WeatherAPIClient

async def get_weather_data():
    async with WeatherAPIClient(api_key="your_key") as client:
        location = Coordinates(latitude=40.7128, longitude=-74.0060)
        weather_data = await client.get_current_weather(location)
        return weather_data
```

### 2. Rate Limiting

Respect API rate limits to avoid service disruption:

```python
client = WeatherAPIClient(
    api_key="your_key",
    rate_limit_calls=100,  # Max calls
    rate_limit_period=60   # Per 60 seconds
)
```

### 3. Caching

Use caching to reduce API calls and improve performance:

```python
from weather_pipeline.core import CacheManager

cache_manager = CacheManager(backend="redis")
client = WeatherAPIClient(api_key="your_key", cache_manager=cache_manager)
```

### 4. Circuit Breaker Pattern

Use circuit breakers for resilient API calls:

```python
client = MultiProviderClient(
    circuit_breaker_threshold=5,  # Open after 5 failures
    circuit_breaker_recovery_timeout=60  # Retry after 60 seconds
)
```

### 5. Data Validation

Always validate location coordinates:

```python
from weather_pipeline.models import Coordinates

def validate_coordinates(lat: float, lon: float) -> Coordinates:
    if not (-90 <= lat <= 90):
        raise ValueError(f"Invalid latitude: {lat}")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Invalid longitude: {lon}")
    
    return Coordinates(latitude=lat, longitude=lon)
```

## Examples

### Example 1: Basic Weather Fetching

```python
import asyncio
from weather_pipeline.api import WeatherAPIClient
from weather_pipeline.models import Coordinates

async def basic_weather_example():
    client = WeatherAPIClient(api_key="your_api_key")
    
    # New York coordinates
    location = Coordinates(latitude=40.7128, longitude=-74.0060)
    
    # Get current weather
    current_weather = await client.get_current_weather(location)
    
    print(f"Current weather in {current_weather.city}:")
    print(f"Temperature: {current_weather.temperature}°C")
    print(f"Condition: {current_weather.condition}")
    print(f"Humidity: {current_weather.humidity}%")
    print(f"Wind: {current_weather.wind_speed} m/s")

if __name__ == "__main__":
    asyncio.run(basic_weather_example())
```

### Example 2: Multi-Location Comparison

```python
import asyncio
from weather_pipeline.api import MultiProviderClient
from weather_pipeline.models import Coordinates

async def compare_cities():
    client = MultiProviderClient()
    
    cities = {
        "New York": Coordinates(latitude=40.7128, longitude=-74.0060),
        "London": Coordinates(latitude=51.5074, longitude=-0.1278),
        "Tokyo": Coordinates(latitude=35.6762, longitude=139.6503),
        "Sydney": Coordinates(latitude=-33.8688, longitude=151.2093),
    }
    
    weather_data = {}
    
    # Fetch weather for all cities concurrently
    tasks = []
    for city_name, location in cities.items():
        task = client.get_weather_data(location)
        tasks.append((city_name, task))
    
    # Wait for all requests to complete
    results = await asyncio.gather(*[task for _, task in tasks])
    
    # Process results
    for (city_name, _), weather in zip(tasks, results):
        weather_data[city_name] = weather
    
    # Display comparison
    print("City Weather Comparison:")
    print("-" * 50)
    for city, weather in weather_data.items():
        if weather:
            print(f"{city:10} | {weather.temperature:5.1f}°C | {weather.condition}")

if __name__ == "__main__":
    asyncio.run(compare_cities())
```

### Example 3: Weather Data Processing

```python
import asyncio
from datetime import datetime, timedelta
from weather_pipeline.api import WeatherAPIClient
from weather_pipeline.models import Coordinates
from weather_pipeline.processing import WeatherDataProcessor

async def process_weather_data():
    client = WeatherAPIClient(api_key="your_api_key")
    processor = WeatherDataProcessor()
    
    location = Coordinates(latitude=40.7128, longitude=-74.0060)
    
    # Collect weather data over time
    weather_points = []
    for i in range(24):  # 24 hours of data
        weather_data = await client.get_current_weather(location)
        weather_points.append(weather_data)
        await asyncio.sleep(3600)  # Wait 1 hour (for demo, use smaller interval)
    
    # Process the collected data
    processed_data = processor.process_time_series(weather_points)
    
    print("Weather Analysis:")
    print(f"Average Temperature: {processed_data.avg_temperature:.1f}°C")
    print(f"Temperature Range: {processed_data.min_temperature:.1f}°C to {processed_data.max_temperature:.1f}°C")
    print(f"Average Humidity: {processed_data.avg_humidity:.1f}%")
    print(f"Precipitation Total: {processed_data.total_precipitation:.1f}mm")

if __name__ == "__main__":
    asyncio.run(process_weather_data())
```

### Example 4: Error Handling and Resilience

```python
import asyncio
from weather_pipeline.api import MultiProviderClient
from weather_pipeline.api.exceptions import APIError, RateLimitError
from weather_pipeline.models import Coordinates

async def resilient_weather_fetch():
    client = MultiProviderClient(
        circuit_breaker_threshold=3,
        circuit_breaker_recovery_timeout=30
    )
    
    location = Coordinates(latitude=40.7128, longitude=-74.0060)
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            weather_data = await client.get_weather_data(location)
            print(f"Successfully fetched weather data:")
            print(f"Temperature: {weather_data.temperature}°C")
            print(f"Source: {weather_data.data_source}")
            return weather_data
            
        except RateLimitError as e:
            print(f"Rate limited, waiting {e.retry_after} seconds...")
            await asyncio.sleep(e.retry_after)
            retry_count += 1
            
        except APIError as e:
            print(f"API error from {e.provider}: {e.message}")
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    
    print("Failed to fetch weather data after all retries")
    return None

if __name__ == "__main__":
    asyncio.run(resilient_weather_fetch())
```

## Configuration

### Environment Variables

```bash
# API Keys
WEATHER_API_WEATHERAPI_KEY=your_weatherapi_key
WEATHER_API_OPENWEATHER_API_KEY=your_openweather_key

# Rate Limiting
WEATHER_API_RATE_LIMIT_CALLS=100
WEATHER_API_RATE_LIMIT_PERIOD=60

# Timeouts
WEATHER_API_TIMEOUT=30
WEATHER_API_CONNECT_TIMEOUT=10

# Circuit Breaker
WEATHER_API_CIRCUIT_BREAKER_THRESHOLD=5
WEATHER_API_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60

# Caching
WEATHER_CACHE_TTL=300
WEATHER_CACHE_BACKEND=redis
```

### Configuration File

```python
from weather_pipeline.config import WeatherConfig

config = WeatherConfig(
    weatherapi_key="your_weatherapi_key",
    openweather_key="your_openweather_key",
    default_timeout=30,
    rate_limit_calls=100,
    rate_limit_period=60,
    cache_ttl=300
)
```

## Testing

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock, patch
from weather_pipeline.api import WeatherAPIClient
from weather_pipeline.models import Coordinates

@pytest.mark.asyncio
async def test_weather_api_client():
    with patch('aiohttp.ClientSession.get') as mock_get:
        # Mock API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "current": {
                "temp_c": 22.5,
                "humidity": 65,
                "condition": {"text": "Partly cloudy"}
            }
        }
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Test client
        client = WeatherAPIClient(api_key="test_key")
        location = Coordinates(latitude=40.7128, longitude=-74.0060)
        
        weather_data = await client.get_current_weather(location)
        
        assert weather_data.temperature == 22.5
        assert weather_data.humidity == 65
        assert weather_data.condition == "Partly cloudy"
```

### Integration Testing

```python
import pytest
from weather_pipeline.api import WeatherAPIClient
from weather_pipeline.models import Coordinates

@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_api_integration():
    """Test with real API - requires valid API key"""
    api_key = os.getenv("WEATHER_API_WEATHERAPI_KEY")
    if not api_key:
        pytest.skip("API key not available")
    
    client = WeatherAPIClient(api_key=api_key)
    location = Coordinates(latitude=40.7128, longitude=-74.0060)
    
    weather_data = await client.get_current_weather(location)
    
    assert weather_data is not None
    assert isinstance(weather_data.temperature, (int, float))
    assert 0 <= weather_data.humidity <= 100
```

## Support

For issues, questions, or contributions:

1. Check the [GitHub Issues](https://github.com/your-org/weather-pipeline/issues)
2. Review the [troubleshooting guide](../user-guides/troubleshooting.md)
3. Consult the [FAQ](../user-guides/faq.md)

## Version History

- **v1.0.0**: Initial API release with WeatherAPI.com support
- **v1.1.0**: Added multi-provider support and circuit breaker pattern
- **v1.2.0**: Added OpenWeatherMap and 7Timer support
- **v1.3.0**: Enhanced error handling and caching
- **v1.4.0**: Added async context managers and improved rate limiting
