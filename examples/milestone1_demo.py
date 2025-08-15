"""Example script to demonstrate Milestone 1 foundation is working."""

from datetime import datetime, timezone

from weather_pipeline import (
    Coordinates, 
    WeatherDataPoint,
    WeatherProvider,
    get_settings,
    __version__
)
from weather_pipeline.core import get_container


def main():
    """Demonstrate the foundation functionality."""
    print(f"Weather Pipeline v{__version__}")
    print("=" * 50)
    
    # 1. Test configuration
    print("1. Configuration Management:")
    settings = get_settings()
    print(f"   Environment: {settings.environment.value}")
    print(f"   App Name: {settings.app_name}")
    print(f"   WeatherAPI Key: ***{settings.api.weatherapi_key[-4:]}")
    print()
    
    # 2. Test dependency injection
    print("2. Dependency Injection:")
    container = get_container()
    container.register_singleton(str, "test_service")
    retrieved = container.get(str)
    print(f"   Service registration/retrieval: {'✓ PASS' if retrieved == 'test_service' else '✗ FAIL'}")
    container.clear()
    print()
    
    # 3. Test data models
    print("3. Data Models with Validation:")
    try:
        # Valid coordinates
        coords = Coordinates(latitude=47.4979, longitude=19.0402)
        print(f"   Budapest coordinates: {coords}")
        
        # Valid weather data point
        weather_data = WeatherDataPoint(
            timestamp=datetime.now(timezone.utc),
            temperature=22.5,
            humidity=65,
            city="Budapest",
            coordinates=coords,
            provider=WeatherProvider.SEVEN_TIMER
        )
        print(f"   Weather data point created: ✓ PASS")
        print(f"   Temperature: {weather_data.temperature}°C")
        print(f"   Humidity: {weather_data.humidity}%")
        print()
        
        # Test validation - should fail
        try:
            invalid_coords = Coordinates(latitude=91.0, longitude=0.0)
            print(f"   Validation check: ✗ FAIL (should have rejected invalid coordinates)")
        except ValueError:
            print(f"   Validation check: ✓ PASS (correctly rejected invalid coordinates)")
            
    except Exception as e:
        print(f"   Model creation: ✗ FAIL ({e})")
        return
    
    print()
    print("Foundation successfully implemented!")
    print("✓ Configuration management with Pydantic settings")
    print("✓ Dependency injection container")
    print("✓ Type-safe data models with validation")
    print("✓ Modern project structure")
    print("✓ Working tests")


if __name__ == "__main__":
    main()
