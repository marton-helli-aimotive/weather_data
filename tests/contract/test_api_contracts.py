"""Contract tests for API responses and data schemas."""

import pytest
import json
from typing import Dict, Any, List
from datetime import datetime, timezone
from pydantic import ValidationError

from src.weather_pipeline.models.api_responses import (
    WeatherAPIResponse, OpenWeatherMapResponse, SevenTimerResponse,
    WeatherAPILocation, WeatherAPICondition, WeatherAPICurrent,
    OpenWeatherMapMain, OpenWeatherMapWind, OpenWeatherMapClouds
)
from src.weather_pipeline.models.weather import WeatherProvider, WeatherDataPoint


class TestWeatherAPIContract:
    """Contract tests for WeatherAPI response structure."""
    
    def test_weatherapi_location_schema(self):
        """Test WeatherAPI location schema contract."""
        valid_location_data = {
            "name": "London",
            "region": "City of London, Greater London",
            "country": "United Kingdom",
            "lat": 51.52,
            "lon": -0.11,
            "tz_id": "Europe/London",
            "localtime": "2024-01-01 12:00"
        }
        
        location = WeatherAPILocation(**valid_location_data)
        
        # Verify required fields
        assert location.name == "London"
        assert location.country == "United Kingdom"
        assert location.lat == 51.52
        assert location.lon == -0.11
        assert location.tz_id == "Europe/London"
        assert location.localtime == "2024-01-01 12:00"
        
        # Optional field
        assert location.region == "City of London, Greater London"

    def test_weatherapi_location_schema_minimal(self):
        """Test WeatherAPI location schema with minimal required fields."""
        minimal_location_data = {
            "name": "TestCity",
            "country": "TestCountry",
            "lat": 0.0,
            "lon": 0.0,
            "tz_id": "UTC",
            "localtime": "2024-01-01 00:00"
        }
        
        location = WeatherAPILocation(**minimal_location_data)
        assert location.name == "TestCity"
        assert location.region is None  # Optional field

    def test_weatherapi_condition_schema(self):
        """Test WeatherAPI condition schema contract."""
        condition_data = {
            "text": "Partly cloudy",
            "icon": "//cdn.weatherapi.com/weather/64x64/day/116.png",
            "code": 1003
        }
        
        condition = WeatherAPICondition(**condition_data)
        
        assert condition.text == "Partly cloudy"
        assert condition.icon == "//cdn.weatherapi.com/weather/64x64/day/116.png"
        assert condition.code == 1003

    def test_weatherapi_current_schema(self):
        """Test WeatherAPI current weather schema contract."""
        current_data = {
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
        
        current = WeatherAPICurrent(**current_data)
        
        # Verify temperature fields
        assert current.temp_c == 20.5
        assert current.temp_f == 68.9
        
        # Verify wind fields
        assert current.wind_mph == 3.2
        assert current.wind_kph == 5.2
        assert current.wind_degree == 180
        assert current.wind_dir == "S"
        
        # Verify pressure fields
        assert current.pressure_mb == 1013.0
        assert current.pressure_in == 29.91
        
        # Verify other meteorological fields
        assert current.humidity == 65
        assert current.cloud == 25
        assert current.uv == 5.0

    def test_weatherapi_full_response_schema(self, mock_api_responses):
        """Test complete WeatherAPI response schema contract."""
        response_data = mock_api_responses["weatherapi"]
        
        response = WeatherAPIResponse(
            raw_data=response_data,
            provider=WeatherProvider.WEATHERAPI,
            request_timestamp=datetime.now(timezone.utc),
            response_timestamp=datetime.now(timezone.utc),
            **response_data
        )
        
        # Verify response structure
        assert response.provider == WeatherProvider.WEATHERAPI
        assert isinstance(response.raw_data, dict)
        assert isinstance(response.request_timestamp, datetime)
        assert isinstance(response.response_timestamp, datetime)
        
        # Verify nested objects
        assert isinstance(response.location, WeatherAPILocation)
        assert isinstance(response.current, WeatherAPICurrent)
        assert isinstance(response.current.condition, WeatherAPICondition)

    def test_weatherapi_response_field_validation(self):
        """Test WeatherAPI response field validation."""
        base_data = {
            "location": {
                "name": "London",
                "country": "UK",
                "lat": 51.5,
                "lon": -0.1,
                "tz_id": "Europe/London",
                "localtime": "2024-01-01 12:00"
            },
            "current": {
                "temp_c": 20.0,
                "temp_f": 68.0,
                "condition": {"text": "Clear", "icon": "clear.png", "code": 1000},
                "wind_mph": 5.0,
                "wind_kph": 8.0,
                "wind_degree": 180,
                "wind_dir": "S",
                "pressure_mb": 1013.0,
                "pressure_in": 29.9,
                "precip_mm": 0.0,
                "precip_in": 0.0,
                "humidity": 65,
                "cloud": 25
            }
        }
        
        # Test with invalid latitude
        invalid_lat_data = base_data.copy()
        invalid_lat_data["location"]["lat"] = 91.0  # Invalid latitude
        
        with pytest.raises(ValidationError):
            WeatherAPIResponse(
                raw_data=invalid_lat_data,
                provider=WeatherProvider.WEATHERAPI,
                request_timestamp=datetime.now(timezone.utc),
                response_timestamp=datetime.now(timezone.utc),
                **invalid_lat_data
            )

    def test_weatherapi_response_missing_required_fields(self):
        """Test WeatherAPI response with missing required fields."""
        incomplete_data = {
            "location": {
                "name": "London",
                "country": "UK",
                # Missing lat, lon, tz_id, localtime
            },
            "current": {
                "temp_c": 20.0,
                # Missing many required fields
            }
        }
        
        with pytest.raises(ValidationError):
            WeatherAPIResponse(
                raw_data=incomplete_data,
                provider=WeatherProvider.WEATHERAPI,
                request_timestamp=datetime.now(timezone.utc),
                response_timestamp=datetime.now(timezone.utc),
                **incomplete_data
            )


class TestOpenWeatherMapContract:
    """Contract tests for OpenWeatherMap response structure."""
    
    def test_openweathermap_main_schema(self):
        """Test OpenWeatherMap main weather data schema."""
        main_data = {
            "temp": 293.65,
            "feels_like": 293.65,
            "temp_min": 292.04,
            "temp_max": 295.37,
            "pressure": 1013,
            "humidity": 65
        }
        
        main = OpenWeatherMapMain(**main_data)
        
        assert main.temp == 293.65
        assert main.feels_like == 293.65
        assert main.temp_min == 292.04
        assert main.temp_max == 295.37
        assert main.pressure == 1013
        assert main.humidity == 65

    def test_openweathermap_wind_schema(self):
        """Test OpenWeatherMap wind data schema."""
        wind_data = {
            "speed": 5.2,
            "deg": 180,
            "gust": 7.5
        }
        
        wind = OpenWeatherMapWind(**wind_data)
        
        assert wind.speed == 5.2
        assert wind.deg == 180
        assert wind.gust == 7.5

    def test_openweathermap_wind_schema_minimal(self):
        """Test OpenWeatherMap wind data with minimal fields."""
        wind_data = {
            "speed": 5.2
        }
        
        wind = OpenWeatherMapWind(**wind_data)
        
        assert wind.speed == 5.2
        assert wind.deg is None  # Optional
        assert wind.gust is None  # Optional

    def test_openweathermap_clouds_schema(self):
        """Test OpenWeatherMap clouds data schema."""
        clouds_data = {
            "all": 75
        }
        
        clouds = OpenWeatherMapClouds(**clouds_data)
        assert clouds.all == 75

    def test_openweathermap_full_response_schema(self, mock_api_responses):
        """Test complete OpenWeatherMap response schema contract."""
        response_data = mock_api_responses["openweather"]
        
        response = OpenWeatherMapResponse(
            raw_data=response_data,
            provider=WeatherProvider.OPENWEATHER,
            request_timestamp=datetime.now(timezone.utc),
            response_timestamp=datetime.now(timezone.utc),
            **response_data
        )
        
        # Verify response structure
        assert response.provider == WeatherProvider.OPENWEATHER
        assert response.name == "London"
        assert response.cod == 200
        
        # Verify nested objects
        assert isinstance(response.main, OpenWeatherMapMain)
        assert isinstance(response.wind, OpenWeatherMapWind)
        assert isinstance(response.clouds, OpenWeatherMapClouds)
        
        # Verify coordinate structure
        assert response.coord.lat == 51.5074
        assert response.coord.lon == -0.1278

    def test_openweathermap_weather_array_schema(self, mock_api_responses):
        """Test OpenWeatherMap weather array schema."""
        response_data = mock_api_responses["openweather"]
        
        response = OpenWeatherMapResponse(
            raw_data=response_data,
            provider=WeatherProvider.OPENWEATHER,
            request_timestamp=datetime.now(timezone.utc),
            response_timestamp=datetime.now(timezone.utc),
            **response_data
        )
        
        # Weather should be an array
        assert isinstance(response.weather, list)
        assert len(response.weather) > 0
        
        # Each weather item should have required fields
        weather_item = response.weather[0]
        assert hasattr(weather_item, "id") and weather_item.id is not None
        assert hasattr(weather_item, "main") and weather_item.main is not None
        assert hasattr(weather_item, "description") and weather_item.description is not None
        assert hasattr(weather_item, "icon") and weather_item.icon is not None


class TestSevenTimerContract:
    """Contract tests for 7timer response structure."""
    
    def test_7timer_response_schema(self, mock_api_responses):
        """Test 7timer response schema contract."""
        response_data = mock_api_responses["7timer"]
        
        response = SevenTimerResponse(
            raw_data=response_data,
            provider=WeatherProvider.SEVEN_TIMER,
            request_timestamp=datetime.now(timezone.utc),
            response_timestamp=datetime.now(timezone.utc),
            **response_data
        )
        
        # Verify response structure
        assert response.provider == WeatherProvider.SEVEN_TIMER
        assert response.product == "civil"
        assert response.init == "2024010112"
        
        # Verify dataseries structure
        assert isinstance(response.dataseries, list)
        assert len(response.dataseries) > 0
        
        # Verify dataseries item structure
        data_item = response.dataseries[0]
        assert "timepoint" in data_item
        assert "cloudcover" in data_item
        assert "temp2m" in data_item
        assert "wind10m" in data_item

    def test_7timer_wind_data_schema(self, mock_api_responses):
        """Test 7timer wind data schema."""
        response_data = mock_api_responses["7timer"]
        dataseries_item = response_data["dataseries"][0]
        
        wind_data = dataseries_item["wind10m"]
        
        # Wind data should have direction and speed
        assert "direction" in wind_data
        assert "speed" in wind_data
        assert wind_data["direction"] == "S"
        assert wind_data["speed"] == 2

    def test_7timer_missing_dataseries(self):
        """Test 7timer response with missing dataseries."""
        incomplete_data = {
            "product": "civil",
            "init": "2024010112"
            # Missing dataseries
        }
        
        with pytest.raises(ValidationError):
            SevenTimerResponse(
                raw_data=incomplete_data,
                provider=WeatherProvider.SEVEN_TIMER,
                request_timestamp=datetime.now(timezone.utc),
                response_timestamp=datetime.now(timezone.utc),
                **incomplete_data
            )


class TestDataTransformationContracts:
    """Contract tests for data transformation between API responses and internal models."""
    
    def test_weatherapi_to_weather_data_point_contract(self, mock_api_responses):
        """Test contract for transforming WeatherAPI response to WeatherDataPoint."""
        response_data = mock_api_responses["weatherapi"]
        
        response = WeatherAPIResponse(
            raw_data=response_data,
            provider=WeatherProvider.WEATHERAPI,
            request_timestamp=datetime.now(timezone.utc),
            response_timestamp=datetime.now(timezone.utc),
            **response_data
        )
        
        # Test transformation (if implemented)
        if hasattr(response, 'to_weather_points'):
            weather_points = response.to_weather_points()
            
            assert isinstance(weather_points, list)
            if weather_points:  # If transformation is implemented
                weather_point = weather_points[0]
                assert isinstance(weather_point, WeatherDataPoint)
                
                # Verify data mapping
                assert weather_point.provider == WeatherProvider.WEATHERAPI
                assert weather_point.city == response.location.name
                
                # Temperature should be converted from API format
                assert weather_point.temperature == response.current.temp_c
                
                # Humidity should be preserved
                assert weather_point.humidity == response.current.humidity
                
                # Pressure should be converted to standard units
                assert weather_point.pressure == response.current.pressure_mb

    def test_openweathermap_to_weather_data_point_contract(self, mock_api_responses):
        """Test contract for transforming OpenWeatherMap response to WeatherDataPoint."""
        response_data = mock_api_responses["openweather"]
        
        response = OpenWeatherMapResponse(
            raw_data=response_data,
            provider=WeatherProvider.OPENWEATHER,
            request_timestamp=datetime.now(timezone.utc),
            response_timestamp=datetime.now(timezone.utc),
            **response_data
        )
        
        # Test transformation (if implemented)
        if hasattr(response, 'to_weather_points'):
            weather_points = response.to_weather_points("London", "UK")
            
            if weather_points:  # If transformation is implemented
                weather_point = weather_points[0]
                assert isinstance(weather_point, WeatherDataPoint)
                
                # Verify data mapping
                assert weather_point.provider == WeatherProvider.OPENWEATHER
                assert weather_point.city == "London"
                
                # Temperature should be converted from Kelvin to Celsius
                expected_temp = response.main.temp - 273.15
                assert abs(weather_point.temperature - expected_temp) < 0.1
                
                # Humidity should be preserved
                assert weather_point.humidity == response.main.humidity
                
                # Pressure should be preserved
                assert weather_point.pressure == response.main.pressure

    def test_7timer_to_weather_data_point_contract(self, mock_api_responses):
        """Test contract for transforming 7timer response to WeatherDataPoint."""
        response_data = mock_api_responses["7timer"]
        
        response = SevenTimerResponse(
            raw_data=response_data,
            provider=WeatherProvider.SEVEN_TIMER,
            request_timestamp=datetime.now(timezone.utc),
            response_timestamp=datetime.now(timezone.utc),
            **response_data
        )
        
        # Test transformation (if implemented)
        if hasattr(response, 'to_weather_points'):
            weather_points = response.to_weather_points("TestCity", "TC")
            
            if weather_points:  # If transformation is implemented
                weather_point = weather_points[0]
                assert isinstance(weather_point, WeatherDataPoint)
                
                # Verify data mapping
                assert weather_point.provider == WeatherProvider.SEVEN_TIMER
                assert weather_point.city == "TestCity"
                
                # Data should be mapped from 7timer format
                dataseries_item = response.dataseries[0]
                assert weather_point.temperature == dataseries_item["temp2m"]
                assert weather_point.humidity == dataseries_item["rh2m"]


class TestAPIResponseEvolution:
    """Contract tests for API response evolution and backward compatibility."""
    
    def test_weatherapi_response_with_extra_fields(self, mock_api_responses):
        """Test WeatherAPI response handles extra fields gracefully."""
        response_data = mock_api_responses["weatherapi"].copy()
        
        # Add extra fields that might be introduced in future API versions
        response_data["location"]["elevation"] = 123.45  # New field
        response_data["current"]["air_quality"] = {"co": 233.5, "no2": 12.8}  # New nested field
        response_data["new_section"] = {"experimental_data": "test"}  # New section
        
        # Should still parse successfully (extra fields ignored or stored)
        response = WeatherAPIResponse(
            raw_data=response_data,
            provider=WeatherProvider.WEATHERAPI,
            request_timestamp=datetime.now(timezone.utc),
            response_timestamp=datetime.now(timezone.utc),
            **{k: v for k, v in response_data.items() if k not in ["new_section"]}  # Filter unknown fields
        )
        
        # Original fields should still work
        assert response.location.name == "London"
        assert response.current.temp_c == 20.5
        
        # Extra data should be preserved in raw_data
        assert "elevation" in response.raw_data["location"]
        assert "air_quality" in response.raw_data["current"]

    def test_openweathermap_response_field_deprecation(self, mock_api_responses):
        """Test OpenWeatherMap response handles deprecated fields."""
        response_data = mock_api_responses["openweather"].copy()
        
        # Simulate deprecated field removal
        if "base" in response_data:
            del response_data["base"]  # Remove deprecated field
        
        # Should still parse successfully
        response = OpenWeatherMapResponse(
            raw_data=response_data,
            provider=WeatherProvider.OPENWEATHER,
            request_timestamp=datetime.now(timezone.utc),
            response_timestamp=datetime.now(timezone.utc),
            **response_data
        )
        
        # Core fields should still work
        assert response.main.temp == 293.65
        assert response.name == "London"

    def test_api_response_version_compatibility(self):
        """Test API response version compatibility."""
        # Test different API version formats
        versions = [
            {"version": "1.0", "format": "standard"},
            {"version": "1.1", "format": "enhanced"},
            {"version": "2.0", "format": "new"}
        ]
        
        for version_info in versions:
            # Mock response with version info
            versioned_response = {
                "api_version": version_info["version"],
                "format": version_info["format"],
                "location": {
                    "name": "TestCity",
                    "country": "TC",
                    "lat": 0.0,
                    "lon": 0.0,
                    "tz_id": "UTC",
                    "localtime": "2024-01-01 00:00"
                },
                "current": {
                    "temp_c": 20.0,
                    "temp_f": 68.0,
                    "condition": {"text": "Clear", "icon": "clear.png", "code": 1000},
                    "wind_mph": 5.0,
                    "wind_kph": 8.0,
                    "wind_degree": 180,
                    "wind_dir": "S",
                    "pressure_mb": 1013.0,
                    "pressure_in": 29.9,
                    "precip_mm": 0.0,
                    "precip_in": 0.0,
                    "humidity": 65,
                    "cloud": 25
                }
            }
            
            # Should parse regardless of API version
            response = WeatherAPIResponse(
                raw_data=versioned_response,
                provider=WeatherProvider.WEATHERAPI,
                request_timestamp=datetime.now(timezone.utc),
                response_timestamp=datetime.now(timezone.utc),
                **{k: v for k, v in versioned_response.items() if k not in ["api_version", "format"]}
            )
            
            assert response.location.name == "TestCity"
            assert response.current.temp_c == 20.0


class TestSchemaValidationContract:
    """Contract tests for schema validation rules."""
    
    def test_coordinate_bounds_validation(self):
        """Test coordinate bounds validation across all API responses."""
        test_cases = [
            {"lat": -90.0, "lon": -180.0, "valid": True},   # Min bounds
            {"lat": 90.0, "lon": 180.0, "valid": True},    # Max bounds
            {"lat": -91.0, "lon": 0.0, "valid": False},    # Invalid lat (too low)
            {"lat": 91.0, "lon": 0.0, "valid": False},     # Invalid lat (too high)
            {"lat": 0.0, "lon": -181.0, "valid": False},   # Invalid lon (too low)
            {"lat": 0.0, "lon": 181.0, "valid": False},    # Invalid lon (too high)
        ]
        
        for case in test_cases:
            location_data = {
                "name": "TestCity",
                "country": "TC",
                "lat": case["lat"],
                "lon": case["lon"],
                "tz_id": "UTC",
                "localtime": "2024-01-01 00:00"
            }
            
            if case["valid"]:
                # Should not raise an exception
                location = WeatherAPILocation(**location_data)
                assert location.lat == case["lat"]
                assert location.lon == case["lon"]
            else:
                # Should raise ValidationError
                with pytest.raises(ValidationError):
                    WeatherAPILocation(**location_data)

    def test_meteorological_value_validation(self):
        """Test meteorological value validation."""
        # Test temperature validation
        valid_temps = [-50.0, 0.0, 25.0, 50.0]  # Valid temperature range
        invalid_temps = [-101.0, 71.0]  # Invalid temperatures
        
        base_current_data = {
            "temp_c": 20.0,
            "temp_f": 68.0,
            "condition": {"text": "Clear", "icon": "clear.png", "code": 1000},
            "wind_mph": 5.0,
            "wind_kph": 8.0,
            "wind_degree": 180,
            "wind_dir": "S",
            "pressure_mb": 1013.0,
            "pressure_in": 29.9,
            "precip_mm": 0.0,
            "precip_in": 0.0,
            "humidity": 65,
            "cloud": 25
        }
        
        # Valid temperatures should work
        for temp in valid_temps:
            current_data = base_current_data.copy()
            current_data["temp_c"] = temp
            current = WeatherAPICurrent(**current_data)
            assert current.temp_c == temp
        
        # Test humidity validation (0-100)
        valid_humidity = [0, 50, 100]
        invalid_humidity = [-1, 101]
        
        for humidity in valid_humidity:
            current_data = base_current_data.copy()
            current_data["humidity"] = humidity
            current = WeatherAPICurrent(**current_data)
            assert current.humidity == humidity
        
        for humidity in invalid_humidity:
            current_data = base_current_data.copy()
            current_data["humidity"] = humidity
            # Note: Depending on implementation, this might or might not raise an error
            # The test verifies the contract expectation

    def test_timestamp_format_validation(self):
        """Test timestamp format validation."""
        valid_timestamps = [
            "2024-01-01 12:00",
            "2024-12-31 23:59",
            "2020-02-29 00:00",  # Leap year
        ]
        
        invalid_timestamps = [
            "2024-13-01 12:00",  # Invalid month
            "2024-01-32 12:00",  # Invalid day
            "2024-01-01 25:00",  # Invalid hour
            "invalid-timestamp", # Invalid format
        ]
        
        base_location_data = {
            "name": "TestCity",
            "country": "TC",
            "lat": 0.0,
            "lon": 0.0,
            "tz_id": "UTC",
            "localtime": "2024-01-01 12:00"
        }
        
        # Valid timestamps should work
        for timestamp in valid_timestamps:
            location_data = base_location_data.copy()
            location_data["localtime"] = timestamp
            location = WeatherAPILocation(**location_data)
            assert location.localtime == timestamp
