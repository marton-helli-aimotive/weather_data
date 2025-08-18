"""Unit tests for data models and validation."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any
from pydantic import ValidationError

from src.weather_pipeline.models.weather import (
    Coordinates, WeatherDataPoint, WeatherProvider,
    DataQualityMetrics, LocationInfo, WeatherAlert
)
from src.weather_pipeline.models.api_responses import (
    WeatherAPIResponse, OpenWeatherMapResponse, SevenTimerResponse
)


class TestCoordinates:
    """Test Coordinates model validation and behavior."""

    def test_valid_coordinates(self):
        """Test valid coordinate creation."""
        coords = Coordinates(latitude=47.4979, longitude=19.0402)
        assert coords.latitude == 47.4979
        assert coords.longitude == 19.0402

    def test_boundary_coordinates(self):
        """Test boundary coordinate values."""
        # Test extremes
        extreme_coords = Coordinates(latitude=90.0, longitude=180.0)
        assert extreme_coords.latitude == 90.0
        assert extreme_coords.longitude == 180.0

        extreme_coords_negative = Coordinates(latitude=-90.0, longitude=-180.0)
        assert extreme_coords_negative.latitude == -90.0
        assert extreme_coords_negative.longitude == -180.0

    def test_invalid_latitude(self):
        """Test invalid latitude values."""
        with pytest.raises(ValidationError):
            Coordinates(latitude=91.0, longitude=0.0)

        with pytest.raises(ValidationError):
            Coordinates(latitude=-91.0, longitude=0.0)

    def test_invalid_longitude(self):
        """Test invalid longitude values."""
        with pytest.raises(ValidationError):
            Coordinates(latitude=0.0, longitude=181.0)

        with pytest.raises(ValidationError):
            Coordinates(latitude=0.0, longitude=-181.0)

    def test_coordinates_immutable(self):
        """Test that coordinates are immutable."""
        coords = Coordinates(latitude=47.4979, longitude=19.0402)
        with pytest.raises(ValidationError):
            coords.latitude = 50.0

    def test_coordinates_equality(self):
        """Test coordinate equality comparison."""
        coords1 = Coordinates(latitude=47.4979, longitude=19.0402)
        coords2 = Coordinates(latitude=47.4979, longitude=19.0402)
        coords3 = Coordinates(latitude=50.0, longitude=20.0)

        assert coords1 == coords2
        assert coords1 != coords3

    def test_coordinates_hash(self):
        """Test that coordinates are hashable."""
        coords1 = Coordinates(latitude=47.4979, longitude=19.0402)
        coords2 = Coordinates(latitude=47.4979, longitude=19.0402)
        
        coord_set = {coords1, coords2}
        assert len(coord_set) == 1  # Should be deduplicated


class TestWeatherDataPoint:
    """Test WeatherDataPoint model validation and behavior."""

    def test_valid_weather_data_point(self, sample_coordinates):
        """Test creating a valid weather data point."""
        coords = Coordinates(**sample_coordinates["london"])
        
        weather_point = WeatherDataPoint(
            timestamp=datetime.now(timezone.utc),
            temperature=20.5,
            humidity=65,
            pressure=1013.25,
            wind_speed=5.2,
            wind_direction=180,
            precipitation=0.0,
            visibility=10.0,
            cloud_cover=25,
            uv_index=5.0,
            city="London",
            country="UK",
            coordinates=coords,
            provider=WeatherProvider.WEATHERAPI,
            is_forecast=True,
            confidence_score=0.95
        )

        assert weather_point.temperature == 20.5
        assert weather_point.city == "London"
        assert weather_point.provider == WeatherProvider.WEATHERAPI

    def test_temperature_validation(self, sample_coordinates):
        """Test temperature validation ranges."""
        coords = Coordinates(**sample_coordinates["london"])
        
        # Valid temperature
        weather_point = WeatherDataPoint(
            timestamp=datetime.now(timezone.utc),
            temperature=-50.0,  # Extreme but valid
            humidity=65,
            pressure=1013.25,
            city="London",
            coordinates=coords,
            provider=WeatherProvider.WEATHERAPI
        )
        assert weather_point.temperature == -50.0

        # Invalid temperature - too low
        with pytest.raises(ValueError):
            WeatherDataPoint(
                timestamp=datetime.now(timezone.utc),
                temperature=-101.0,
                humidity=65,
                pressure=1013.25,
                city="London",
                coordinates=coords,
                provider=WeatherProvider.WEATHERAPI
            )

        # Invalid temperature - too high
        with pytest.raises(ValueError):
            WeatherDataPoint(
                timestamp=datetime.now(timezone.utc),
                temperature=71.0,  # Above 70°C limit
                humidity=65,
                pressure=1013.25,
                city="London",
                coordinates=coords,
                provider=WeatherProvider.WEATHERAPI
            )

    def test_humidity_validation(self, sample_coordinates):
        """Test humidity validation."""
        coords = Coordinates(**sample_coordinates["london"])

        # Invalid humidity - negative
        with pytest.raises(ValueError):
            WeatherDataPoint(
                timestamp=datetime.now(timezone.utc),
                temperature=20.0,
                humidity=-1,
                pressure=1013.25,
                city="London",
                coordinates=coords,
                provider=WeatherProvider.WEATHERAPI
            )

        # Invalid humidity - over 100
        with pytest.raises(ValueError):
            WeatherDataPoint(
                timestamp=datetime.now(timezone.utc),
                temperature=20.0,
                humidity=101,
                pressure=1013.25,
                city="London",
                coordinates=coords,
                provider=WeatherProvider.WEATHERAPI
            )

    def test_pressure_validation(self, sample_coordinates):
        """Test pressure validation."""
        coords = Coordinates(**sample_coordinates["london"])

        # Invalid pressure - negative
        with pytest.raises(ValidationError):
            WeatherDataPoint(
                timestamp=datetime.now(timezone.utc),
                temperature=20.0,
                humidity=65,
                pressure=-10.0,  # Negative pressure should be invalid
                city="London",
                coordinates=coords,
                provider=WeatherProvider.WEATHERAPI
            )

        # Valid high pressure (no upper limit in model)
        high_pressure_data = WeatherDataPoint(
            timestamp=datetime.now(timezone.utc),
            temperature=20.0,
            humidity=65,
            pressure=1101.0,  # High pressure should be valid
            city="London",
            coordinates=coords,
            provider=WeatherProvider.WEATHERAPI
        )
        assert high_pressure_data.pressure == 1101.0

    def test_wind_validation(self, sample_coordinates):
        """Test wind speed and direction validation."""
        coords = Coordinates(**sample_coordinates["london"])

        # Invalid wind speed - negative
        with pytest.raises(ValueError):
            WeatherDataPoint(
                timestamp=datetime.now(timezone.utc),
                temperature=20.0,
                humidity=65,
                pressure=1013.25,
                wind_speed=-1.0,
                city="London",
                coordinates=coords,
                provider=WeatherProvider.WEATHERAPI
            )

        # Invalid wind direction - negative
        with pytest.raises(ValueError):
            WeatherDataPoint(
                timestamp=datetime.now(timezone.utc),
                temperature=20.0,
                humidity=65,
                pressure=1013.25,
                wind_direction=-1,
                city="London",
                coordinates=coords,
                provider=WeatherProvider.WEATHERAPI
            )

        # Invalid wind direction - over 360
        with pytest.raises(ValueError):
            WeatherDataPoint(
                timestamp=datetime.now(timezone.utc),
                temperature=20.0,
                humidity=65,
                pressure=1013.25,
                wind_direction=361,
                city="London",
                coordinates=coords,
                provider=WeatherProvider.WEATHERAPI
            )

    def test_required_fields(self, sample_coordinates):
        """Test that required fields are enforced."""
        coords = Coordinates(**sample_coordinates["london"])

        # Missing required field should raise error
        with pytest.raises(ValueError):
            WeatherDataPoint(
                # Missing timestamp
                temperature=20.0,
                humidity=65,
                pressure=1013.25,
                city="London",
                coordinates=coords,
                provider=WeatherProvider.WEATHERAPI
            )

    def test_city_name_validation(self, sample_coordinates):
        """Test city name validation."""
        coords = Coordinates(**sample_coordinates["london"])

        # Empty city name should fail
        with pytest.raises(ValueError):
            WeatherDataPoint(
                timestamp=datetime.now(timezone.utc),
                temperature=20.0,
                humidity=65,
                pressure=1013.25,
                city="",  # Empty string
                coordinates=coords,
                provider=WeatherProvider.WEATHERAPI
            )

    def test_confidence_score_validation(self, sample_coordinates):
        """Test confidence score validation."""
        coords = Coordinates(**sample_coordinates["london"])

        # Invalid confidence score - negative
        with pytest.raises(ValueError):
            WeatherDataPoint(
                timestamp=datetime.now(timezone.utc),
                temperature=20.0,
                humidity=65,
                pressure=1013.25,
                city="London",
                coordinates=coords,
                provider=WeatherProvider.WEATHERAPI,
                confidence_score=-0.1
            )

        # Invalid confidence score - over 1.0
        with pytest.raises(ValueError):
            WeatherDataPoint(
                timestamp=datetime.now(timezone.utc),
                temperature=20.0,
                humidity=65,
                pressure=1013.25,
                city="London",
                coordinates=coords,
                provider=WeatherProvider.WEATHERAPI,
                confidence_score=1.1
            )


class TestWeatherProvider:
    """Test WeatherProvider enum."""

    def test_provider_values(self):
        """Test provider enum values."""
        assert WeatherProvider.SEVEN_TIMER == "7timer"
        assert WeatherProvider.OPENWEATHER == "openweather"
        assert WeatherProvider.WEATHERAPI == "weatherapi"

    def test_provider_iteration(self):
        """Test iterating over providers."""
        providers = list(WeatherProvider)
        assert len(providers) == 3
        assert WeatherProvider.SEVEN_TIMER in providers
        assert WeatherProvider.OPENWEATHER in providers
        assert WeatherProvider.WEATHERAPI in providers


class TestDataQualityMetrics:
    """Test DataQualityMetrics model."""

    def test_valid_metrics(self):
        """Test creating valid data quality metrics."""
        metrics = DataQualityMetrics(
            total_records=100,
            valid_records=95,
            missing_temperature=2,
            missing_humidity=3,
            outliers_detected=1,
            completeness_score=0.95,
            quality_score=0.92,
            assessment_time=datetime.now(timezone.utc),
            data_time_range_start=datetime.now(timezone.utc),
            data_time_range_end=datetime.now(timezone.utc)
        )

        assert metrics.total_records == 100
        assert metrics.completeness_score == 0.95
        assert metrics.quality_score == 0.92

    def test_invalid_scores(self):
        """Test validation of score fields."""
        # Invalid completeness score
        with pytest.raises(ValueError):
            DataQualityMetrics(
                total_records=100,
                missing_values=5,
                completeness_score=1.5  # Over 1.0
            )

        # Invalid accuracy score
        with pytest.raises(ValueError):
            DataQualityMetrics(
                total_records=100,
                missing_values=5,
                accuracy_score=-0.1  # Below 0.0
            )

    def test_calculated_properties(self):
        """Test calculated properties of metrics."""
        metrics = DataQualityMetrics(
            total_records=100,
            valid_records=90,
            missing_temperature=5,
            missing_humidity=5,
            outliers_detected=2,
            completeness_score=0.9,
            quality_score=0.85,
            assessment_time=datetime.now(timezone.utc),
            data_time_range_start=datetime.now(timezone.utc),
            data_time_range_end=datetime.now(timezone.utc)
        )

        assert metrics.total_records == 100
        assert metrics.valid_records == 90
        assert metrics.outliers_detected == 2


class TestAPIResponseModels:
    """Test API response model validation."""

    def test_weatherapi_response_validation(self, mock_api_responses):
        """Test WeatherAPI response model validation."""
        response_data = mock_api_responses["weatherapi"]
        
        response = WeatherAPIResponse(
            raw_data=response_data,
            provider=WeatherProvider.WEATHERAPI,
            request_timestamp=datetime.now(timezone.utc),
            response_timestamp=datetime.now(timezone.utc),
            **response_data
        )

        assert response.provider == WeatherProvider.WEATHERAPI
        assert response.location.name == "London"
        assert response.current.temp_c == 20.5

    def test_openweather_response_validation(self, mock_api_responses):
        """Test OpenWeatherMap response model validation."""
        response_data = mock_api_responses["openweather"]
        
        response = OpenWeatherMapResponse(
            raw_data=response_data,
            provider=WeatherProvider.OPENWEATHER,
            request_timestamp=datetime.now(timezone.utc),
            response_timestamp=datetime.now(timezone.utc),
            **response_data
        )

        assert response.provider == WeatherProvider.OPENWEATHER
        assert response.name == "London"
        assert response.main.temp == 293.65

    def test_7timer_response_validation(self, mock_api_responses):
        """Test 7timer response model validation."""
        response_data = mock_api_responses["7timer"]
        
        response = SevenTimerResponse(
            raw_data=response_data,
            provider=WeatherProvider.SEVEN_TIMER,
            request_timestamp=datetime.now(timezone.utc),
            response_timestamp=datetime.now(timezone.utc),
            **response_data
        )

        assert response.provider == WeatherProvider.SEVEN_TIMER
        assert response.product == "civil"
        assert len(response.dataseries) == 1

    def test_invalid_provider_in_response(self, mock_api_responses):
        """Test invalid provider in response."""
        response_data = mock_api_responses["weatherapi"]
        
        with pytest.raises(ValueError):
            WeatherAPIResponse(
                raw_data=response_data,
                provider="invalid_provider",  # Invalid provider
                request_timestamp=datetime.now(timezone.utc),
                response_timestamp=datetime.now(timezone.utc),
                **response_data
            )
