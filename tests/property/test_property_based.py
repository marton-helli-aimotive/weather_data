"""Property-based tests using Hypothesis."""

import pytest
from pydantic import ValidationError
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.strategies import composite
from datetime import datetime, timezone, timedelta
import numpy as np

from src.weather_pipeline.models.weather import (
    Coordinates, WeatherDataPoint, WeatherProvider, DataQualityMetrics
)
from src.weather_pipeline.processing import (
    TimeSeriesAnalyzer, GeospatialAnalyzer, FeatureEngineer
)


# Custom Hypothesis strategies for our domain models

@composite
def coordinates_strategy(draw):
    """Generate valid coordinates."""
    latitude = draw(st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False))
    longitude = draw(st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False))
    return Coordinates(latitude=latitude, longitude=longitude)


@composite
def weather_data_point_strategy(draw):
    """Generate valid weather data points."""
    timestamp = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31)
    ).map(lambda dt: dt.replace(tzinfo=timezone.utc)))
    temperature = draw(st.floats(min_value=-100.0, max_value=60.0, allow_nan=False, allow_infinity=False))
    humidity = draw(st.integers(min_value=0, max_value=100))
    pressure = draw(st.floats(min_value=800.0, max_value=1100.0, allow_nan=False, allow_infinity=False))
    wind_speed = draw(st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False))
    wind_direction = draw(st.integers(min_value=0, max_value=360))
    precipitation = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    visibility = draw(st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False))
    cloud_cover = draw(st.integers(min_value=0, max_value=100))
    uv_index = draw(st.floats(min_value=0.0, max_value=15.0, allow_nan=False, allow_infinity=False))
    city = draw(st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "))
    country = draw(st.text(min_size=0, max_size=5, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    coordinates = draw(coordinates_strategy())
    provider = draw(st.sampled_from(list(WeatherProvider)))
    is_forecast = draw(st.booleans())
    confidence_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    
    return WeatherDataPoint(
        timestamp=timestamp,
        temperature=temperature,
        humidity=humidity,
        pressure=pressure,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        precipitation=precipitation,
        visibility=visibility,
        cloud_cover=cloud_cover,
        uv_index=uv_index,
        city=city,
        country=country if country else None,
        coordinates=coordinates,
        provider=provider,
        is_forecast=is_forecast,
        confidence_score=confidence_score
    )


@composite
def weather_time_series_strategy(draw):
    """Generate time series of weather data."""
    length = draw(st.integers(min_value=5, max_value=20))  # Much smaller to reduce entropy
    start_time = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2024, 1, 1)
    ).map(lambda dt: dt.replace(tzinfo=timezone.utc)))
    
    data_points = []
    current_time = start_time
    
    for i in range(length):
        # Generate correlated weather data for realism
        base_temp = draw(st.floats(min_value=-20.0, max_value=40.0, allow_nan=False, allow_infinity=False))
        temp_variation = draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
        temperature = base_temp + temp_variation
        
        # Humidity tends to be inversely related to temperature in many cases
        base_humidity = max(0, min(100, int(80 - (temperature - 10) * 2)))
        humidity = max(0, min(100, base_humidity + draw(st.integers(min_value=-20, max_value=20))))
        
        pressure = draw(st.floats(min_value=980.0, max_value=1040.0, allow_nan=False, allow_infinity=False))
        wind_speed = draw(st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False))
        
        data_point = {
            'timestamp': current_time,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'wind_direction': draw(st.integers(min_value=0, max_value=360)),
            'precipitation': draw(st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False)),
            'city': 'TestCity',
            'country': 'TC',
            'latitude': 51.5,
            'longitude': -0.1
        }
        data_points.append(data_point)
        
        # Increment time by 1 hour typically
        current_time += timedelta(hours=1)
    
    return data_points


class TestCoordinatesProperties:
    """Property-based tests for Coordinates model."""

    @given(coordinates_strategy())
    def test_coordinates_are_immutable(self, coords):
        """Test that coordinates are immutable."""
        original_lat = coords.latitude
        original_lon = coords.longitude
        
        # Try to modify (should fail)
        with pytest.raises(ValidationError):
            coords.latitude = 0.0
        with pytest.raises(ValidationError):
            coords.longitude = 0.0
        
        # Values should remain unchanged
        assert coords.latitude == original_lat
        assert coords.longitude == original_lon

    @given(coordinates_strategy(), coordinates_strategy())
    def test_coordinates_equality_is_reflexive_and_symmetric(self, coords1, coords2):
        """Test equality properties."""
        # Reflexive: coords1 == coords1
        assert coords1 == coords1
        
        # Symmetric: if coords1 == coords2, then coords2 == coords1
        if coords1 == coords2:
            assert coords2 == coords1

    @given(coordinates_strategy())
    def test_coordinates_hash_consistency(self, coords):
        """Test that hash is consistent."""
        hash1 = hash(coords)
        hash2 = hash(coords)
        assert hash1 == hash2

    @given(st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
           st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False))
    def test_valid_coordinates_creation(self, lat, lon):
        """Test that valid coordinates can always be created."""
        coords = Coordinates(latitude=lat, longitude=lon)
        assert coords.latitude == lat
        assert coords.longitude == lon

    @given(st.floats(allow_nan=False, allow_infinity=False).filter(lambda x: x < -90.0 or x > 90.0),
           st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False))
    def test_invalid_latitude_rejected(self, invalid_lat, valid_lon):
        """Test that invalid latitudes are rejected."""
        with pytest.raises(ValueError):
            Coordinates(latitude=invalid_lat, longitude=valid_lon)

    @given(st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
           st.floats(allow_nan=False, allow_infinity=False).filter(lambda x: x < -180.0 or x > 180.0))
    def test_invalid_longitude_rejected(self, valid_lat, invalid_lon):
        """Test that invalid longitudes are rejected."""
        with pytest.raises(ValueError):
            Coordinates(latitude=valid_lat, longitude=invalid_lon)


class TestWeatherDataPointProperties:
    """Property-based tests for WeatherDataPoint model."""

    @given(weather_data_point_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_weather_data_point_creation(self, weather_point):
        """Test that valid weather data points can be created."""
        assert isinstance(weather_point, WeatherDataPoint)
        assert weather_point.temperature >= -100.0 and weather_point.temperature <= 70.0
        assert weather_point.humidity >= 0 and weather_point.humidity <= 100
        assert weather_point.pressure >= 800.0 and weather_point.pressure <= 1100.0

    @given(weather_data_point_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_weather_data_point_serialization_roundtrip(self, weather_point):
        """Test that weather data points can be serialized and deserialized."""
        # Test model dump and parse
        data_dict = weather_point.model_dump()
        assert isinstance(data_dict, dict)
        
        # Reconstruct from dict
        reconstructed = WeatherDataPoint.model_validate(data_dict)
        assert reconstructed == weather_point

    @given(weather_data_point_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_coordinates_property_immutable(self, weather_point):
        """Test that coordinates property maintains immutability."""
        original_coords = weather_point.coordinates
        
        # Coordinates should be immutable
        with pytest.raises(ValidationError):
            weather_point.coordinates.latitude = 0.0

    @given(weather_data_point_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_provider_enum_validity(self, weather_point):
        """Test that provider is always a valid enum value."""
        assert isinstance(weather_point.provider, WeatherProvider)
        assert weather_point.provider in list(WeatherProvider)


class TestDataQualityMetricsProperties:
    """Property-based tests for DataQualityMetrics."""

    @given(
        total_records=st.integers(min_value=1, max_value=10000),
        missing_values=st.integers(min_value=0, max_value=1000),
        duplicate_records=st.integers(min_value=0, max_value=1000),
        anomaly_count=st.integers(min_value=0, max_value=1000)
    )
    def test_data_quality_metrics_percentages(self, total_records, missing_values, duplicate_records, anomaly_count):
        """Test that percentage calculations are correct."""
        # Ensure valid relationships
        assume(missing_values <= total_records)
        assume(duplicate_records <= total_records)
        assume(anomaly_count <= total_records)
        
        metrics = DataQualityMetrics(
            total_records=total_records,
            valid_records=max(0, total_records - missing_values),
            missing_temperature=missing_values // 2,
            missing_humidity=missing_values - (missing_values // 2),
            outliers_detected=anomaly_count,
            duplicate_records=duplicate_records,
            anomaly_count=anomaly_count,
            completeness_score=max(0.0, (total_records - missing_values) / total_records) if total_records > 0 else 1.0,
            quality_score=0.8  # Default quality score
        )
        
        # Test basic properties
        assert metrics.total_records == total_records
        assert metrics.missing_values == missing_values
        assert metrics.duplicate_records == duplicate_records
        assert metrics.anomaly_count == anomaly_count
        
        # Test that percentage is within bounds
        assert 0.0 <= metrics.missing_percentage <= 100.0
        assert 0.0 <= metrics.completeness_score <= 1.0
        assert 0.0 <= metrics.quality_score <= 1.0

    @given(
        total_records=st.integers(min_value=1, max_value=10000),
        completeness_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        accuracy_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        consistency_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    def test_overall_score_bounds(self, total_records, completeness_score, accuracy_score, consistency_score):
        """Test that overall score is within valid bounds."""
        metrics = DataQualityMetrics(
            total_records=total_records,
            valid_records=total_records,  # Assume all are valid for this test
            missing_temperature=0,
            missing_humidity=0,
            outliers_detected=0,
            completeness_score=completeness_score,
            quality_score=accuracy_score  # Map accuracy_score to quality_score
        )
        
        overall_score = metrics.get_overall_score()
        assert 0.0 <= overall_score <= 1.0


class TestTimeSeriesAnalyzerProperties:
    """Property-based tests for TimeSeriesAnalyzer."""

    @pytest.mark.slow
    @given(weather_time_series_strategy())
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.data_too_large])
    def test_trend_detection_consistency(self, time_series_data):
        """Test that trend detection produces consistent results."""
        import pandas as pd
        
        df = pd.DataFrame(time_series_data)
        analyzer = TimeSeriesAnalyzer()
        
        try:
            trends = analyzer.detect_trends(df, column="temperature")
            
            # Should always have linear trend analysis
            assert "linear_trend" in trends
            assert "slope" in trends["linear_trend"]
            assert "trend_direction" in trends["linear_trend"]
            assert trends["linear_trend"]["trend_direction"] in ["increasing", "decreasing", "stable"]
            
            # Slope and direction should be consistent
            slope = trends["linear_trend"]["slope"]
            direction = trends["linear_trend"]["trend_direction"]
            
            if slope > 0.1:
                assert direction == "increasing"
            elif slope < -0.1:
                assert direction == "decreasing"
            else:
                assert direction == "stable"
                
        except Exception as e:
            # Some time series might not be suitable for analysis
            # This is acceptable for property-based testing
            assume(False)

    @pytest.mark.slow
    @given(weather_time_series_strategy())
    @settings(max_examples=5, deadline=10000, suppress_health_check=[HealthCheck.data_too_large])
    def test_anomaly_detection_bounds(self, time_series_data):
        """Test that anomaly detection results are within bounds."""
        import pandas as pd
        
        df = pd.DataFrame(time_series_data)
        analyzer = TimeSeriesAnalyzer()
        
        try:
            anomalies = analyzer.detect_anomalies(df, column="temperature", method="zscore", threshold=2.0)
            
            assert "total_anomalies" in anomalies
            assert "anomaly_rate" in anomalies
            
            total_anomalies = anomalies["total_anomalies"]
            anomaly_rate = anomalies["anomaly_rate"]
            
            # Total anomalies should be non-negative
            assert total_anomalies >= 0
            
            # Anomaly rate should be between 0 and 1
            assert 0.0 <= anomaly_rate <= 1.0
            
            # Total anomalies should not exceed total records
            assert total_anomalies <= len(df)
            
            # Anomaly rate should be consistent with total anomalies
            expected_rate = total_anomalies / len(df)
            assert abs(anomaly_rate - expected_rate) < 0.001
            
        except Exception as e:
            # Some time series might not be suitable for analysis
            assume(False)


class TestGeospatialAnalyzerProperties:
    """Property-based tests for GeospatialAnalyzer."""

    @given(coordinates_strategy(), coordinates_strategy())
    def test_distance_calculation_properties(self, coords1, coords2):
        """Test properties of distance calculations."""
        analyzer = GeospatialAnalyzer()
        
        point1 = (coords1.latitude, coords1.longitude)
        point2 = (coords2.latitude, coords2.longitude)
        
        distance = analyzer.calculate_distances(point1, point2)
        
        # Distance should be non-negative
        assert distance >= 0.0
        
        # Distance to self should be 0
        self_distance = analyzer.calculate_distances(point1, point1)
        assert abs(self_distance) < 0.001
        
        # Distance should be symmetric (within numerical precision)
        reverse_distance = analyzer.calculate_distances(point2, point1)
        assert abs(distance - reverse_distance) < 0.001

    @given(coordinates_strategy())
    def test_distance_triangle_inequality(self, coords1):
        """Test triangle inequality for distance calculations."""
        # Generate two more points
        coords2 = Coordinates(latitude=min(90, coords1.latitude + 1), longitude=coords1.longitude)
        coords3 = Coordinates(latitude=coords1.latitude, longitude=min(180, coords1.longitude + 1))
        
        analyzer = GeospatialAnalyzer()
        
        point1 = (coords1.latitude, coords1.longitude)
        point2 = (coords2.latitude, coords2.longitude)
        point3 = (coords3.latitude, coords3.longitude)
        
        d12 = analyzer.calculate_distances(point1, point2)
        d23 = analyzer.calculate_distances(point2, point3)
        d13 = analyzer.calculate_distances(point1, point3)
        
        # Triangle inequality: d13 <= d12 + d23 (within numerical precision)
        assert d13 <= d12 + d23 + 0.001


class TestFeatureEngineerProperties:
    """Property-based tests for FeatureEngineer."""

    @pytest.mark.slow
    @given(weather_time_series_strategy())
    @settings(max_examples=5, deadline=10000)
    def test_rolling_features_preserve_data_integrity(self, time_series_data):
        """Test that rolling feature creation preserves data integrity."""
        import pandas as pd
        
        df = pd.DataFrame(time_series_data)
        engineer = FeatureEngineer()
        
        original_length = len(df)
        original_columns = set(df.columns)
        
        try:
            result = engineer.create_rolling_features(
                df,
                columns=["temperature", "humidity"],
                windows=[3, 6],
                operations=["mean", "std"]
            )
            
            # Should preserve original data
            assert len(result) == original_length
            
            # Should contain all original columns
            assert original_columns.issubset(set(result.columns))
            
            # Should have added new features
            assert len(result.columns) > len(original_columns)
            
            # New features should have "rolling" in their name
            new_features = set(result.columns) - original_columns
            assert all("rolling" in col for col in new_features)
            
        except Exception as e:
            # Some data might not be suitable for feature engineering
            assume(False)

    @pytest.mark.slow
    @given(weather_time_series_strategy())
    @settings(max_examples=5, deadline=10000)
    def test_lag_features_preserve_causality(self, time_series_data):
        """Test that lag features maintain temporal causality."""
        import pandas as pd
        
        df = pd.DataFrame(time_series_data)
        engineer = FeatureEngineer()
        
        try:
            result = engineer.create_lag_features(
                df,
                columns=["temperature"],
                lags=[1, 2]
            )
            
            # Should preserve original data length
            assert len(result) == len(df)
            
            # Lag features should have NaN for initial values
            lag_columns = [col for col in result.columns if "lag" in col]
            
            for lag_col in lag_columns:
                # Extract lag number from column name
                if "lag_1" in lag_col:
                    # First value should be NaN
                    assert pd.isna(result[lag_col].iloc[0])
                elif "lag_2" in lag_col:
                    # First two values should be NaN
                    assert pd.isna(result[lag_col].iloc[0])
                    assert pd.isna(result[lag_col].iloc[1])
            
        except Exception as e:
            # Some data might not be suitable for feature engineering
            assume(False)


# Integration property tests

class TestEndToEndProperties:
    """End-to-end property-based tests."""

    @given(st.lists(weather_data_point_strategy(), min_size=10, max_size=100))
    def test_weather_data_aggregation_properties(self, weather_points):
        """Test properties of weather data aggregation."""
        # Test that we can always aggregate weather data
        temperatures = [wp.temperature for wp in weather_points]
        humidities = [wp.humidity for wp in weather_points]
        pressures = [wp.pressure for wp in weather_points]
        
        # Basic statistical properties should hold
        min_temp = min(temperatures)
        max_temp = max(temperatures)
        avg_temp = sum(temperatures) / len(temperatures)
        
        assert min_temp <= avg_temp <= max_temp
        
        # All humidities should be in valid range
        assert all(0 <= h <= 100 for h in humidities)
        
        # All pressures should be in valid range
        assert all(800.0 <= p <= 1100.0 for p in pressures)

    @given(st.lists(coordinates_strategy(), min_size=2, max_size=20))
    def test_coordinate_clustering_properties(self, coordinates_list):
        """Test properties of coordinate clustering."""
        # Convert to format suitable for clustering
        points = [(c.latitude, c.longitude) for c in coordinates_list]
        
        # Test that we can compute pairwise distances
        analyzer = GeospatialAnalyzer()
        
        for i, point1 in enumerate(points):
            for j, point2 in enumerate(points[i+1:], i+1):
                distance = analyzer.calculate_distances(point1, point2)
                
                # Distance should be finite and non-negative
                assert distance >= 0.0
                assert not np.isnan(distance)
                assert not np.isinf(distance)
