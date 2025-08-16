"""Tests for Milestone 3 advanced data processing capabilities."""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta

from weather_pipeline.processing import (
    TimeSeriesAnalyzer,
    GeospatialAnalyzer,
    FeatureEngineer,
    DataQualityMonitor,
    PerformanceComparator,
    CacheManager
)


@pytest.fixture
def sample_weather_data():
    """Create sample weather data for testing."""
    np.random.seed(42)
    n_records = 100
    
    timestamps = pd.date_range('2024-01-01', periods=n_records, freq='H')
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'city': np.random.choice(['New York', 'London', 'Tokyo'], n_records),
        'temperature': np.random.normal(20, 5, n_records),
        'humidity': np.random.randint(30, 90, n_records),
        'pressure': np.random.normal(1013, 20, n_records),
        'wind_speed': np.random.exponential(5, n_records),
        'latitude': np.random.uniform(40, 45, n_records),
        'longitude': np.random.uniform(-75, -70, n_records)
    })


class TestTimeSeriesAnalyzer:
    """Test time series analysis capabilities."""
    
    def test_detect_trends(self, sample_weather_data):
        """Test trend detection functionality."""
        analyzer = TimeSeriesAnalyzer()
        
        trends = analyzer.detect_trends(sample_weather_data, column="temperature")
        
        assert "linear_trend" in trends
        assert "slope" in trends["linear_trend"]
        assert "trend_direction" in trends["linear_trend"]
        assert trends["linear_trend"]["trend_direction"] in ["increasing", "decreasing", "stable"]
        
        assert "stationarity" in trends
        assert "is_stationary" in trends["stationarity"]
        assert isinstance(trends["stationarity"]["is_stationary"], bool)
        
    def test_analyze_seasonality(self, sample_weather_data):
        """Test seasonality analysis."""
        analyzer = TimeSeriesAnalyzer()
        
        # Add more data for seasonality analysis
        longer_data = pd.concat([sample_weather_data] * 3, ignore_index=True)
        seasonality = analyzer.analyze_seasonality(longer_data, column="temperature")
        
        # Should return seasonality analysis results for sufficient data
        assert "decomposition" in seasonality
        assert "period_used" in seasonality["decomposition"]
        assert "seasonal_strength" in seasonality["decomposition"]
        assert isinstance(seasonality["decomposition"]["seasonal_strength"], (float, int))
        assert isinstance(seasonality["decomposition"]["period_used"], int)
        
    def test_detect_anomalies(self, sample_weather_data):
        """Test anomaly detection."""
        analyzer = TimeSeriesAnalyzer()
        
        # Add some obvious anomalies
        data_with_anomalies = sample_weather_data.copy()
        data_with_anomalies.loc[10, 'temperature'] = 100  # Extreme value
        
        anomalies = analyzer.detect_anomalies(data_with_anomalies, method="zscore", threshold=2.0)
        
        assert "total_anomalies" in anomalies
        assert "anomaly_rate" in anomalies
        assert anomalies["total_anomalies"] >= 0
        assert 0 <= anomalies["anomaly_rate"] <= 1
        
    def test_forecast_simple(self, sample_weather_data):
        """Test simple forecasting."""
        analyzer = TimeSeriesAnalyzer()
        
        forecast = analyzer.forecast_simple(sample_weather_data, steps=12, method="linear")
        
        assert "forecast_values" in forecast
        assert len(forecast["forecast_values"]) == 12
        assert "last_observed" in forecast


class TestGeospatialAnalyzer:
    """Test geospatial analysis capabilities."""
    
    def test_cluster_weather_patterns(self, sample_weather_data):
        """Test weather pattern clustering."""
        analyzer = GeospatialAnalyzer()
        
        clusters = analyzer.cluster_weather_patterns(
            sample_weather_data,
            features=["temperature", "humidity", "pressure"],
            method="kmeans",
            n_clusters=3
        )
        
        assert "n_clusters" in clusters
        assert clusters["n_clusters"] == 3
        assert "cluster_labels" in clusters
        assert len(clusters["cluster_labels"]) > 0
        
    def test_calculate_distances(self):
        """Test distance calculations."""
        analyzer = GeospatialAnalyzer()
        
        # Test known distance (approximately)
        ny_coords = (40.7128, -74.0060)
        boston_coords = (42.3601, -71.0589)
        
        distance = analyzer.calculate_distances(ny_coords, boston_coords)
        
        # Distance should be roughly 300km
        assert 250 <= distance <= 350
        
    def test_find_nearest_stations(self, sample_weather_data):
        """Test finding nearest weather stations."""
        analyzer = GeospatialAnalyzer()
        
        target_point = (42.0, -72.0)
        
        nearest = analyzer.find_nearest_stations(
            target_point,
            sample_weather_data,
            n_nearest=3
        )
        
        assert "nearest_stations" in nearest
        assert len(nearest["nearest_stations"]) <= 3
        assert "mean_distance" in nearest


class TestFeatureEngineer:
    """Test feature engineering capabilities."""
    
    def test_create_rolling_features(self, sample_weather_data):
        """Test rolling feature creation."""
        engineer = FeatureEngineer()
        
        original_columns = len(sample_weather_data.columns)
        
        result = engineer.create_rolling_features(
            sample_weather_data,
            columns=["temperature", "humidity"],
            windows=[3, 6],
            operations=["mean", "std"]
        )
        
        # Should have added features
        assert len(result.columns) > original_columns
        
        # Check for specific rolling features
        rolling_features = [col for col in result.columns if "rolling" in col]
        assert len(rolling_features) > 0
        
    def test_create_lag_features(self, sample_weather_data):
        """Test lag feature creation."""
        engineer = FeatureEngineer()
        
        result = engineer.create_lag_features(
            sample_weather_data,
            columns=["temperature"],
            lags=[1, 2]
        )
        
        # Check for lag features
        lag_features = [col for col in result.columns if "lag" in col]
        assert len(lag_features) == 2  # 1 column × 2 lags
        
    def test_create_derived_metrics(self, sample_weather_data):
        """Test derived metric creation."""
        engineer = FeatureEngineer()
        
        result = engineer.create_derived_metrics(sample_weather_data)
        
        # Should have derived features like heat index
        derived_features = [col for col in result.columns 
                          if col in ["heat_index", "wind_chill", "dew_point"]]
        assert len(derived_features) > 0
        
    def test_create_temporal_features(self, sample_weather_data):
        """Test temporal feature creation."""
        engineer = FeatureEngineer()
        
        result = engineer.create_temporal_features(sample_weather_data)
        
        # Should have temporal features
        temporal_features = [col for col in result.columns 
                           if col in ["hour", "month", "day_of_week"]]
        assert len(temporal_features) > 0


class TestDataQualityMonitor:
    """Test data quality monitoring."""
    
    def test_assess_data_quality_good_data(self, sample_weather_data):
        """Test quality assessment on good data."""
        monitor = DataQualityMonitor(alerting_enabled=False)
        
        metrics = monitor.assess_data_quality(sample_weather_data)
        
        assert metrics.total_records == len(sample_weather_data)
        assert metrics.completeness_score >= 0.8  # Should be high for good data
        assert metrics.quality_score >= 0.8
        
    def test_assess_data_quality_poor_data(self, sample_weather_data):
        """Test quality assessment on poor data."""
        monitor = DataQualityMonitor(alerting_enabled=False)
        
        # Introduce quality issues
        poor_data = sample_weather_data.copy()
        poor_data.loc[0:10, 'temperature'] = np.nan  # Missing values
        poor_data.loc[20:22, 'temperature'] = -999   # Invalid values
        
        metrics = monitor.assess_data_quality(poor_data)
        
        assert metrics.missing_temperature > 0
        assert metrics.quality_score < 1.0  # Should be lower due to issues
        
    def test_check_data_freshness(self, sample_weather_data):
        """Test data freshness checking."""
        monitor = DataQualityMonitor()
        
        # Update timestamps to be recent
        fresh_data = sample_weather_data.copy()
        fresh_data['timestamp'] = pd.date_range(
            datetime.now() - timedelta(hours=2),
            periods=len(fresh_data),
            freq='1min'
        )
        
        freshness = monitor.check_data_freshness(fresh_data)
        
        assert "is_fresh" in freshness
        assert "hours_since_latest" in freshness
        assert freshness["hours_since_latest"] >= 0
        
    def test_validate_schema(self, sample_weather_data):
        """Test schema validation."""
        monitor = DataQualityMonitor()
        
        validation = monitor.validate_schema(sample_weather_data)
        
        assert "schema_valid" in validation
        assert "missing_columns" in validation
        assert "extra_columns" in validation
        assert isinstance(validation["schema_valid"], bool)


class TestPerformanceComparator:
    """Test performance comparison functionality."""
    
    def test_benchmark_operation(self, sample_weather_data):
        """Test benchmarking a single operation."""
        comparator = PerformanceComparator()
        
        # Simple aggregation benchmark
        pandas_func = lambda df: df.groupby('city')['temperature'].mean()
        polars_func = lambda df: df.group_by('city').agg([
            pl.col('temperature').mean().alias('temp_mean')
        ])
        
        result = comparator.benchmark_operation(
            "test_aggregation",
            pandas_func,
            polars_func,
            sample_weather_data,
            iterations=2
        )
        
        assert "pandas" in result
        assert "polars" in result
        assert "operation" in result
        assert result["operation"] == "test_aggregation"
        
    def test_comprehensive_benchmark(self, sample_weather_data):
        """Test comprehensive benchmarking."""
        comparator = PerformanceComparator()
        
        results = comparator.comprehensive_benchmark(sample_weather_data)
        
        assert "benchmarks" in results
        assert "summary" in results
        assert "total_benchmarks" in results["summary"]
        assert len(results["benchmarks"]) > 0


@pytest.mark.asyncio
class TestCacheManager:
    """Test cache management functionality."""
    
    async def test_cache_weather_data(self):
        """Test weather data caching."""
        cache_manager = CacheManager()
        
        test_data = {
            "temperature": 22.5,
            "humidity": 65,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache data
        success = await cache_manager.cache_weather_data(
            location="test_city",
            provider="test_provider", 
            data=test_data
        )
        assert success
        
        # Retrieve data
        cached_data = await cache_manager.get_weather_data(
            location="test_city",
            provider="test_provider"
        )
        
        assert cached_data is not None
        assert cached_data["temperature"] == 22.5
        
    async def test_cache_processed_data(self):
        """Test processed data caching."""
        cache_manager = CacheManager()
        
        test_result = {"analysis": "positive_trend", "confidence": 0.85}
        
        # Cache processed result
        success = await cache_manager.cache_processed_data(
            processing_key="trend_analysis",
            result=test_result,
            parameters={"window": 24}
        )
        assert success
        
        # Retrieve processed result
        cached_result = await cache_manager.get_processed_data(
            processing_key="trend_analysis",
            parameters={"window": 24}
        )
        
        assert cached_result is not None
        assert cached_result["confidence"] == 0.85
        
    async def test_cache_stats(self):
        """Test cache statistics."""
        cache_manager = CacheManager()
        
        # Perform some cache operations
        await cache_manager.cache_weather_data("city1", "provider1", {"temp": 20})
        await cache_manager.get_weather_data("city1", "provider1")  # Hit
        await cache_manager.get_weather_data("city2", "provider2")  # Miss
        
        stats = cache_manager.get_cache_stats()
        
        assert "hit_rate" in stats
        assert "total_requests" in stats
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_complete_analysis_workflow(self, sample_weather_data):
        """Test a complete analysis workflow using multiple components."""
        
        # 1. Feature engineering
        engineer = FeatureEngineer()
        enhanced_data = engineer.create_rolling_features(sample_weather_data)
        enhanced_data = engineer.create_temporal_features(enhanced_data)
        
        # 2. Quality assessment  
        monitor = DataQualityMonitor(alerting_enabled=False)
        quality_metrics = monitor.assess_data_quality(enhanced_data)
        
        # 3. Time series analysis
        analyzer = TimeSeriesAnalyzer()
        trends = analyzer.detect_trends(enhanced_data, column="temperature")
        
        # 4. Geospatial analysis
        geo_analyzer = GeospatialAnalyzer()
        clusters = geo_analyzer.cluster_weather_patterns(
            enhanced_data,
            features=["temperature", "humidity"],
            n_clusters=2
        )
        
        # Verify workflow completed successfully
        assert quality_metrics.total_records > 0
        assert "linear_trend" in trends
        assert "n_clusters" in clusters
        assert len(enhanced_data.columns) > len(sample_weather_data.columns)
        
    def test_pandas_polars_compatibility(self, sample_weather_data):
        """Test that components work with both pandas and Polars."""
        
        # Convert to Polars
        polars_data = pl.from_pandas(sample_weather_data)
        
        # Test feature engineering with both
        engineer = FeatureEngineer()
        
        pandas_result = engineer.create_rolling_features(sample_weather_data)
        polars_result = engineer.create_rolling_features(polars_data)
        
        # Both should succeed and add similar features
        assert len(pandas_result.columns) > len(sample_weather_data.columns)
        assert len(polars_result.columns) > len(polars_data.columns)
        
        # Test time series analysis
        analyzer = TimeSeriesAnalyzer()
        
        pandas_trends = analyzer.detect_trends(sample_weather_data)
        polars_trends = analyzer.detect_trends(polars_data)
        
        # Both should produce results
        assert "linear_trend" in pandas_trends
        assert "linear_trend" in polars_trends
