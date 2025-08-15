"""
Example demonstrating Milestone 3 advanced data processing capabilities.

This script showcases:
1. Time series analysis
2. Geospatial analysis  
3. Feature engineering
4. Data quality monitoring
5. Performance comparison
6. Streaming processing
7. Cache management
"""

import asyncio
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import polars as pl

from weather_pipeline.processing import (
    TimeSeriesAnalyzer,
    GeospatialAnalyzer,
    FeatureEngineer,
    DataQualityMonitor,
    PerformanceComparator,
    StreamProcessor,
    CacheManager
)


def generate_sample_weather_data(n_records: int = 1000) -> pd.DataFrame:
    """Generate sample weather data for demonstration."""
    np.random.seed(42)
    
    # Generate timestamps (hourly data for ~40 days)
    start_date = datetime.now() - timedelta(days=40)
    timestamps = pd.date_range(start_date, periods=n_records, freq='H')
    
    # Cities with different climates
    cities = ["New York", "Miami", "Seattle", "Phoenix", "Denver"]
    city_list = np.random.choice(cities, n_records)
    
    # Generate realistic weather data with some patterns
    temperature = []
    humidity = []
    pressure = []
    latitude = []
    longitude = []
    
    city_coords = {
        "New York": (40.7128, -74.0060),
        "Miami": (25.7617, -80.1918), 
        "Seattle": (47.6062, -122.3321),
        "Phoenix": (33.4484, -112.0740),
        "Denver": (39.7392, -104.9903)
    }
    
    for i, city in enumerate(city_list):
        # Seasonal temperature variation
        day_of_year = timestamps[i].dayofyear
        seasonal_temp = 10 * np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Daily temperature variation
        hour_of_day = timestamps[i].hour
        daily_temp = 5 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/2)
        
        # Base temperature by city
        base_temps = {"New York": 15, "Miami": 25, "Seattle": 12, "Phoenix": 30, "Denver": 10}
        base_temp = base_temps[city]
        
        temp = base_temp + seasonal_temp + daily_temp + np.random.normal(0, 3)
        temperature.append(temp)
        
        # Humidity (inversely correlated with temperature)
        hum = max(20, min(95, 80 - (temp - 20) * 1.5 + np.random.normal(0, 10)))
        humidity.append(hum)
        
        # Pressure (with some weather patterns)
        press = 1013 + np.random.normal(0, 15) + np.sin(i * 0.1) * 5
        pressure.append(press)
        
        # Coordinates
        lat, lon = city_coords[city]
        latitude.append(lat + np.random.normal(0, 0.1))  # Small variation
        longitude.append(lon + np.random.normal(0, 0.1))
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'city': city_list,
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'latitude': latitude,
        'longitude': longitude,
        'wind_speed': np.random.exponential(5, n_records),
        'precipitation': np.random.exponential(0.5, n_records),
        'cloud_cover': np.random.randint(0, 101, n_records)
    })


async def demonstrate_time_series_analysis():
    """Demonstrate time series analysis capabilities."""
    print("\n=== TIME SERIES ANALYSIS ===")
    
    # Generate sample data
    data = generate_sample_weather_data(2000)
    analyzer = TimeSeriesAnalyzer()
    
    # Trend detection
    print("1. Trend Detection:")
    trends = analyzer.detect_trends(data, column="temperature")
    print(f"   Linear trend slope: {trends['linear_trend']['slope']:.4f}")
    print(f"   Trend direction: {trends['linear_trend']['trend_direction']}")
    print(f"   R-squared: {trends['linear_trend']['r_squared']:.4f}")
    print(f"   Is stationary: {trends['stationarity']['is_stationary']}")
    
    # Seasonality analysis
    print("\n2. Seasonality Analysis:")
    seasonality = analyzer.analyze_seasonality(data, column="temperature")
    if "decomposition" in seasonality:
        print(f"   Seasonal strength: {seasonality['decomposition']['seasonal_strength']:.4f}")
        if "monthly_pattern" in seasonality:
            print(f"   Peak month: {seasonality['monthly_pattern']['peak_month']}")
            print(f"   Seasonal amplitude: {seasonality['monthly_pattern']['seasonal_amplitude']:.2f}°C")
    
    # Anomaly detection
    print("\n3. Anomaly Detection:")
    anomalies = analyzer.detect_anomalies(data, column="temperature", method="zscore")
    print(f"   Total anomalies detected: {anomalies['total_anomalies']}")
    print(f"   Anomaly rate: {anomalies['anomaly_rate']:.2%}")
    
    # Simple forecasting
    print("\n4. Simple Forecasting:")
    forecast = analyzer.forecast_simple(data, column="temperature", steps=24)
    if "forecast_values" in forecast:
        print(f"   24-hour forecast range: {min(forecast['forecast_values']):.1f}°C to {max(forecast['forecast_values']):.1f}°C")
        print(f"   Last observed: {forecast['last_observed']:.1f}°C")


async def demonstrate_geospatial_analysis():
    """Demonstrate geospatial analysis capabilities."""
    print("\n=== GEOSPATIAL ANALYSIS ===")
    
    data = generate_sample_weather_data(500)
    analyzer = GeospatialAnalyzer()
    
    # Weather pattern clustering
    print("1. Weather Pattern Clustering:")
    clusters = analyzer.cluster_weather_patterns(
        data, 
        features=["temperature", "humidity", "pressure", "latitude", "longitude"],
        method="kmeans",
        n_clusters=3
    )
    if "n_clusters" in clusters:
        print(f"   Number of clusters: {clusters['n_clusters']}")
        print(f"   Data points: {clusters['data_points']}")
        print(f"   Features used: {len(clusters['features_used'])}")
    
    # Distance calculations
    print("\n2. Distance Calculations:")
    ny_coords = (40.7128, -74.0060)  # New York
    miami_coords = (25.7617, -80.1918)  # Miami
    distance = analyzer.calculate_distances(ny_coords, miami_coords)
    print(f"   Distance NY to Miami: {distance:.0f} km")
    
    # Find nearest stations
    print("\n3. Nearest Station Search:")
    target_point = (41.0, -75.0)  # Point near NY
    nearest = analyzer.find_nearest_stations(
        target_point, 
        data[['city', 'latitude', 'longitude']].drop_duplicates(),
        n_nearest=3
    )
    if "nearest_stations" in nearest:
        print(f"   Found {len(nearest['nearest_stations'])} nearest stations")
        print(f"   Average distance: {nearest['mean_distance']:.1f} km")


async def demonstrate_feature_engineering():
    """Demonstrate feature engineering capabilities."""
    print("\n=== FEATURE ENGINEERING ===")
    
    data = generate_sample_weather_data(1000)
    engineer = FeatureEngineer()
    
    original_columns = len(data.columns)
    
    # Rolling features
    print("1. Creating Rolling Features...")
    data_with_rolling = engineer.create_rolling_features(
        data, 
        columns=["temperature", "humidity"],
        windows=[6, 12, 24],
        operations=["mean", "std", "min", "max"]
    )
    print(f"   Added {len(data_with_rolling.columns) - original_columns} rolling features")
    
    # Lag features
    print("\n2. Creating Lag Features...")
    data_with_lags = engineer.create_lag_features(
        data_with_rolling,
        columns=["temperature", "pressure"],
        lags=[1, 3, 6, 12]
    )
    print(f"   Total columns after lags: {len(data_with_lags.columns)}")
    
    # Derived metrics
    print("\n3. Creating Derived Metrics...")
    data_with_derived = engineer.create_derived_metrics(data_with_lags)
    derived_features = [col for col in data_with_derived.columns 
                       if col in ["heat_index", "wind_chill", "dew_point", "pressure_tendency"]]
    print(f"   Derived features: {derived_features}")
    
    # Temporal features
    print("\n4. Creating Temporal Features...")
    data_final = engineer.create_temporal_features(data_with_derived)
    temporal_features = [col for col in data_final.columns 
                        if col in ["hour", "month", "season", "hour_sin", "hour_cos"]]
    print(f"   Temporal features: {temporal_features}")
    print(f"   Final dataset: {len(data_final.columns)} columns, {len(data_final)} rows")


async def demonstrate_data_quality():
    """Demonstrate data quality monitoring."""
    print("\n=== DATA QUALITY MONITORING ===")
    
    # Create data with some quality issues
    data = generate_sample_weather_data(1000)
    
    # Introduce some quality issues for demonstration
    data.loc[50:55, 'temperature'] = np.nan  # Missing values
    data.loc[100:102, 'temperature'] = -999  # Invalid values
    data.loc[200, 'humidity'] = 150  # Out of range
    
    monitor = DataQualityMonitor()
    
    # Comprehensive quality assessment
    print("1. Data Quality Assessment:")
    metrics = monitor.assess_data_quality(data)
    print(f"   Total records: {metrics.total_records}")
    print(f"   Completeness score: {metrics.completeness_score:.2f}")
    print(f"   Overall quality score: {metrics.quality_score:.2f}")
    print(f"   Outliers detected: {metrics.outliers_detected}")
    
    # Freshness check
    print("\n2. Data Freshness Check:")
    freshness = monitor.check_data_freshness(data)
    if "is_fresh" in freshness:
        print(f"   Is data fresh: {freshness['is_fresh']}")
        print(f"   Hours since latest: {freshness['hours_since_latest']:.1f}")
        print(f"   Freshness status: {freshness['freshness_status']}")
    
    # Schema validation
    print("\n3. Schema Validation:")
    schema = monitor.validate_schema(data)
    print(f"   Schema valid: {schema['schema_valid']}")
    print(f"   Missing columns: {schema['missing_columns']}")
    print(f"   Extra columns: {schema['extra_columns']}")
    
    # Get recent alerts
    alerts = monitor.get_recent_alerts()
    print(f"\n4. Quality Alerts: {len(alerts)} alerts generated")


async def demonstrate_performance_comparison():
    """Demonstrate performance comparison between pandas and Polars."""
    print("\n=== PERFORMANCE COMPARISON ===")
    
    # Generate larger dataset for meaningful comparison
    data = generate_sample_weather_data(5000)
    comparator = PerformanceComparator()
    
    print("Running comprehensive performance benchmarks...")
    print("(This may take a few moments)")
    
    # Run comprehensive benchmark
    results = comparator.comprehensive_benchmark(data)
    
    print(f"\nBenchmark Results:")
    print(f"Total benchmarks: {results['summary']['total_benchmarks']}")
    print(f"Pandas wins: {results['summary']['pandas_wins']}")
    print(f"Polars wins: {results['summary']['polars_wins']}")
    print(f"Overall winner: {results['summary']['overall_winner']}")
    
    if results['summary']['average_speedup_ratio']:
        print(f"Average speedup ratio: {results['summary']['average_speedup_ratio']:.2f}x")
    
    # Show top performance differences
    print("\nTop Performance Differences:")
    benchmarks = results['benchmarks']
    for benchmark in benchmarks[:3]:  # Show first 3
        if benchmark.get('speedup_ratio'):
            print(f"   {benchmark['operation']}: {benchmark['speedup_ratio']:.2f}x speedup ({benchmark['winner']} wins)")


async def demonstrate_streaming_processing():
    """Demonstrate streaming data processing."""
    print("\n=== STREAMING DATA PROCESSING ===")
    
    from weather_pipeline.processing.streaming import (
        StreamingPipeline, 
        RealTimeAnomalyDetector,
        RealTimeTrendDetector
    )
    
    # Create streaming pipeline
    pipeline = StreamingPipeline()
    await pipeline.setup_default_pipeline()
    
    print("1. Streaming Pipeline Setup:")
    status = pipeline.get_pipeline_status()
    print(f"   Processors: {status['processor_count']}")
    print(f"   Pipeline ready: {len(status['processors']) > 0}")
    
    # Simulate streaming data
    print("\n2. Simulating Streaming Data:")
    sample_data = generate_sample_weather_data(100)
    
    # Add data points to stream
    for i, row in sample_data.iterrows():
        data_point = row.to_dict()
        await pipeline.add_data(data_point)
        
        # Stop after adding enough data for demo
        if i >= 50:
            break
    
    # Get buffer stats
    buffer_stats = pipeline.get_pipeline_status()["buffer_stats"]
    print(f"   Buffer size: {buffer_stats['size']}")
    print(f"   Memory usage: {buffer_stats.get('memory_usage_mb', 0):.2f} MB")


async def demonstrate_cache_management():
    """Demonstrate cache management capabilities."""
    print("\n=== CACHE MANAGEMENT ===")
    
    cache_manager = CacheManager()
    
    # Cache some weather data
    print("1. Caching Weather Data:")
    sample_weather = {
        "temperature": 22.5,
        "humidity": 65,
        "pressure": 1013.2,
        "timestamp": datetime.now().isoformat()
    }
    
    success = await cache_manager.cache_weather_data(
        location="New York",
        provider="test",
        data=sample_weather
    )
    print(f"   Cache success: {success}")
    
    # Retrieve cached data
    cached_data = await cache_manager.get_weather_data(
        location="New York",
        provider="test"
    )
    print(f"   Retrieved from cache: {cached_data is not None}")
    
    # Cache processed data
    print("\n2. Caching Processed Data:")
    processed_result = {"analysis": "trend_up", "confidence": 0.85}
    await cache_manager.cache_processed_data(
        processing_key="trend_analysis",
        result=processed_result,
        parameters={"window": 24, "method": "linear"}
    )
    
    # Get cache statistics
    print("\n3. Cache Statistics:")
    stats = cache_manager.get_cache_stats()
    print(f"   Hit rate: {stats['hit_rate']:.2%}")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Cache hits: {stats['hits']}")
    print(f"   Cache misses: {stats['misses']}")


async def main():
    """Run all demonstrations."""
    print("Weather Data Pipeline - Advanced Processing Capabilities Demo")
    print("=" * 60)
    
    try:
        await demonstrate_time_series_analysis()
        await demonstrate_geospatial_analysis()
        await demonstrate_feature_engineering()
        await demonstrate_data_quality()
        await demonstrate_performance_comparison()
        await demonstrate_streaming_processing()
        await demonstrate_cache_management()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("Milestone 3 advanced processing capabilities are working correctly.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
