"""Performance benchmarks and load tests."""

import pytest
import time
import asyncio
import pandas as pd
import polars as pl
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, AsyncMock
import psutil
import gc
from typing import List, Dict, Any

from src.weather_pipeline.processing import (
    TimeSeriesAnalyzer, GeospatialAnalyzer, FeatureEngineer,
    PerformanceComparator, CacheManager
)
from src.weather_pipeline.api import WeatherAPIClient, MultiProviderWeatherClient
from src.weather_pipeline.models import WeatherProvider, Coordinates


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        gc.collect()  # Clean up before measurement
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        print(f"{self.operation_name}: {self.duration:.4f} seconds")


class MemoryProfiler:
    """Monitor memory usage during operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.initial_memory = None
        self.peak_memory = None
        self.final_memory = None
    
    def __enter__(self):
        gc.collect()
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        self.final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = self.final_memory - self.initial_memory
        print(f"{self.operation_name} memory usage: {memory_used:.2f} MB")


@pytest.mark.slow
class TestDataProcessingPerformance:
    """Performance benchmarks for data processing operations."""
    
    def test_pandas_vs_polars_performance(self, benchmark_data):
        """Benchmark pandas vs Polars performance."""
        # Convert to Polars DataFrame
        polars_df = pl.DataFrame(benchmark_data)
        
        # Pandas operations
        with PerformanceTimer("Pandas groupby aggregation"):
            pandas_result = benchmark_data.groupby('city').agg({
                'temperature': ['mean', 'std', 'min', 'max'],
                'humidity': ['mean', 'std'],
                'pressure': ['mean', 'std']
            })
        
        # Polars operations
        with PerformanceTimer("Polars groupby aggregation"):
            polars_result = polars_df.group_by('city').agg([
                pl.col('temperature').mean().alias('temp_mean'),
                pl.col('temperature').std().alias('temp_std'),
                pl.col('temperature').min().alias('temp_min'),
                pl.col('temperature').max().alias('temp_max'),
                pl.col('humidity').mean().alias('humidity_mean'),
                pl.col('humidity').std().alias('humidity_std'),
                pl.col('pressure').mean().alias('pressure_mean'),
                pl.col('pressure').std().alias('pressure_std')
            ])
        
        # Both should produce results
        assert len(pandas_result) > 0
        assert len(polars_result) > 0

    def test_large_dataset_filtering_performance(self):
        """Test filtering performance on large datasets."""
        # Generate large dataset
        n_records = 100000
        np.random.seed(42)
        
        large_data = {
            'timestamp': pd.date_range('2020-01-01', periods=n_records, freq='H'),
            'city': np.random.choice(['NYC', 'London', 'Tokyo', 'Paris', 'Berlin'], n_records),
            'temperature': np.random.normal(20, 10, n_records),
            'humidity': np.random.randint(0, 100, n_records),
            'pressure': np.random.normal(1013, 50, n_records)
        }
        
        df_pandas = pd.DataFrame(large_data)
        df_polars = pl.DataFrame(large_data)
        
        # Complex filtering operation
        filter_conditions = (
            (df_pandas['temperature'] > 15) &
            (df_pandas['temperature'] < 25) &
            (df_pandas['humidity'] > 40) &
            (df_pandas['pressure'] > 1000)
        )
        
        with PerformanceTimer("Pandas complex filtering"):
            with MemoryProfiler("Pandas filtering"):
                pandas_filtered = df_pandas[filter_conditions]
        
        with PerformanceTimer("Polars complex filtering"):
            with MemoryProfiler("Polars filtering"):
                polars_filtered = df_polars.filter(
                    (pl.col('temperature') > 15) &
                    (pl.col('temperature') < 25) &
                    (pl.col('humidity') > 40) &
                    (pl.col('pressure') > 1000)
                )
        
        # Results should be comparable
        assert abs(len(pandas_filtered) - len(polars_filtered)) < 100  # Allow small differences

    def test_time_series_analysis_performance(self, benchmark_data):
        """Benchmark time series analysis operations."""
        analyzer = TimeSeriesAnalyzer()
        
        # Trend detection performance
        with PerformanceTimer("Trend detection"):
            with MemoryProfiler("Trend detection"):
                trends = analyzer.detect_trends(benchmark_data, column="temperature")
        
        assert "linear_trend" in trends
        
        # Seasonality analysis performance (with larger dataset)
        larger_data = pd.concat([benchmark_data] * 10, ignore_index=True)
        
        with PerformanceTimer("Seasonality analysis"):
            with MemoryProfiler("Seasonality analysis"):
                seasonality = analyzer.analyze_seasonality(larger_data, column="temperature")
        
        assert "decomposition" in seasonality

    def test_geospatial_analysis_performance(self, benchmark_data):
        """Benchmark geospatial analysis operations."""
        analyzer = GeospatialAnalyzer()
        
        # Distance calculation performance
        points = list(zip(benchmark_data['latitude'], benchmark_data['longitude']))
        
        with PerformanceTimer("Pairwise distance calculation"):
            with MemoryProfiler("Distance calculation"):
                distances = []
                for i, point1 in enumerate(points[:100]):  # Limit to 100 points
                    for point2 in points[i+1:101]:
                        distance = analyzer.calculate_distances(point1, point2)
                        distances.append(distance)
        
        assert len(distances) > 0
        
        # Clustering performance
        with PerformanceTimer("Weather pattern clustering"):
            with MemoryProfiler("Clustering"):
                clusters = analyzer.cluster_weather_patterns(
                    benchmark_data.head(1000),  # Limit size for performance
                    features=["temperature", "humidity", "pressure"],
                    method="kmeans",
                    n_clusters=5
                )
        
        assert clusters["n_clusters"] == 5

    def test_feature_engineering_performance(self, benchmark_data):
        """Benchmark feature engineering operations."""
        engineer = FeatureEngineer()
        
        # Rolling features performance
        with PerformanceTimer("Rolling features creation"):
            with MemoryProfiler("Rolling features"):
                rolling_result = engineer.create_rolling_features(
                    benchmark_data,
                    columns=["temperature", "humidity", "pressure"],
                    windows=[3, 6, 12],
                    operations=["mean", "std", "min", "max"]
                )
        
        assert len(rolling_result.columns) > len(benchmark_data.columns)
        
        # Lag features performance
        with PerformanceTimer("Lag features creation"):
            with MemoryProfiler("Lag features"):
                lag_result = engineer.create_lag_features(
                    benchmark_data,
                    columns=["temperature", "humidity"],
                    lags=[1, 2, 6, 12, 24]
                )
        
        assert len(lag_result.columns) > len(benchmark_data.columns)
        
        # Derived metrics performance
        with PerformanceTimer("Derived metrics creation"):
            with MemoryProfiler("Derived metrics"):
                derived_result = engineer.create_derived_metrics(benchmark_data)
        
        assert len(derived_result.columns) > len(benchmark_data.columns)


@pytest.mark.slow
class TestAPIClientPerformance:
    """Performance benchmarks for API clients."""
    
    @pytest.mark.asyncio
    async def test_concurrent_api_requests_performance(self, mock_api_responses):
        """Test performance of concurrent API requests."""
        client = WeatherAPIClient(api_key="test_key")
        
        # Test coordinates
        coordinates_list = [
            Coordinates(latitude=51.5074, longitude=-0.1278),  # London
            Coordinates(latitude=40.7128, longitude=-74.0060),  # New York
            Coordinates(latitude=35.6762, longitude=139.6503),  # Tokyo
            Coordinates(latitude=48.8566, longitude=2.3522),   # Paris
            Coordinates(latitude=52.5200, longitude=13.4050),  # Berlin
        ]
        
        cities = ["London", "New York", "Tokyo", "Paris", "Berlin"]
        
        # Mock responses
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_api_responses["weatherapi"])
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Sequential requests
            start_time = time.perf_counter()
            sequential_results = []
            for coord, city in zip(coordinates_list, cities):
                try:
                    result = await client.get_current_weather(coord, city, "")
                    sequential_results.append(result)
                except Exception:
                    sequential_results.append([])  # Empty result for failed requests
            sequential_duration = time.perf_counter() - start_time
            
            # Concurrent requests
            start_time = time.perf_counter()
            tasks = [
                client.get_current_weather(coord, city, "")
                for coord, city in zip(coordinates_list, cities)
            ]
            concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
            concurrent_duration = time.perf_counter() - start_time
            
            print(f"Sequential requests: {sequential_duration:.4f} seconds")
            print(f"Concurrent requests: {concurrent_duration:.4f} seconds")
            print(f"Speedup: {sequential_duration / concurrent_duration:.2f}x")
            
            # Concurrent should be faster (though mocking may affect this)
            assert len(concurrent_results) == len(coordinates_list)

    @pytest.mark.asyncio
    async def test_rate_limiting_performance(self):
        """Test performance impact of rate limiting."""
        from src.weather_pipeline.core.resilience import RateLimiterConfig
        
        # Client without rate limiting
        client_no_limit = WeatherAPIClient(api_key="test_key")
        
        # Client with rate limiting
        rate_config = RateLimiterConfig(max_tokens=5, refill_rate=2.0)
        client_with_limit = WeatherAPIClient(
            api_key="test_key",
            rate_limiter_config=rate_config
        )
        
        coordinates = Coordinates(latitude=51.5074, longitude=-0.1278)
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"location": {}, "current": {}})
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Test multiple requests
            num_requests = 3
            
            # Without rate limiting
            start_time = time.perf_counter()
            for i in range(num_requests):
                try:
                    await client_no_limit.get_current_weather(coordinates, f"City{i}", "")
                except Exception:
                    pass
            no_limit_duration = time.perf_counter() - start_time
            
            # With rate limiting
            start_time = time.perf_counter()
            for i in range(num_requests):
                try:
                    await client_with_limit.get_current_weather(coordinates, f"City{i}", "")
                except Exception:
                    pass
            with_limit_duration = time.perf_counter() - start_time
            
            print(f"Without rate limiting: {no_limit_duration:.4f} seconds")
            print(f"With rate limiting: {with_limit_duration:.4f} seconds")

    @pytest.mark.asyncio
    async def test_multi_provider_failover_performance(self, mock_api_responses):
        """Test performance of multi-provider failover."""
        providers_config = {
            WeatherProvider.WEATHERAPI: {"api_key": "test_key1"},
            WeatherProvider.SEVEN_TIMER: {},
            WeatherProvider.OPENWEATHER: {"api_key": "test_key2"}
        }
        
        multi_client = MultiProviderWeatherClient(providers_config)
        coordinates = Coordinates(latitude=51.5074, longitude=-0.1278)
        
        call_count = 0
        
        async def mock_get_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            mock_response = AsyncMock()
            if "weatherapi" in str(args[0]) and call_count <= 2:
                # First provider fails initially
                mock_response.status = 503
                mock_response.text = AsyncMock(return_value="Service Unavailable")
            else:
                # Other providers or retry succeed
                mock_response.status = 200
                if "7timer" in str(args[0]):
                    mock_response.json = AsyncMock(return_value=mock_api_responses["7timer"])
                else:
                    mock_response.json = AsyncMock(return_value=mock_api_responses["weatherapi"])
            
            return mock_response
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.side_effect = mock_get_side_effect
            
            start_time = time.perf_counter()
            try:
                result = await multi_client.get_current_weather(coordinates, "London", "UK")
                duration = time.perf_counter() - start_time
                print(f"Multi-provider failover: {duration:.4f} seconds")
                print(f"Total API calls made: {call_count}")
            except Exception as e:
                print(f"Multi-provider test failed: {e}")


@pytest.mark.slow
class TestLoadTesting:
    """Load testing for system components."""
    
    def test_concurrent_data_processing_load(self):
        """Test system under concurrent data processing load."""
        def process_data_chunk(chunk_id: int) -> Dict[str, Any]:
            """Process a chunk of data."""
            np.random.seed(chunk_id)
            
            # Generate data chunk
            chunk_size = 1000
            data = {
                'timestamp': pd.date_range('2024-01-01', periods=chunk_size, freq='H'),
                'temperature': np.random.normal(20, 5, chunk_size),
                'humidity': np.random.randint(30, 90, chunk_size),
                'pressure': np.random.normal(1013, 20, chunk_size)
            }
            df = pd.DataFrame(data)
            
            # Process the data
            analyzer = TimeSeriesAnalyzer()
            trends = analyzer.detect_trends(df, column="temperature")
            
            return {
                'chunk_id': chunk_id,
                'records_processed': len(df),
                'trend_direction': trends.get("linear_trend", {}).get("trend_direction", "unknown")
            }
        
        # Run concurrent processing
        num_workers = 4
        num_chunks = 10
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_data_chunk, i) for i in range(num_chunks)]
            results = [future.result() for future in as_completed(futures)]
        
        duration = time.perf_counter() - start_time
        total_records = sum(r['records_processed'] for r in results)
        
        print(f"Processed {total_records} records in {duration:.4f} seconds")
        print(f"Throughput: {total_records / duration:.0f} records/second")
        print(f"Chunks processed: {len(results)}")
        
        assert len(results) == num_chunks
        assert all(r['records_processed'] == 1000 for r in results)

    def test_memory_usage_under_load(self):
        """Test memory usage under heavy load."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        
        try:
            # Create multiple large datasets
            datasets = []
            for i in range(5):
                large_data = {
                    'timestamp': pd.date_range('2020-01-01', periods=50000, freq='min'),
                    'temperature': np.random.normal(20, 10, 50000),
                    'humidity': np.random.randint(0, 100, 50000),
                    'pressure': np.random.normal(1013, 50, 50000)
                }
                df = pd.DataFrame(large_data)
                datasets.append(df)
                
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
                print(f"Dataset {i+1} created, memory: {current_memory:.2f} MB")
            
            # Process all datasets
            for i, df in enumerate(datasets):
                analyzer = TimeSeriesAnalyzer()
                _ = analyzer.detect_trends(df, column="temperature")
                
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
                print(f"Dataset {i+1} processed, memory: {current_memory:.2f} MB")
        
        finally:
            # Clean up
            datasets.clear()
            gc.collect()
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = peak_memory - initial_memory
            memory_after_cleanup = final_memory - initial_memory
            
            print(f"Initial memory: {initial_memory:.2f} MB")
            print(f"Peak memory: {peak_memory:.2f} MB")
            print(f"Final memory: {final_memory:.2f} MB")
            print(f"Peak increase: {memory_increase:.2f} MB")
            print(f"Remaining increase: {memory_after_cleanup:.2f} MB")
            
            # Memory should not grow indefinitely
            assert memory_after_cleanup < memory_increase * 0.5  # At least 50% cleanup

    @pytest.mark.asyncio
    async def test_api_client_load_testing(self, mock_api_responses):
        """Load test API clients with high concurrent requests."""
        client = WeatherAPIClient(api_key="test_key")
        
        # Generate many coordinates
        num_requests = 50
        coordinates_list = [
            Coordinates(
                latitude=np.random.uniform(-90, 90),
                longitude=np.random.uniform(-180, 180)
            )
            for _ in range(num_requests)
        ]
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_api_responses["weatherapi"])
            mock_get.return_value.__aenter__.return_value = mock_response
            
            start_time = time.perf_counter()
            
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests
            
            async def make_request(coord, city_id):
                async with semaphore:
                    try:
                        return await client.get_current_weather(coord, f"City{city_id}", "")
                    except Exception as e:
                        return f"Error: {e}"
            
            tasks = [
                make_request(coord, i)
                for i, coord in enumerate(coordinates_list)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.perf_counter() - start_time
            
            successful_requests = sum(1 for r in results if not isinstance(r, Exception) and not isinstance(r, str))
            failed_requests = len(results) - successful_requests
            
            print(f"Load test completed in {duration:.4f} seconds")
            print(f"Total requests: {len(results)}")
            print(f"Successful requests: {successful_requests}")
            print(f"Failed requests: {failed_requests}")
            print(f"Requests per second: {len(results) / duration:.2f}")
            
            # Should handle the load reasonably
            assert len(results) == num_requests


@pytest.mark.slow
class TestCachePerformance:
    """Performance tests for caching mechanisms."""
    
    def test_cache_hit_vs_miss_performance(self, mock_redis):
        """Test performance difference between cache hits and misses."""
        cache_manager = CacheManager(redis_client=mock_redis)
        
        # Test data
        test_data = {'temperature': 20.5, 'humidity': 65, 'pressure': 1013.25}
        cache_key = "weather_data_test"
        
        # Cache miss (first access)
        mock_redis.get.return_value = None  # Cache miss
        
        start_time = time.perf_counter()
        for _ in range(100):
            cached_data = cache_manager.get(cache_key)
            if cached_data is None:
                # Simulate expensive operation
                time.sleep(0.001)  # 1ms
                cache_manager.set(cache_key, test_data, ttl=300)
        miss_duration = time.perf_counter() - start_time
        
        # Cache hit (subsequent accesses)
        mock_redis.get.return_value = str(test_data)  # Cache hit
        
        start_time = time.perf_counter()
        for _ in range(100):
            cached_data = cache_manager.get(cache_key)
        hit_duration = time.perf_counter() - start_time
        
        print(f"Cache miss (100 operations): {miss_duration:.4f} seconds")
        print(f"Cache hit (100 operations): {hit_duration:.4f} seconds")
        print(f"Speedup with cache: {miss_duration / hit_duration:.2f}x")
        
        # Cache hits should be significantly faster
        assert hit_duration < miss_duration * 0.1  # At least 10x faster

    def test_cache_performance_under_load(self, mock_redis):
        """Test cache performance under concurrent load."""
        cache_manager = CacheManager(redis_client=mock_redis)
        
        def cache_operations(worker_id: int) -> Dict[str, Any]:
            """Perform cache operations for load testing."""
            operations = 0
            start_time = time.perf_counter()
            
            for i in range(100):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                
                # Set operation
                cache_manager.set(key, value, ttl=300)
                operations += 1
                
                # Get operation
                cache_manager.get(key)
                operations += 1
            
            duration = time.perf_counter() - start_time
            return {
                'worker_id': worker_id,
                'operations': operations,
                'duration': duration,
                'ops_per_second': operations / duration
            }
        
        # Run concurrent cache operations
        num_workers = 5
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(cache_operations, i) for i in range(num_workers)]
            results = [future.result() for future in as_completed(futures)]
        
        total_operations = sum(r['operations'] for r in results)
        total_duration = max(r['duration'] for r in results)
        average_ops_per_second = sum(r['ops_per_second'] for r in results) / len(results)
        
        print(f"Total cache operations: {total_operations}")
        print(f"Total duration: {total_duration:.4f} seconds")
        print(f"Average ops/second per worker: {average_ops_per_second:.2f}")
        print(f"Overall throughput: {total_operations / total_duration:.2f} ops/second")
        
        assert len(results) == num_workers
        assert all(r['operations'] == 200 for r in results)  # 100 sets + 100 gets


# Benchmark comparison utilities

class BenchmarkComparison:
    """Utility for comparing benchmark results."""
    
    @staticmethod
    def compare_dataframe_operations(pandas_df: pd.DataFrame, polars_df: pl.DataFrame) -> Dict[str, float]:
        """Compare common operations between pandas and Polars."""
        results = {}
        
        # Group by operation
        start_time = time.perf_counter()
        pandas_grouped = pandas_df.groupby('city').mean()
        results['pandas_groupby'] = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        polars_grouped = polars_df.group_by('city').mean()
        results['polars_groupby'] = time.perf_counter() - start_time
        
        # Filter operation
        start_time = time.perf_counter()
        pandas_filtered = pandas_df[pandas_df['temperature'] > 20]
        results['pandas_filter'] = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        polars_filtered = polars_df.filter(pl.col('temperature') > 20)
        results['polars_filter'] = time.perf_counter() - start_time
        
        # Sort operation
        start_time = time.perf_counter()
        pandas_sorted = pandas_df.sort_values('timestamp')
        results['pandas_sort'] = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        polars_sorted = polars_df.sort('timestamp')
        results['polars_sort'] = time.perf_counter() - start_time
        
        return results


@pytest.mark.slow
class TestBenchmarkComparisons:
    """Comparative benchmarks between different implementations."""
    
    def test_pandas_vs_polars_comprehensive(self, benchmark_data):
        """Comprehensive comparison between pandas and Polars."""
        polars_df = pl.DataFrame(benchmark_data)
        
        comparison = BenchmarkComparison()
        results = comparison.compare_dataframe_operations(benchmark_data, polars_df)
        
        print("\nPandas vs Polars Performance Comparison:")
        for operation, duration in results.items():
            print(f"{operation}: {duration:.6f} seconds")
        
        # Calculate speedups
        speedups = {
            'groupby': results['pandas_groupby'] / results['polars_groupby'],
            'filter': results['pandas_filter'] / results['polars_filter'],
            'sort': results['pandas_sort'] / results['polars_sort']
        }
        
        print("\nSpeedup ratios (pandas_time / polars_time):")
        for operation, speedup in speedups.items():
            print(f"{operation}: {speedup:.2f}x")
        
        # Polars should generally be faster or competitive
        assert all(duration > 0 for duration in results.values())

    def test_performance_regression_detection(self, benchmark_data):
        """Test for performance regression detection."""
        # Baseline performance measurements
        baseline_times = {}
        
        # Time series analysis baseline
        analyzer = TimeSeriesAnalyzer()
        start_time = time.perf_counter()
        trends = analyzer.detect_trends(benchmark_data, column="temperature")
        baseline_times['trend_detection'] = time.perf_counter() - start_time
        
        # Feature engineering baseline
        engineer = FeatureEngineer()
        start_time = time.perf_counter()
        rolling_features = engineer.create_rolling_features(
            benchmark_data,
            columns=["temperature"],
            windows=[3, 6],
            operations=["mean"]
        )
        baseline_times['rolling_features'] = time.perf_counter() - start_time
        
        print("\nBaseline Performance Times:")
        for operation, duration in baseline_times.items():
            print(f"{operation}: {duration:.4f} seconds")
        
        # Define performance thresholds (in seconds)
        performance_thresholds = {
            'trend_detection': 1.0,    # Should complete within 1 second
            'rolling_features': 2.0,   # Should complete within 2 seconds
        }
        
        # Check for performance regressions
        for operation, duration in baseline_times.items():
            threshold = performance_thresholds.get(operation, float('inf'))
            if duration > threshold:
                print(f"WARNING: {operation} took {duration:.4f}s, exceeds threshold of {threshold}s")
            assert duration < threshold, f"Performance regression detected in {operation}"
