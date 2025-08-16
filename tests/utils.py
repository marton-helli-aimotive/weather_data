"""Test utilities and helpers."""

import pytest
import asyncio
import time
import functools
from typing import Dict, Any, List, Callable
from contextlib import contextmanager
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


class TestTimer:
    """Context manager for timing test execution."""
    
    def __init__(self, test_name: str, fail_threshold: float = None):
        self.test_name = test_name
        self.fail_threshold = fail_threshold
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.perf_counter() - self.start_time
        print(f"Test '{self.test_name}' completed in {self.duration:.4f} seconds")
        
        if self.fail_threshold and self.duration > self.fail_threshold:
            pytest.fail(f"Test '{self.test_name}' exceeded time threshold: {self.duration:.4f}s > {self.fail_threshold}s")


def timeout(seconds: float):
    """Decorator to add timeout to test functions."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Test timed out after {seconds} seconds")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(seconds))
                
                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)  # Cancel alarm
                    return result
                except TimeoutError:
                    signal.alarm(0)  # Cancel alarm
                    pytest.fail(f"Test timed out after {seconds} seconds")
            
            return sync_wrapper
    return decorator


def performance_test(max_duration: float = None, max_memory_mb: float = None):
    """Decorator to mark tests as performance tests with thresholds."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import psutil
            import gc
            
            # Clean up before test
            gc.collect()
            
            # Record initial state
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Record final state
                end_time = time.perf_counter()
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                duration = end_time - start_time
                memory_used = final_memory - initial_memory
                
                print(f"Performance metrics for {func.__name__}:")
                print(f"  Duration: {duration:.4f} seconds")
                print(f"  Memory used: {memory_used:.2f} MB")
                
                # Check thresholds
                if max_duration and duration > max_duration:
                    pytest.fail(f"Test exceeded duration threshold: {duration:.4f}s > {max_duration}s")
                
                if max_memory_mb and memory_used > max_memory_mb:
                    pytest.fail(f"Test exceeded memory threshold: {memory_used:.2f}MB > {max_memory_mb}MB")
            
            return result
        
        # Mark as performance test
        wrapper = pytest.mark.performance(wrapper)
        return wrapper
    
    return decorator


class MockAPIServer:
    """Mock API server for testing."""
    
    def __init__(self, responses: Dict[str, Any]):
        self.responses = responses
        self.call_count = 0
        self.call_history = []
    
    async def mock_request(self, url: str, **kwargs) -> Dict[str, Any]:
        """Mock API request."""
        self.call_count += 1
        self.call_history.append({
            'url': url,
            'kwargs': kwargs,
            'timestamp': datetime.now(timezone.utc)
        })
        
        # Determine response based on URL
        if 'weatherapi' in url:
            return self.responses.get('weatherapi', {})
        elif 'openweathermap' in url:
            return self.responses.get('openweather', {})
        elif '7timer' in url:
            return self.responses.get('7timer', {})
        else:
            return {'error': 'Unknown API'}
    
    def reset(self):
        """Reset call history."""
        self.call_count = 0
        self.call_history.clear()


class DataGenerator:
    """Utility for generating test data."""
    
    @staticmethod
    def weather_time_series(
        start_date: datetime,
        periods: int,
        freq: str = 'H',
        cities: List[str] = None,
        seed: int = 42
    ) -> pd.DataFrame:
        """Generate realistic weather time series data."""
        np.random.seed(seed)
        
        if cities is None:
            cities = ['TestCity']
        
        data = []
        timestamps = pd.date_range(start_date, periods=periods, freq=freq)
        
        for city_idx, city in enumerate(cities):
            base_temp = 20 + city_idx * 5  # Different base temperatures
            base_lat = 50 + city_idx * 5
            base_lon = 0 + city_idx * 10
            
            for i, timestamp in enumerate(timestamps):
                # Add seasonal and daily patterns
                seasonal_temp = 10 * np.sin(i * 2 * np.pi / (365 * 24))  # Yearly cycle
                daily_temp = 5 * np.sin(i * 2 * np.pi / 24)  # Daily cycle
                noise = np.random.normal(0, 2)
                
                temperature = base_temp + seasonal_temp + daily_temp + noise
                
                # Correlated humidity (inverse relationship with temperature)
                humidity = max(0, min(100, 80 - (temperature - base_temp) * 2 + np.random.normal(0, 10)))
                
                # Pressure with random walk
                pressure = 1013 + np.random.normal(0, 10)
                
                record = {
                    'timestamp': timestamp,
                    'temperature': temperature,
                    'humidity': int(humidity),
                    'pressure': pressure,
                    'wind_speed': max(0, np.random.exponential(5)),
                    'wind_direction': np.random.randint(0, 360),
                    'precipitation': max(0, np.random.exponential(0.5)),
                    'visibility': max(0.1, 10 + np.random.normal(0, 2)),
                    'cloud_cover': np.random.randint(0, 101),
                    'uv_index': max(0, min(15, 5 + np.random.normal(0, 2))),
                    'city': city,
                    'country': f'C{city_idx}',
                    'latitude': base_lat + np.random.normal(0, 0.1),
                    'longitude': base_lon + np.random.normal(0, 0.1),
                    'provider': np.random.choice(['weatherapi', 'openweather', '7timer']),
                    'is_forecast': np.random.choice([True, False]),
                    'confidence_score': np.random.uniform(0.8, 1.0)
                }
                data.append(record)
        
        return pd.DataFrame(data)
    
    @staticmethod
    def api_response(provider: str, city: str = "TestCity") -> Dict[str, Any]:
        """Generate mock API response."""
        responses = {
            'weatherapi': {
                "location": {
                    "name": city,
                    "region": "Test Region",
                    "country": "Test Country",
                    "lat": 50.0,
                    "lon": 10.0,
                    "tz_id": "UTC",
                    "localtime": "2024-01-01 12:00"
                },
                "current": {
                    "temp_c": 20.0,
                    "temp_f": 68.0,
                    "condition": {
                        "text": "Clear",
                        "icon": "clear.png",
                        "code": 1000
                    },
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
            },
            'openweather': {
                "coord": {"lon": 10.0, "lat": 50.0},
                "weather": [
                    {
                        "id": 800,
                        "main": "Clear",
                        "description": "clear sky",
                        "icon": "01d"
                    }
                ],
                "main": {
                    "temp": 293.15,  # 20°C in Kelvin
                    "feels_like": 293.15,
                    "temp_min": 291.15,
                    "temp_max": 295.15,
                    "pressure": 1013,
                    "humidity": 65
                },
                "wind": {
                    "speed": 2.5,
                    "deg": 180
                },
                "clouds": {
                    "all": 25
                },
                "name": city,
                "cod": 200
            },
            '7timer': {
                "product": "civil",
                "init": "2024010112",
                "dataseries": [
                    {
                        "timepoint": 3,
                        "cloudcover": 3,
                        "temp2m": 20,
                        "rh2m": 65,
                        "wind10m": {
                            "direction": "S",
                            "speed": 2
                        }
                    }
                ]
            }
        }
        
        return responses.get(provider, {})


class TestReporter:
    """Utility for test reporting and metrics collection."""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = []
        self.coverage_data = {}
    
    def record_test_result(self, test_name: str, status: str, duration: float, **kwargs):
        """Record test execution result."""
        result = {
            'test_name': test_name,
            'status': status,
            'duration': duration,
            'timestamp': datetime.now(timezone.utc),
            **kwargs
        }
        self.test_results.append(result)
    
    def record_performance_metric(self, metric_name: str, value: float, unit: str = ''):
        """Record performance metric."""
        metric = {
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'timestamp': datetime.now(timezone.utc)
        }
        self.performance_metrics.append(metric)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'passed'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'failed'])
        
        avg_duration = np.mean([r['duration'] for r in self.test_results]) if self.test_results else 0
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'avg_duration': avg_duration
            },
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'generated_at': datetime.now(timezone.utc)
        }
        
        return report


@contextmanager
def mock_environment(**env_vars):
    """Context manager for mocking environment variables."""
    with patch.dict('os.environ', env_vars):
        yield


@contextmanager
def mock_time(mock_datetime: datetime):
    """Context manager for mocking current time."""
    with patch('datetime.datetime') as mock_dt:
        mock_dt.now.return_value = mock_datetime
        mock_dt.utcnow.return_value = mock_datetime
        yield mock_dt


class AsyncTestCase:
    """Base class for async test cases."""
    
    def setUp(self):
        """Set up async test case."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up async test case."""
        self.loop.close()
    
    def run_async(self, coro):
        """Run async function in test."""
        return self.loop.run_until_complete(coro)


# Test data validation utilities

def validate_weather_data(df: pd.DataFrame) -> List[str]:
    """Validate weather data and return list of issues."""
    issues = []
    
    required_columns = ['timestamp', 'temperature', 'humidity', 'city']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
    
    if 'temperature' in df.columns:
        invalid_temps = df[(df['temperature'] < -100) | (df['temperature'] > 100)]
        if len(invalid_temps) > 0:
            issues.append(f"Invalid temperature values: {len(invalid_temps)} records")
    
    if 'humidity' in df.columns:
        invalid_humidity = df[(df['humidity'] < 0) | (df['humidity'] > 100)]
        if len(invalid_humidity) > 0:
            issues.append(f"Invalid humidity values: {len(invalid_humidity)} records")
    
    if 'timestamp' in df.columns:
        if df['timestamp'].isnull().any():
            issues.append("Missing timestamp values")
    
    return issues


def assert_weather_data_valid(df: pd.DataFrame):
    """Assert that weather data is valid."""
    issues = validate_weather_data(df)
    if issues:
        pytest.fail(f"Weather data validation failed: {'; '.join(issues)}")


# Performance assertion utilities

def assert_execution_time(func: Callable, max_seconds: float, *args, **kwargs):
    """Assert that function executes within time limit."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    duration = time.perf_counter() - start_time
    
    if duration > max_seconds:
        pytest.fail(f"Function {func.__name__} took {duration:.4f}s, expected < {max_seconds}s")
    
    return result


def assert_memory_usage(func: Callable, max_mb: float, *args, **kwargs):
    """Assert that function uses memory within limit."""
    import psutil
    import gc
    
    gc.collect()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    result = func(*args, **kwargs)
    
    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_used = final_memory - initial_memory
    
    if memory_used > max_mb:
        pytest.fail(f"Function {func.__name__} used {memory_used:.2f}MB, expected < {max_mb}MB")
    
    return result


# Global test reporter instance
test_reporter = TestReporter()
