"""Simplified integration tests for API functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from abc import ABC, abstractmethod

from src.weather_pipeline.api.base import BaseWeatherClient, WeatherAPIError
from src.weather_pipeline.core.resilience import RateLimiterConfig, RetryConfig
from src.weather_pipeline.models.weather import WeatherProvider


class TestAPIIntegration:
    """Test API integration functionality."""

    def test_weather_api_error(self):
        """Test WeatherAPIError exception."""
        error = WeatherAPIError("Test error message", WeatherProvider.WEATHERAPI)
        assert str(error) == "Test error message"
        assert error.provider == WeatherProvider.WEATHERAPI

    def test_rate_limiter_config(self):
        """Test RateLimiterConfig creation."""
        config = RateLimiterConfig(max_requests=10, window_size=60.0)
        
        assert config.max_requests == 10
        assert config.window_size == 60.0

    def test_retry_config(self):
        """Test RetryConfig creation."""
        config = RetryConfig(max_retries=3, base_delay=1.0)
        
        assert config.max_retries == 3
        assert config.base_delay == 1.0

    def test_weather_provider_enum(self):
        """Test WeatherProvider enum usage."""
        provider = WeatherProvider.WEATHERAPI
        assert provider == "weatherapi"
        
        provider = WeatherProvider.OPENWEATHER
        assert provider == "openweather"
