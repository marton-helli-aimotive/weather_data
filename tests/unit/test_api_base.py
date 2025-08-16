"""Unit tests for API client base classes and utilities."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone
import aiohttp

from weather_pipeline.api.base import BaseWeatherClient, WeatherAPIError
from weather_pipeline.api.factory import WeatherClientFactory
from weather_pipeline.models import WeatherProvider, Coordinates
from weather_pipeline.core.resilience import RateLimiterConfig, RetryConfig
from weather_pipeline.core.circuit_breaker import CircuitBreakerConfig


class MockWeatherClient(BaseWeatherClient):
    """Mock implementation for testing abstract base class."""
    
    def __init__(self, **kwargs):
        super().__init__(WeatherProvider.WEATHERAPI, **kwargs)
    
    async def get_current_weather(self, coordinates, city, country=None):
        """Mock implementation."""
        return []
    
    async def get_forecast(self, coordinates, city, country=None, days=5):
        """Mock implementation."""
        return []
    
    async def get_raw_response(self, coordinates, city, country=None, endpoint="current"):
        """Mock implementation."""
        return {}
    
    def _build_url(self, endpoint, **params):
        """Mock implementation."""
        return f"https://mock.api/{endpoint}"
    
    def _parse_response(self, response_data, coordinates, city, country=None):
        """Mock implementation."""
        return {}


class TestBaseWeatherClient:
    """Test BaseWeatherClient abstract base class."""

    def test_client_initialization(self):
        """Test client initialization with default parameters."""
        client = MockWeatherClient(api_key="test_key")
        
        assert client.provider == WeatherProvider.WEATHERAPI
        assert client.api_key == "test_key"
        assert client.timeout == 30.0
        assert client.resilient_client is not None
        assert client.circuit_breaker is not None

    def test_client_initialization_with_custom_config(self):
        """Test client initialization with custom configuration."""
        rate_limiter_config = RateLimiterConfig(max_tokens=5, refill_rate=1.0)
        retry_config = RetryConfig(max_attempts=5, backoff_factor=0.5)
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=3,
            reset_timeout=60,
            half_open_max_calls=2
        )
        
        client = MockWeatherClient(
            api_key="test_key",
            timeout=60.0,
            rate_limiter_config=rate_limiter_config,
            retry_config=retry_config,
            circuit_breaker_config=circuit_breaker_config
        )
        
        assert client.timeout == 60.0
        assert client.api_key == "test_key"

    def test_api_error_handling(self):
        """Test API error handling."""
        client = MockWeatherClient(api_key="test_key")
        
        # Test authentication error
        with pytest.raises(WeatherAPIError) as exc_info:
            client._handle_api_error(401, "Unauthorized")
        assert "authentication" in str(exc_info.value).lower()
        
        # Test rate limit error
        with pytest.raises(WeatherAPIError) as exc_info:
            client._handle_api_error(429, "Too Many Requests")
        assert "rate limit" in str(exc_info.value).lower()
        
        # Test not found error
        with pytest.raises(WeatherAPIError) as exc_info:
            client._handle_api_error(404, "Not Found")
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        client = MockWeatherClient(api_key="test_key")
        
        # Mock successful weather data retrieval
        with patch.object(client, 'get_current_weather', return_value=[]):
            result = await client.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        client = MockWeatherClient(api_key="test_key")
        
        # Mock failed weather data retrieval
        with patch.object(client, 'get_current_weather', side_effect=Exception("Connection error")):
            result = await client.health_check()
            assert result is False

    def test_circuit_breaker_state(self):
        """Test circuit breaker state retrieval."""
        client = MockWeatherClient(api_key="test_key")
        
        state = client.get_circuit_breaker_state()
        assert state in ["closed", "open", "half_open"]

    def test_metrics_retrieval(self):
        """Test client metrics retrieval."""
        client = MockWeatherClient(api_key="test_key")
        
        metrics = client.get_metrics()
        
        assert "provider" in metrics
        assert "circuit_breaker_state" in metrics
        assert "circuit_breaker_metrics" in metrics
        assert "available_tokens" in metrics
        assert metrics["provider"] == "weatherapi"

    def test_url_building(self):
        """Test URL building."""
        client = MockWeatherClient(api_key="test_key")
        
        url = client._build_url("current")
        assert url == "https://mock.api/current"

    def test_missing_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        
        class IncompleteClient(BaseWeatherClient):
            def __init__(self):
                super().__init__(WeatherProvider.WEATHERAPI)
        
        with pytest.raises(TypeError):
            IncompleteClient()


class TestWeatherClientFactory:
    """Test WeatherClientFactory."""

    def test_get_supported_providers(self):
        """Test getting supported providers."""
        providers = WeatherClientFactory.get_supported_providers()
        
        assert isinstance(providers, list)
        assert len(providers) > 0
        assert all(isinstance(p, WeatherProvider) for p in providers)

    def test_create_client_weatherapi(self, test_settings):
        """Test creating WeatherAPI client."""
        client = WeatherClientFactory.create_client(
            WeatherProvider.WEATHERAPI,
            api_key="test_key"
        )
        
        assert client.provider == WeatherProvider.WEATHERAPI
        assert client.api_key == "test_key"

    def test_create_client_openweather(self, test_settings):
        """Test creating OpenWeatherMap client."""
        client = WeatherClientFactory.create_client(
            WeatherProvider.OPENWEATHER,
            api_key="test_key"
        )
        
        assert client.provider == WeatherProvider.OPENWEATHER
        assert client.api_key == "test_key"

    def test_create_client_7timer(self, test_settings):
        """Test creating 7timer client."""
        client = WeatherClientFactory.create_client(
            WeatherProvider.SEVEN_TIMER
        )
        
        assert client.provider == WeatherProvider.SEVEN_TIMER

    def test_create_client_with_configs(self, test_settings):
        """Test creating client with custom configurations."""
        rate_limiter_config = RateLimiterConfig(max_tokens=5, refill_rate=1.0)
        
        client = WeatherClientFactory.create_client(
            WeatherProvider.WEATHERAPI,
            api_key="test_key",
            rate_limiter_config=rate_limiter_config
        )
        
        assert client.provider == WeatherProvider.WEATHERAPI

    def test_create_client_missing_api_key(self, test_settings):
        """Test creating client without required API key."""
        with pytest.raises(TypeError):
            WeatherClientFactory.create_client(
                WeatherProvider.WEATHERAPI
                # Missing api_key
            )

    def test_create_client_invalid_provider(self, test_settings):
        """Test creating client with invalid provider."""
        with pytest.raises(ValueError):
            WeatherClientFactory.create_client(
                "invalid_provider",
                api_key="test_key"
            )

    def test_create_multi_provider_client(self, test_settings):
        """Test creating multi-provider client."""
        providers = [WeatherProvider.WEATHERAPI, WeatherProvider.SEVEN_TIMER]
        api_keys = {WeatherProvider.WEATHERAPI: "test_key1"}
        
        multi_client = WeatherClientFactory.create_multi_provider_client(
            providers=providers,
            api_keys=api_keys
        )
        
        assert len(multi_client.clients) == 2
        assert WeatherProvider.WEATHERAPI in multi_client.clients
        assert WeatherProvider.SEVEN_TIMER in multi_client.clients

    def test_register_custom_client(self):
        """Test registering a custom client type."""
        # Save original registry
        original_registry = WeatherClientFactory._client_registry.copy()
        
        try:
            # Register custom client
            WeatherClientFactory.register_client(
                WeatherProvider.WEATHERAPI,
                MockWeatherClient
            )
            
            # Verify registration
            assert WeatherClientFactory._client_registry[WeatherProvider.WEATHERAPI] == MockWeatherClient
            
        finally:
            # Restore original registry
            WeatherClientFactory._client_registry = original_registry


class TestAPIClientExceptions:
    """Test API client exception handling."""

    def test_weather_api_error_types(self):
        """Test different types of WeatherAPIError."""
        from weather_pipeline.api.base import (
            WeatherAPIAuthenticationError,
            WeatherAPIRateLimitError,
            WeatherAPINotFoundError,
            WeatherAPITimeoutError
        )
        
        # Test inheritance
        assert issubclass(WeatherAPIAuthenticationError, WeatherAPIError)
        assert issubclass(WeatherAPIRateLimitError, WeatherAPIError)
        assert issubclass(WeatherAPINotFoundError, WeatherAPIError)
        assert issubclass(WeatherAPITimeoutError, WeatherAPIError)
        
        # Test instantiation with required provider parameter
        auth_error = WeatherAPIAuthenticationError("Invalid API key", WeatherProvider.WEATHERAPI)
        assert str(auth_error) == "Invalid API key"
        
        rate_error = WeatherAPIRateLimitError("Rate limit exceeded", WeatherProvider.WEATHERAPI)
        assert str(rate_error) == "Rate limit exceeded"

    def test_error_context_preservation(self):
        """Test that error context is preserved."""
        try:
            raise WeatherAPIError("Original error", WeatherProvider.WEATHERAPI, 500)
        except WeatherAPIError as e:
            assert str(e) == "Original error"
            assert e.provider == WeatherProvider.WEATHERAPI
            assert e.status_code == 500
