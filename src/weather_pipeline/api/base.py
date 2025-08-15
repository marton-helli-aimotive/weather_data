"""Abstract base class and interface for weather API providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from ..core.circuit_breaker import CircuitBreakerConfig, circuit_breaker_registry
from ..core.resilience import RateLimiterConfig, RetryConfig, ResilientAPIClient
from ..models.api_responses import APIResponse
from ..models.weather import Coordinates, WeatherDataPoint, WeatherProvider

logger = logging.getLogger(__name__)


class WeatherAPIError(Exception):
    """Base exception for weather API errors."""
    
    def __init__(self, message: str, provider: WeatherProvider, status_code: Optional[int] = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)


class WeatherAPITimeoutError(WeatherAPIError):
    """Raised when API request times out."""


class WeatherAPIRateLimitError(WeatherAPIError):
    """Raised when API rate limit is exceeded."""


class WeatherAPIAuthenticationError(WeatherAPIError):
    """Raised when API authentication fails."""


class WeatherAPINotFoundError(WeatherAPIError):
    """Raised when requested location is not found."""


class BaseWeatherClient(ABC):
    """Abstract base class for all weather API clients."""
    
    def __init__(
        self,
        provider: WeatherProvider,
        api_key: Optional[str] = None,
        rate_limiter_config: Optional[RateLimiterConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        timeout: float = 30.0,
    ):
        self.provider = provider
        self.api_key = api_key
        self.timeout = timeout
        
        # Set up resilience components
        self.resilient_client = ResilientAPIClient(
            name=f"{provider.value}_client",
            rate_limiter_config=rate_limiter_config,
            retry_config=retry_config
        )
        
        # Set up circuit breaker
        self.circuit_breaker = circuit_breaker_registry.get_breaker(
            name=f"{provider.value}_circuit_breaker",
            config=circuit_breaker_config
        )
        
        self.logger = logging.getLogger(f"{__name__}.{provider.value}")
    
    @abstractmethod
    async def get_current_weather(
        self,
        coordinates: Coordinates,
        city: str,
        country: Optional[str] = None
    ) -> List[WeatherDataPoint]:
        """Get current weather for a location."""
        pass
    
    @abstractmethod
    async def get_forecast(
        self,
        coordinates: Coordinates,
        city: str,
        country: Optional[str] = None,
        days: int = 5
    ) -> List[WeatherDataPoint]:
        """Get weather forecast for a location."""
        pass
    
    @abstractmethod
    async def get_raw_response(
        self,
        coordinates: Coordinates,
        city: str,
        country: Optional[str] = None,
        endpoint: str = "current"
    ) -> APIResponse:
        """Get raw API response for debugging/analysis."""
        pass
    
    @abstractmethod
    def _build_url(self, endpoint: str, **params) -> str:
        """Build API request URL."""
        pass
    
    @abstractmethod
    def _parse_response(self, response_data: dict, coordinates: Coordinates, city: str, country: Optional[str] = None) -> APIResponse:
        """Parse API response into standard format."""
        pass
    
    def _handle_api_error(self, status_code: int, response_text: str) -> None:
        """Handle API error responses."""
        if status_code == 401:
            raise WeatherAPIAuthenticationError(
                f"Authentication failed for {self.provider.value}",
                self.provider,
                status_code
            )
        elif status_code == 404:
            raise WeatherAPINotFoundError(
                f"Location not found for {self.provider.value}",
                self.provider,
                status_code
            )
        elif status_code == 429:
            raise WeatherAPIRateLimitError(
                f"Rate limit exceeded for {self.provider.value}",
                self.provider,
                status_code
            )
        else:
            raise WeatherAPIError(
                f"API error for {self.provider.value}: {response_text}",
                self.provider,
                status_code
            )
    
    async def health_check(self) -> bool:
        """Check if the API is accessible and responsive."""
        try:
            # Try a simple request to test coordinates (London)
            test_coords = Coordinates(latitude=51.5074, longitude=-0.1278)
            await self.get_current_weather(test_coords, "London", "UK")
            return True
        except Exception as e:
            self.logger.warning(f"Health check failed for {self.provider.value}: {e}")
            return False
    
    def get_circuit_breaker_state(self) -> str:
        """Get current circuit breaker state."""
        return self.circuit_breaker.get_state().value
    
    def get_metrics(self) -> dict:
        """Get client metrics."""
        return {
            "provider": self.provider.value,
            "circuit_breaker_state": self.circuit_breaker.get_state().value,
            "circuit_breaker_metrics": self.circuit_breaker.get_metrics().model_dump(),
            "available_tokens": self.resilient_client.rate_limiter.get_available_tokens(),
        }
