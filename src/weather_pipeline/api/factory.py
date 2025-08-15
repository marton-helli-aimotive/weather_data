"""Factory pattern for weather API clients."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from .base import BaseWeatherClient
from .seven_timer import SevenTimerClient
from .weatherapi import WeatherAPIClient
from .openweather import OpenWeatherMapClient
from ..core.circuit_breaker import CircuitBreakerConfig
from ..core.resilience import RateLimiterConfig, RetryConfig
from ..models.weather import WeatherProvider


class WeatherClientFactory:
    """Factory for creating weather API clients."""
    
    # Registry of available clients
    _client_registry: Dict[WeatherProvider, Type[BaseWeatherClient]] = {
        WeatherProvider.SEVEN_TIMER: SevenTimerClient,
        WeatherProvider.WEATHERAPI: WeatherAPIClient,
        WeatherProvider.OPENWEATHER: OpenWeatherMapClient,
    }
    
    @classmethod
    def create_client(
        self,
        provider: WeatherProvider,
        api_key: Optional[str] = None,
        rate_limiter_config: Optional[RateLimiterConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        timeout: float = 30.0,
        **kwargs
    ) -> BaseWeatherClient:
        """Create a weather API client for the specified provider."""
        
        if provider not in self._client_registry:
            raise ValueError(f"Unsupported weather provider: {provider}")
        
        client_class = self._client_registry[provider]
        
        # Build client arguments
        client_args = {
            "rate_limiter_config": rate_limiter_config,
            "retry_config": retry_config,
            "circuit_breaker_config": circuit_breaker_config,
            "timeout": timeout,
            **kwargs
        }
        
        # Add API key if provided
        if api_key:
            client_args["api_key"] = api_key
        
        return client_class(**client_args)
    
    @classmethod
    def create_multi_provider_client(
        self,
        providers: List[WeatherProvider],
        api_keys: Optional[Dict[WeatherProvider, str]] = None,
        rate_limiter_configs: Optional[Dict[WeatherProvider, RateLimiterConfig]] = None,
        retry_configs: Optional[Dict[WeatherProvider, RetryConfig]] = None,
        circuit_breaker_configs: Optional[Dict[WeatherProvider, CircuitBreakerConfig]] = None,
        timeout: float = 30.0,
        **kwargs
    ) -> "MultiProviderWeatherClient":
        """Create a multi-provider client with fallback capabilities."""
        
        api_keys = api_keys or {}
        rate_limiter_configs = rate_limiter_configs or {}
        retry_configs = retry_configs or {}
        circuit_breaker_configs = circuit_breaker_configs or {}
        
        clients = {}
        for provider in providers:
            try:
                clients[provider] = self.create_client(
                    provider=provider,
                    api_key=api_keys.get(provider),
                    rate_limiter_config=rate_limiter_configs.get(provider),
                    retry_config=retry_configs.get(provider),
                    circuit_breaker_config=circuit_breaker_configs.get(provider),
                    timeout=timeout,
                    **kwargs
                )
            except Exception as e:
                # Log warning but continue with other providers
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to create client for {provider}: {e}")
        
        if not clients:
            raise ValueError("No weather clients could be created")
        
        return MultiProviderWeatherClient(clients, providers)
    
    @classmethod
    def get_supported_providers(self) -> List[WeatherProvider]:
        """Get list of supported weather providers."""
        return list(self._client_registry.keys())
    
    @classmethod
    def register_client(self, provider: WeatherProvider, client_class: Type[BaseWeatherClient]) -> None:
        """Register a new client type (for extensibility)."""
        self._client_registry[provider] = client_class


class MultiProviderWeatherClient:
    """Client that manages multiple weather providers with automatic fallback."""
    
    def __init__(self, clients: Dict[WeatherProvider, BaseWeatherClient], priority_order: List[WeatherProvider]):
        self.clients = clients
        self.priority_order = [p for p in priority_order if p in clients]
        
        if not self.priority_order:
            raise ValueError("No valid providers in priority order")
        
        import logging
        self.logger = logging.getLogger(__name__)
    
    async def get_current_weather(self, coordinates, city: str, country: Optional[str] = None):
        """Get current weather with automatic provider fallback."""
        last_exception = None
        
        for provider in self.priority_order:
            client = self.clients[provider]
            try:
                self.logger.debug(f"Trying provider {provider} for current weather")
                result = await client.get_current_weather(coordinates, city, country)
                self.logger.info(f"Successfully retrieved current weather from {provider}")
                return result
            except Exception as e:
                self.logger.warning(f"Provider {provider} failed for current weather: {e}")
                last_exception = e
                continue
        
        # If all providers failed
        raise Exception(f"All weather providers failed. Last error: {last_exception}")
    
    async def get_forecast(self, coordinates, city: str, country: Optional[str] = None, days: int = 5):
        """Get weather forecast with automatic provider fallback."""
        last_exception = None
        
        for provider in self.priority_order:
            client = self.clients[provider]
            try:
                self.logger.debug(f"Trying provider {provider} for forecast")
                result = await client.get_forecast(coordinates, city, country, days)
                self.logger.info(f"Successfully retrieved forecast from {provider}")
                return result
            except Exception as e:
                self.logger.warning(f"Provider {provider} failed for forecast: {e}")
                last_exception = e
                continue
        
        # If all providers failed
        raise Exception(f"All weather providers failed. Last error: {last_exception}")
    
    async def health_check_all(self) -> Dict[WeatherProvider, bool]:
        """Check health of all providers."""
        health_status = {}
        
        for provider, client in self.clients.items():
            try:
                health_status[provider] = await client.health_check()
            except Exception as e:
                self.logger.error(f"Health check failed for {provider}: {e}")
                health_status[provider] = False
        
        return health_status
    
    def get_all_metrics(self) -> Dict[WeatherProvider, dict]:
        """Get metrics from all providers."""
        return {provider: client.get_metrics() for provider, client in self.clients.items()}
    
    def get_available_providers(self) -> List[WeatherProvider]:
        """Get list of available providers."""
        return list(self.clients.keys())
