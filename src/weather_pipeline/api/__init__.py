"""Weather API client package."""

from .base import BaseWeatherClient, WeatherAPIError, WeatherAPITimeoutError, WeatherAPIRateLimitError, WeatherAPIAuthenticationError, WeatherAPINotFoundError
from .factory import WeatherClientFactory, MultiProviderWeatherClient
from .seven_timer import SevenTimerClient
from .weatherapi import WeatherAPIClient  
from .openweather import OpenWeatherMapClient

__all__ = [
    # Base classes
    "BaseWeatherClient",
    
    # Concrete clients
    "SevenTimerClient",
    "WeatherAPIClient",
    "OpenWeatherMapClient",
    
    # Factory and multi-provider
    "WeatherClientFactory",
    "MultiProviderWeatherClient",
    
    # Exceptions
    "WeatherAPIError",
    "WeatherAPITimeoutError",
    "WeatherAPIRateLimitError", 
    "WeatherAPIAuthenticationError",
    "WeatherAPINotFoundError",
]
