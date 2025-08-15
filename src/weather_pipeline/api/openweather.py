"""OpenWeatherMap API client implementation."""

from __future__ import annotations

import aiohttp
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlencode

from .base import BaseWeatherClient
from ..models.api_responses import OpenWeatherMapResponse
from ..models.weather import Coordinates, WeatherDataPoint, WeatherProvider


class OpenWeatherMapClient(BaseWeatherClient):
    """Client for OpenWeatherMap API."""
    
    BASE_URL = "http://api.openweathermap.org/data/2.5"
    
    def __init__(self, api_key: str, **kwargs):
        if not api_key:
            raise ValueError("OpenWeatherMap requires an API key")
        super().__init__(provider=WeatherProvider.OPENWEATHER, api_key=api_key, **kwargs)
    
    async def get_current_weather(
        self,
        coordinates: Coordinates,
        city: str,
        country: Optional[str] = None
    ) -> List[WeatherDataPoint]:
        """Get current weather from OpenWeatherMap."""
        response = await self.get_raw_response(coordinates, city, country, "weather")
        return response.to_weather_points(city, country)
    
    async def get_forecast(
        self,
        coordinates: Coordinates,
        city: str,
        country: Optional[str] = None,
        days: int = 5
    ) -> List[WeatherDataPoint]:
        """Get weather forecast from OpenWeatherMap."""
        # OpenWeatherMap free tier provides 5-day forecast with 3-hour intervals
        response = await self.get_raw_response(coordinates, city, country, "forecast")
        
        # Note: This would need to be adapted for forecast response format
        # For now, returning current weather format
        return response.to_weather_points(city, country) if hasattr(response, 'to_weather_points') else []
    
    async def get_raw_response(
        self,
        coordinates: Coordinates,
        city: str,
        country: Optional[str] = None,
        endpoint: str = "weather",
        **params
    ) -> OpenWeatherMapResponse:
        """Get raw API response from OpenWeatherMap."""
        url = self._build_url(
            endpoint,
            appid=self.api_key,
            lat=coordinates.latitude,
            lon=coordinates.longitude,
            units="metric",  # Get temperature in Celsius
            **params
        )
        
        async def make_request():
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        self._handle_api_error(response.status, response_text)
                    
                    data = await response.json()
                    return self._parse_response(data, coordinates, city, country)
        
        # Execute with circuit breaker protection
        return await self.circuit_breaker.call(
            self.resilient_client.execute_with_resilience,
            make_request
        )
    
    def _build_url(self, endpoint: str, **params) -> str:
        """Build OpenWeatherMap API URL."""
        return f"{self.BASE_URL}/{endpoint}?{urlencode(params)}"
    
    def _parse_response(
        self,
        response_data: dict,
        coordinates: Coordinates,
        city: str,
        country: Optional[str] = None
    ) -> OpenWeatherMapResponse:
        """Parse OpenWeatherMap API response."""
        return OpenWeatherMapResponse(
            raw_data=response_data,
            provider=WeatherProvider.OPENWEATHER,
            request_timestamp=datetime.now(timezone.utc),
            response_timestamp=datetime.now(timezone.utc),
            **response_data
        )
