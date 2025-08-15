"""WeatherAPI client implementation."""

from __future__ import annotations

import aiohttp
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlencode

from .base import BaseWeatherClient
from ..models.api_responses import WeatherAPIResponse
from ..models.weather import Coordinates, WeatherDataPoint, WeatherProvider


class WeatherAPIClient(BaseWeatherClient):
    """Client for WeatherAPI service."""
    
    BASE_URL = "http://api.weatherapi.com/v1"
    
    def __init__(self, api_key: str, **kwargs):
        if not api_key:
            raise ValueError("WeatherAPI requires an API key")
        super().__init__(provider=WeatherProvider.WEATHERAPI, api_key=api_key, **kwargs)
    
    async def get_current_weather(
        self,
        coordinates: Coordinates,
        city: str,
        country: Optional[str] = None
    ) -> List[WeatherDataPoint]:
        """Get current weather from WeatherAPI."""
        response = await self.get_raw_response(coordinates, city, country, "current")
        return response.to_weather_points()
    
    async def get_forecast(
        self,
        coordinates: Coordinates,
        city: str,
        country: Optional[str] = None,
        days: int = 5
    ) -> List[WeatherDataPoint]:
        """Get weather forecast from WeatherAPI."""
        response = await self.get_raw_response(coordinates, city, country, "forecast", days=min(days, 10))
        return response.to_weather_points()
    
    async def get_raw_response(
        self,
        coordinates: Coordinates,
        city: str,
        country: Optional[str] = None,
        endpoint: str = "current",
        **params
    ) -> WeatherAPIResponse:
        """Get raw API response from WeatherAPI."""
        url = self._build_url(
            endpoint,
            key=self.api_key,
            q=f"{coordinates.latitude},{coordinates.longitude}",
            aqi="no",  # Don't include air quality data
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
        """Build WeatherAPI URL."""
        return f"{self.BASE_URL}/{endpoint}.json?{urlencode(params)}"
    
    def _parse_response(
        self,
        response_data: dict,
        coordinates: Coordinates,
        city: str,
        country: Optional[str] = None
    ) -> WeatherAPIResponse:
        """Parse WeatherAPI response."""
        return WeatherAPIResponse(
            raw_data=response_data,
            provider=WeatherProvider.WEATHERAPI,
            request_timestamp=datetime.now(timezone.utc),
            response_timestamp=datetime.now(timezone.utc),
            **response_data
        )
