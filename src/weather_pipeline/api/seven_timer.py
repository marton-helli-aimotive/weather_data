"""7Timer API client implementation."""

from __future__ import annotations

import aiohttp
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlencode

from .base import BaseWeatherClient
from ..models.api_responses import SevenTimerResponse
from ..models.weather import Coordinates, WeatherDataPoint, WeatherProvider


class SevenTimerClient(BaseWeatherClient):
    """Client for 7Timer weather API."""
    
    BASE_URL = "http://www.7timer.info/bin/api.pl"
    
    def __init__(self, **kwargs):
        # 7Timer doesn't require API key
        super().__init__(provider=WeatherProvider.SEVEN_TIMER, **kwargs)
    
    async def get_current_weather(
        self,
        coordinates: Coordinates,
        city: str,
        country: Optional[str] = None
    ) -> List[WeatherDataPoint]:
        """Get current weather from 7Timer (actually returns forecast)."""
        # 7Timer only provides forecast data, not current weather
        forecast = await self.get_forecast(coordinates, city, country, days=1)
        return forecast[:1] if forecast else []  # Return first forecast point as "current"
    
    async def get_forecast(
        self,
        coordinates: Coordinates,
        city: str,
        country: Optional[str] = None,
        days: int = 5
    ) -> List[WeatherDataPoint]:
        """Get weather forecast from 7Timer."""
        response = await self.get_raw_response(coordinates, city, country)
        return response.to_weather_points(coordinates, city, country)
    
    async def get_raw_response(
        self,
        coordinates: Coordinates,
        city: str,
        country: Optional[str] = None,
        endpoint: str = "current"
    ) -> SevenTimerResponse:
        """Get raw API response from 7Timer."""
        url = self._build_url(
            "astro",  # 7Timer endpoint type
            lon=coordinates.longitude,
            lat=coordinates.latitude,
            ac=0,  # Altitude correction (0 for automatic)
            unit="metric",
            output="json",
            tzshift=0  # UTC timezone
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
        """Build 7Timer API URL."""
        query_params = {"product": endpoint, **params}
        return f"{self.BASE_URL}?{urlencode(query_params)}"
    
    def _parse_response(
        self,
        response_data: dict,
        coordinates: Coordinates,
        city: str,
        country: Optional[str] = None
    ) -> SevenTimerResponse:
        """Parse 7Timer API response."""
        return SevenTimerResponse(
            raw_data=response_data,
            provider=WeatherProvider.SEVEN_TIMER,
            request_timestamp=datetime.now(timezone.utc),
            response_timestamp=datetime.now(timezone.utc),
            **response_data
        )
