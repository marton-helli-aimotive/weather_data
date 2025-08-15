"""Data management for the dashboard - fetching and preparing data for visualization."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.container import DIContainer
from ..models.weather import WeatherDataPoint, Coordinates, WeatherProvider
from ..processing.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class DashboardDataManager:
    """Manages data fetching and preparation for dashboard visualizations."""
    
    def __init__(self, container: DIContainer):
        """Initialize the data manager."""
        self.container = container
        # Try to get cache manager from container, otherwise create a simple in-memory cache
        try:
            self.cache_manager = container.get_optional(CacheManager)
        except:
            self.cache_manager = None
        
        # City coordinates mapping
        self.city_coordinates = {
            'new_york': Coordinates(latitude=40.7128, longitude=-74.0060),
            'london': Coordinates(latitude=51.5074, longitude=-0.1278),
            'tokyo': Coordinates(latitude=35.6762, longitude=139.6503),
            'sydney': Coordinates(latitude=-33.8688, longitude=151.2093),
            'paris': Coordinates(latitude=48.8566, longitude=2.3522),
            'berlin': Coordinates(latitude=52.5200, longitude=13.4050),
            'moscow': Coordinates(latitude=55.7558, longitude=37.6176),
            'beijing': Coordinates(latitude=39.9042, longitude=116.4074),
            'mumbai': Coordinates(latitude=19.0760, longitude=72.8777),
            'rio_de_janeiro': Coordinates(latitude=-22.9068, longitude=-43.1729),
        }
        
        # City display names
        self.city_names = {
            'new_york': 'New York, NY',
            'london': 'London, UK',
            'tokyo': 'Tokyo, JP',
            'sydney': 'Sydney, AU',
            'paris': 'Paris, FR',
            'berlin': 'Berlin, DE',
            'moscow': 'Moscow, RU',
            'beijing': 'Beijing, CN',
            'mumbai': 'Mumbai, IN',
            'rio_de_janeiro': 'Rio de Janeiro, BR',
        }
    
    async def get_weather_data(
        self,
        cities: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch weather data for specified cities and date range.
        
        Args:
            cities: List of city identifiers
            start_date: Start date for data (ISO format)
            end_date: End date for data (ISO format)
            
        Returns:
            List of weather data dictionaries
        """
        logger.info(f"Fetching weather data for cities: {cities}")
        
        # Parse dates
        if start_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        else:
            start_dt = datetime.utcnow() - timedelta(days=7)
            
        if end_date:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        else:
            end_dt = datetime.utcnow()
        
        all_data = []
        
        # Fetch data for each city
        for city in cities:
            try:
                city_data = await self._fetch_city_data(city, start_dt, end_dt)
                all_data.extend(city_data)
            except Exception as e:
                logger.error(f"Failed to fetch data for {city}: {e}")
                # Add mock data to ensure dashboard doesn't break
                mock_data = self._generate_mock_data(city, start_dt, end_dt)
                all_data.extend(mock_data)
        
        logger.info(f"Retrieved {len(all_data)} weather data points")
        return all_data
    
    async def _fetch_city_data(
        self,
        city: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch data for a specific city."""
        cache_key = f"dashboard_data_{city}_{start_date.date()}_{end_date.date()}"
        
        # Try cache first if available
        cached_data = None
        if self.cache_manager:
            try:
                cached_data = await self.cache_manager.get(cache_key)
            except:
                cached_data = None
        
        if cached_data:
            logger.debug(f"Using cached data for {city}")
            return cached_data
        
        # Generate data (in real implementation, this would call the API clients)
        data = self._generate_mock_data(city, start_date, end_date)
        
        # Cache the data for 30 minutes if cache manager is available
        if self.cache_manager:
            try:
                await self.cache_manager.set(cache_key, data, ttl=1800)
            except:
                pass  # Ignore cache errors
        
        return data
    
    def _generate_mock_data(
        self,
        city: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Generate mock weather data for demonstration purposes.
        
        In a real implementation, this would be replaced with actual API calls.
        """
        import random
        import math
        
        data = []
        coordinates = self.city_coordinates.get(city, Coordinates(latitude=0.0, longitude=0.0))
        city_name = self.city_names.get(city, city.replace('_', ' ').title())
        
        # Generate hourly data points
        current = start_date
        hour_count = 0
        
        # Base climate parameters per city
        city_climate = {
            'new_york': {'temp_base': 15, 'temp_range': 25, 'humidity_base': 60},
            'london': {'temp_base': 12, 'temp_range': 15, 'humidity_base': 70},
            'tokyo': {'temp_base': 16, 'temp_range': 20, 'humidity_base': 65},
            'sydney': {'temp_base': 20, 'temp_range': 15, 'humidity_base': 55},
            'paris': {'temp_base': 14, 'temp_range': 18, 'humidity_base': 65},
            'berlin': {'temp_base': 10, 'temp_range': 20, 'humidity_base': 65},
            'moscow': {'temp_base': 5, 'temp_range': 30, 'humidity_base': 70},
            'beijing': {'temp_base': 12, 'temp_range': 35, 'humidity_base': 55},
            'mumbai': {'temp_base': 28, 'temp_range': 8, 'humidity_base': 80},
            'rio_de_janeiro': {'temp_base': 25, 'temp_range': 10, 'humidity_base': 75},
        }
        
        climate = city_climate.get(city, {'temp_base': 15, 'temp_range': 20, 'humidity_base': 60})
        
        while current <= end_date:
            # Add some seasonality and daily patterns
            day_of_year = current.timetuple().tm_yday
            hour_of_day = current.hour
            
            # Seasonal temperature variation
            seasonal_temp = climate['temp_base'] + 10 * math.sin(2 * math.pi * day_of_year / 365)
            
            # Daily temperature variation
            daily_temp_variation = 5 * math.sin(2 * math.pi * hour_of_day / 24)
            
            # Add random noise
            temp_noise = random.uniform(-3, 3)
            temperature = seasonal_temp + daily_temp_variation + temp_noise
            
            # Humidity (inversely related to temperature somewhat)
            humidity = max(20, min(95, climate['humidity_base'] + random.uniform(-15, 15) - (temperature - climate['temp_base']) * 0.5))
            
            # Pressure (with realistic variation)
            pressure = 1013.25 + random.uniform(-20, 20) + 5 * math.sin(2 * math.pi * hour_count / 168)  # Weekly cycle
            
            # Wind speed
            wind_speed = max(0, random.uniform(0, 15) + 2 * math.sin(2 * math.pi * hour_of_day / 24))
            
            # Wind direction
            wind_direction = random.uniform(0, 360)
            
            # Precipitation (occasional)
            precipitation = random.uniform(0, 5) if random.random() < 0.1 else 0
            
            # Visibility
            visibility = max(1, random.uniform(5, 20) - precipitation * 2)
            
            # Cloud cover
            cloud_cover = min(100, max(0, random.uniform(0, 100) + precipitation * 10))
            
            # UV index (based on time of day and season)
            if 6 <= hour_of_day <= 18:  # Daytime
                uv_index = max(0, (temperature - 10) / 5 + random.uniform(-1, 1))
            else:
                uv_index = 0
            
            data_point = {
                'timestamp': current.isoformat(),
                'temperature': round(temperature, 1),
                'humidity': round(humidity),
                'pressure': round(pressure, 1),
                'wind_speed': round(wind_speed, 1),
                'wind_direction': round(wind_direction),
                'precipitation': round(precipitation, 1),
                'visibility': round(visibility, 1),
                'cloud_cover': round(cloud_cover),
                'uv_index': round(uv_index, 1),
                'city': city_name,
                'country': self._get_country_for_city(city),
                'coordinates': {
                    'latitude': coordinates.latitude,
                    'longitude': coordinates.longitude
                },
                'provider': 'mock',
                'is_forecast': current > datetime.utcnow(),
                'confidence_score': random.uniform(0.8, 1.0)
            }
            
            data.append(data_point)
            current += timedelta(hours=1)
            hour_count += 1
        
        return data
    
    def _get_country_for_city(self, city: str) -> str:
        """Get country code for a city."""
        country_mapping = {
            'new_york': 'US',
            'london': 'GB',
            'tokyo': 'JP',
            'sydney': 'AU',
            'paris': 'FR',
            'berlin': 'DE',
            'moscow': 'RU',
            'beijing': 'CN',
            'mumbai': 'IN',
            'rio_de_janeiro': 'BR',
        }
        return country_mapping.get(city, 'UN')
    
    async def get_real_time_data(self, cities: List[str]) -> List[Dict[str, Any]]:
        """Get real-time (current) weather data for cities.
        
        Args:
            cities: List of city identifiers
            
        Returns:
            List of current weather data
        """
        current_time = datetime.utcnow()
        return await self.get_weather_data(
            cities=cities,
            start_date=current_time.isoformat(),
            end_date=current_time.isoformat()
        )
    
    async def get_forecast_data(self, cities: List[str], days: int = 7) -> List[Dict[str, Any]]:
        """Get forecast data for cities.
        
        Args:
            cities: List of city identifiers
            days: Number of days to forecast
            
        Returns:
            List of forecast weather data
        """
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(days=days)
        
        return await self.get_weather_data(
            cities=cities,
            start_date=start_time.isoformat(),
            end_date=end_time.isoformat()
        )
    
    def get_available_cities(self) -> List[Dict[str, str]]:
        """Get list of available cities for selection.
        
        Returns:
            List of city options with labels and values
        """
        return [
            {'label': name, 'value': city_id}
            for city_id, name in self.city_names.items()
        ]
