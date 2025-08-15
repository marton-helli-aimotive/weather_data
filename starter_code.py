"""
Modern Weather Data Pipeline Starter Code
Demonstrates async programming, type hints, and modern Python patterns
"""
import asyncio
import aiohttp
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
import polars as pl
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherProvider(Enum):
    SEVEN_TIMER = "7timer"
    OPENWEATHER = "openweather"

@dataclass(frozen=True)
class Coordinates:
    """Immutable coordinate representation with validation"""
    latitude: float
    longitude: float
    
    def __post_init__(self):
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Invalid longitude: {self.longitude}")

class WeatherDataPoint(BaseModel):
    """Pydantic model for weather data validation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    timestamp: datetime = Field(..., description="Forecast timestamp")
    temperature: float = Field(..., ge=-100, le=60, description="Temperature in Celsius")
    humidity: Optional[int] = Field(None, ge=0, le=100, description="Humidity percentage")
    wind_speed: Optional[float] = Field(None, ge=0, description="Wind speed")
    city: str = Field(..., min_length=1, description="City name")
    provider: WeatherProvider = Field(..., description="Data source")

class WeatherAPIClient:
    """Async weather API client with rate limiting and error handling"""
    
    def __init__(self, session: aiohttp.ClientSession, rate_limit: int = 10):
        self.session = session
        self.rate_limit = asyncio.Semaphore(rate_limit)
    
    async def fetch_7timer_data(self, city: str, coords: Coordinates) -> List[WeatherDataPoint]:
        """Fetch data from 7timer API with error handling"""
        async with self.rate_limit:
            try:
                params = {
                    'lat': coords.latitude,
                    'lon': coords.longitude,
                    'unit': 'metric',
                    'output': 'json'
                }
                
                async with self.session.get(
                    "https://www.7timer.info/bin/astro.php", 
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    return self._parse_7timer_response(data, city)
                    
            except Exception as e:
                logger.error(f"Failed to fetch data for {city}: {e}")
                return []
    
    def _parse_7timer_response(self, data: Dict[str, Any], city: str) -> List[WeatherDataPoint]:
        """Parse 7timer API response into structured data"""
        weather_points = []
        
        for item in data.get('dataseries', []):
            try:
                weather_point = WeatherDataPoint(
                    timestamp=datetime.now(),  # Simplified - should parse actual timestamp
                    temperature=item.get('temp2m', 0),
                    humidity=item.get('rh2m'),
                    wind_speed=item.get('wind10m_speed'),
                    city=city,
                    provider=WeatherProvider.SEVEN_TIMER
                )
                weather_points.append(weather_point)
            except Exception as e:
                logger.warning(f"Failed to parse weather data point: {e}")
                continue
        
        return weather_points

class WeatherDataProcessor:
    """Process and analyze weather data using modern pandas/polars"""
    
    def __init__(self, data: List[WeatherDataPoint]):
        self.raw_data = data
        self._df_pandas: Optional[pd.DataFrame] = None
        self._df_polars: Optional[pl.DataFrame] = None
    
    @property
    def pandas_df(self) -> pd.DataFrame:
        """Lazy-loaded pandas DataFrame"""
        if self._df_pandas is None:
            records = [point.model_dump() for point in self.raw_data]
            self._df_pandas = pd.DataFrame(records)
        return self._df_pandas
    
    @property
    def polars_df(self) -> pl.DataFrame:
        """Lazy-loaded Polars DataFrame for better performance"""
        if self._df_polars is None:
            records = [point.model_dump() for point in self.raw_data]
            self._df_polars = pl.DataFrame(records)
        return self._df_polars
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics using Polars for performance"""
        if not self.raw_data:
            return {}
        
        df = self.polars_df
        
        numerical_stats = (
            df.select([
                pl.col("temperature").min().alias("temp_min"),
                pl.col("temperature").max().alias("temp_max"),
                pl.col("temperature").mean().alias("temp_mean"),
                pl.col("temperature").std().alias("temp_std"),
            ])
        ).to_dict(as_series=False)
        
        categorical_stats = (
            df.group_by("city")
            .agg([
                pl.count().alias("count"),
                pl.col("temperature").mean().alias("avg_temp")
            ])
        ).to_dict(as_series=False)
        
        return {
            "numerical": numerical_stats,
            "by_city": categorical_stats
        }

async def main():
    """Main async function demonstrating the weather data pipeline"""
    
    # City coordinates
    cities = {
        'Budapest': Coordinates(47.4979, 19.0402),
        'London': Coordinates(51.5074, -0.1278),
        'New York': Coordinates(40.7128, -74.0060),
        'Tokyo': Coordinates(35.6762, 139.6503),
        'Cape Town': Coordinates(-33.9249, 18.4241)
    }
    
    # Create async session with proper resource management
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        client = WeatherAPIClient(session)
        
        # Fetch data concurrently for all cities
        tasks = [
            client.fetch_7timer_data(city, coords)
            for city, coords in cities.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and filter out exceptions
        all_weather_data = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
            else:
                all_weather_data.extend(result)
        
        if not all_weather_data:
            logger.error("No weather data collected")
            return
        
        # Process the collected data
        processor = WeatherDataProcessor(all_weather_data)
        stats = processor.calculate_statistics()
        
        print("Weather Data Statistics:")
        print(f"Total data points collected: {len(all_weather_data)}")
        print(f"Numerical statistics: {stats.get('numerical', {})}")
        print(f"By city statistics: {stats.get('by_city', {})}")
        
        # Display DataFrames
        print("\nPandas DataFrame sample:")
        print(processor.pandas_df.head())
        
        print("\nPolars DataFrame info:")
        print(processor.polars_df.head())

if __name__ == "__main__":
    asyncio.run(main())