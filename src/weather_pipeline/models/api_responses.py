"""Provider-specific API response models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .weather import Coordinates, WeatherDataPoint, WeatherProvider


class BaseAPIResponse(BaseModel):
    """Base class for all API responses."""
    
    raw_data: Dict[str, Any] = Field(..., description="Raw API response data")
    provider: WeatherProvider = Field(..., description="API provider")
    request_timestamp: datetime = Field(..., description="When the request was made")
    response_timestamp: datetime = Field(..., description="When the response was received")
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }


# 7Timer API Response Models
class SevenTimerDataPoint(BaseModel):
    """Single data point from 7Timer API."""
    
    timepoint: int = Field(..., description="Hours from init time")
    cloudcover: int = Field(..., ge=1, le=9, description="Cloud cover (1-9 scale)")
    seeing: int = Field(..., ge=1, le=8, description="Atmospheric seeing (1-8 scale)")
    transparency: int = Field(..., ge=1, le=8, description="Atmospheric transparency (1-8 scale)")
    lifted_index: int = Field(..., description="Lifted index for instability")
    rh2m: int = Field(..., ge=0, le=100, description="Relative humidity at 2m (%)")
    wind10m_direction: str = Field(..., description="Wind direction at 10m")
    wind10m_speed: int = Field(..., ge=0, description="Wind speed at 10m")
    temp2m: int = Field(..., description="Temperature at 2m (Celsius)")
    prec_type: str = Field(..., description="Precipitation type")


class SevenTimerResponse(BaseAPIResponse):
    """7Timer API response structure."""
    
    product: str = Field(..., description="Product type (e.g., 'astro')")
    init: str = Field(..., description="Initialization time")
    dataseries: List[SevenTimerDataPoint] = Field(..., description="Weather data series")
    
    def to_weather_points(self, coordinates: Coordinates, city: str, country: Optional[str] = None) -> List[WeatherDataPoint]:
        """Convert 7Timer data to standard WeatherDataPoint format."""
        init_time = datetime.strptime(self.init, "%Y%m%d%H")
        points = []
        
        for data in self.dataseries:
            timestamp = init_time.replace(hour=init_time.hour + data.timepoint, tzinfo=timezone.utc)
            
            # Convert 7Timer specific values to standard format
            humidity = data.rh2m
            temperature = float(data.temp2m)
            cloud_cover = int((data.cloudcover - 1) * 12.5)  # Convert 1-9 scale to 0-100%
            
            # Convert wind direction from string to degrees
            wind_directions = {
                'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
                'S': 180, 'SW': 225, 'W': 270, 'NW': 315
            }
            wind_direction = wind_directions.get(data.wind10m_direction, None)
            wind_speed = float(data.wind10m_speed)
            
            point = WeatherDataPoint(
                timestamp=timestamp,
                temperature=temperature,
                humidity=humidity,
                wind_speed=wind_speed,
                wind_direction=wind_direction,
                cloud_cover=cloud_cover,
                city=city,
                country=country,
                coordinates=coordinates,
                provider=WeatherProvider.SEVEN_TIMER,
                confidence_score=0.8  # 7Timer has decent accuracy
            )
            points.append(point)
        
        return points


# WeatherAPI Response Models
class WeatherAPILocation(BaseModel):
    """Location data from WeatherAPI."""
    
    name: str = Field(..., description="Location name")
    region: Optional[str] = Field(None, description="Region/state")
    country: str = Field(..., description="Country")
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    tz_id: str = Field(..., description="Timezone ID")
    localtime: str = Field(..., description="Local time")


class WeatherAPICondition(BaseModel):
    """Weather condition from WeatherAPI."""
    
    text: str = Field(..., description="Weather condition text")
    icon: str = Field(..., description="Weather icon URL")
    code: int = Field(..., description="Weather condition code")


class WeatherAPICurrent(BaseModel):
    """Current weather from WeatherAPI."""
    
    temp_c: float = Field(..., description="Temperature in Celsius")
    temp_f: float = Field(..., description="Temperature in Fahrenheit")
    condition: WeatherAPICondition = Field(..., description="Weather condition")
    wind_mph: float = Field(..., description="Wind speed in mph")
    wind_kph: float = Field(..., description="Wind speed in kph")
    wind_degree: int = Field(..., description="Wind direction in degrees")
    wind_dir: str = Field(..., description="Wind direction")
    pressure_mb: float = Field(..., description="Pressure in millibars")
    pressure_in: float = Field(..., description="Pressure in inches")
    precip_mm: float = Field(..., description="Precipitation in mm")
    precip_in: float = Field(..., description="Precipitation in inches")
    humidity: int = Field(..., description="Humidity percentage")
    cloud: int = Field(..., description="Cloud cover percentage")
    feelslike_c: float = Field(..., description="Feels like temperature in Celsius")
    vis_km: float = Field(..., description="Visibility in km")
    uv: float = Field(..., description="UV index")


class WeatherAPIForecastDay(BaseModel):
    """Forecast day data from WeatherAPI."""
    
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    day: Dict[str, Any] = Field(..., description="Day forecast data")
    astro: Dict[str, Any] = Field(..., description="Astronomical data")
    hour: List[Dict[str, Any]] = Field(..., description="Hourly forecast data")


class WeatherAPIResponse(BaseAPIResponse):
    """WeatherAPI response structure."""
    
    location: WeatherAPILocation = Field(..., description="Location information")
    current: Optional[WeatherAPICurrent] = Field(None, description="Current weather")
    forecast: Optional[Dict[str, List[WeatherAPIForecastDay]]] = Field(None, description="Forecast data")
    
    def to_weather_points(self) -> List[WeatherDataPoint]:
        """Convert WeatherAPI data to standard WeatherDataPoint format."""
        points = []
        coordinates = Coordinates(latitude=self.location.lat, longitude=self.location.lon)
        
        # Add current weather if available
        if self.current:
            point = WeatherDataPoint(
                timestamp=datetime.now(timezone.utc),
                temperature=self.current.temp_c,
                humidity=self.current.humidity,
                pressure=self.current.pressure_mb,
                wind_speed=self.current.wind_kph / 3.6,  # Convert kph to m/s
                wind_direction=self.current.wind_degree,
                precipitation=self.current.precip_mm,
                visibility=self.current.vis_km,
                cloud_cover=self.current.cloud,
                uv_index=self.current.uv,
                city=self.location.name,
                country=self.location.country,
                coordinates=coordinates,
                provider=WeatherProvider.WEATHERAPI,
                is_forecast=False,
                confidence_score=0.95  # WeatherAPI has high accuracy
            )
            points.append(point)
        
        # Add forecast data if available
        if self.forecast and "forecastday" in self.forecast:
            for forecast_day in self.forecast["forecastday"]:
                for hour_data in forecast_day["hour"]:
                    timestamp = datetime.fromisoformat(hour_data["time"].replace(" ", "T")).replace(tzinfo=timezone.utc)
                    
                    point = WeatherDataPoint(
                        timestamp=timestamp,
                        temperature=hour_data["temp_c"],
                        humidity=hour_data["humidity"],
                        pressure=hour_data["pressure_mb"],
                        wind_speed=hour_data["wind_kph"] / 3.6,  # Convert kph to m/s
                        wind_direction=hour_data["wind_degree"],
                        precipitation=hour_data["precip_mm"],
                        visibility=hour_data["vis_km"],
                        cloud_cover=hour_data["cloud"],
                        uv_index=hour_data["uv"],
                        city=self.location.name,
                        country=self.location.country,
                        coordinates=coordinates,
                        provider=WeatherProvider.WEATHERAPI,
                        confidence_score=0.9
                    )
                    points.append(point)
        
        return points


# OpenWeatherMap Response Models
class OpenWeatherMapCoord(BaseModel):
    """Coordinates from OpenWeatherMap."""
    
    lon: float = Field(..., description="Longitude")
    lat: float = Field(..., description="Latitude")


class OpenWeatherMapWeather(BaseModel):
    """Weather condition from OpenWeatherMap."""
    
    id: int = Field(..., description="Weather condition id")
    main: str = Field(..., description="Group of weather parameters")
    description: str = Field(..., description="Weather condition description")
    icon: str = Field(..., description="Weather icon id")


class OpenWeatherMapMain(BaseModel):
    """Main weather data from OpenWeatherMap."""
    
    temp: float = Field(..., description="Temperature in Kelvin")
    feels_like: float = Field(..., description="Feels like temperature in Kelvin")
    temp_min: float = Field(..., description="Minimum temperature in Kelvin")
    temp_max: float = Field(..., description="Maximum temperature in Kelvin")
    pressure: int = Field(..., description="Atmospheric pressure in hPa")
    humidity: int = Field(..., description="Humidity percentage")
    sea_level: Optional[int] = Field(None, description="Atmospheric pressure on sea level")
    grnd_level: Optional[int] = Field(None, description="Atmospheric pressure on ground level")


class OpenWeatherMapWind(BaseModel):
    """Wind data from OpenWeatherMap."""
    
    speed: float = Field(..., description="Wind speed in m/s")
    deg: Optional[int] = Field(None, description="Wind direction in degrees")
    gust: Optional[float] = Field(None, description="Wind gust in m/s")


class OpenWeatherMapClouds(BaseModel):
    """Cloud data from OpenWeatherMap."""
    
    all: int = Field(..., description="Cloudiness percentage")


class OpenWeatherMapRain(BaseModel):
    """Rain data from OpenWeatherMap."""
    
    one_h: Optional[float] = Field(None, alias="1h", description="Rain volume for last hour")
    three_h: Optional[float] = Field(None, alias="3h", description="Rain volume for last 3 hours")


class OpenWeatherMapSys(BaseModel):
    """System data from OpenWeatherMap."""
    
    type: Optional[int] = Field(None, description="Internal parameter")
    id: Optional[int] = Field(None, description="Internal parameter")
    country: Optional[str] = Field(None, description="Country code")
    sunrise: Optional[int] = Field(None, description="Sunrise time in Unix UTC")
    sunset: Optional[int] = Field(None, description="Sunset time in Unix UTC")


class OpenWeatherMapResponse(BaseAPIResponse):
    """OpenWeatherMap API response structure."""
    
    coord: Optional[OpenWeatherMapCoord] = Field(None, description="Coordinates")
    weather: List[OpenWeatherMapWeather] = Field(..., description="Weather conditions")
    base: Optional[str] = Field(None, description="Internal parameter")
    main: OpenWeatherMapMain = Field(..., description="Main weather data")
    visibility: Optional[int] = Field(None, description="Visibility in meters")
    wind: Optional[OpenWeatherMapWind] = Field(None, description="Wind data")
    clouds: Optional[OpenWeatherMapClouds] = Field(None, description="Clouds data")
    rain: Optional[OpenWeatherMapRain] = Field(None, description="Rain data")
    dt: int = Field(..., description="Time of data calculation, unix UTC")
    sys: Optional[OpenWeatherMapSys] = Field(None, description="System data")
    timezone: Optional[int] = Field(None, description="Shift in seconds from UTC")
    id: Optional[int] = Field(None, description="City ID")
    name: Optional[str] = Field(None, description="City name")
    cod: Optional[int] = Field(None, description="Internal parameter")
    
    def to_weather_points(self, city: str, country: Optional[str] = None) -> List[WeatherDataPoint]:
        """Convert OpenWeatherMap data to standard WeatherDataPoint format."""
        if not self.coord:
            raise ValueError("Coordinate data is required")
        
        coordinates = Coordinates(latitude=self.coord.lat, longitude=self.coord.lon)
        timestamp = datetime.fromtimestamp(self.dt, tz=timezone.utc)
        
        # Get precipitation data
        precipitation = 0.0
        if self.rain:
            precipitation = self.rain.one_h or self.rain.three_h or 0.0
        
        # Get wind data
        wind_speed = self.wind.speed if self.wind else None
        wind_direction = self.wind.deg if self.wind else None
        
        # Get cloud cover
        cloud_cover = self.clouds.all if self.clouds else None
        
        # Get visibility (convert from meters to km)
        visibility_km = self.visibility / 1000 if self.visibility else None
        
        point = WeatherDataPoint(
            timestamp=timestamp,
            temperature=self.main.temp - 273.15,  # Convert Kelvin to Celsius
            humidity=self.main.humidity,
            pressure=float(self.main.pressure),
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            precipitation=precipitation,
            visibility=visibility_km,
            cloud_cover=cloud_cover,
            city=city,
            country=country or (self.sys.country if self.sys else None),
            coordinates=coordinates,
            provider=WeatherProvider.OPENWEATHER,
            is_forecast=False,
            confidence_score=0.92  # OpenWeatherMap has high accuracy
        )
        
        return [point]


# Union type for all API responses
APIResponse = Union[SevenTimerResponse, WeatherAPIResponse, OpenWeatherMapResponse]
