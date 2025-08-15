"""Core data models for the weather pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class WeatherProvider(str, Enum):
    """Supported weather data providers."""
    
    SEVEN_TIMER = "7timer"
    OPENWEATHER = "openweather"  
    WEATHERAPI = "weatherapi"


class Coordinates(BaseModel):
    """Geographic coordinates with validation."""
    
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude in decimal degrees"
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude in decimal degrees"
    )
    
    model_config = {"frozen": True}  # Make immutable


class WeatherDataPoint(BaseModel):
    """Individual weather data point with validation."""
    
    timestamp: datetime = Field(..., description="Forecast timestamp")
    temperature: Optional[float] = Field(
        None,
        ge=-100.0,
        le=60.0,
        description="Temperature in Celsius"
    )
    humidity: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Humidity percentage"
    )
    pressure: Optional[float] = Field(
        None,
        ge=0.0,
        description="Atmospheric pressure in hPa"
    )
    wind_speed: Optional[float] = Field(
        None,
        ge=0.0,
        description="Wind speed in m/s"
    )
    wind_direction: Optional[int] = Field(
        None,
        ge=0,
        le=360,
        description="Wind direction in degrees"
    )
    precipitation: Optional[float] = Field(
        None,
        ge=0.0,
        description="Precipitation in mm"
    )
    visibility: Optional[float] = Field(
        None,
        ge=0.0,
        description="Visibility in km"
    )
    cloud_cover: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Cloud cover percentage"
    )
    uv_index: Optional[float] = Field(
        None,
        ge=0.0,
        description="UV index"
    )
    
    # Metadata
    city: str = Field(..., min_length=1, description="City name")
    country: Optional[str] = Field(None, description="Country name")
    coordinates: Coordinates = Field(..., description="Geographic coordinates")
    provider: WeatherProvider = Field(..., description="Data source provider")
    
    # Data quality indicators
    is_forecast: bool = Field(default=True, description="Is this forecast or historical data")
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Data quality confidence score"
    )
    
    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        return v
    
    model_config = {
        # Allow validation of datetime objects
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }


class WeatherAlert(BaseModel):
    """Weather alert or warning."""
    
    alert_id: str = Field(..., description="Unique alert identifier")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    severity: str = Field(..., description="Alert severity level")
    start_time: datetime = Field(..., description="Alert start time")
    end_time: Optional[datetime] = Field(None, description="Alert end time")
    areas: list[str] = Field(default_factory=list, description="Affected areas")
    tags: list[str] = Field(default_factory=list, description="Alert tags")
    
    provider: WeatherProvider = Field(..., description="Alert source provider")
    created_at: datetime = Field(default_factory=lambda: datetime.now(datetime.timezone.utc))


class LocationInfo(BaseModel):
    """Extended location information."""
    
    name: str = Field(..., description="Location name")
    country: str = Field(..., description="Country name")
    region: Optional[str] = Field(None, description="State/region name")
    coordinates: Coordinates = Field(..., description="Geographic coordinates")
    timezone: str = Field(..., description="Timezone identifier")
    population: Optional[int] = Field(None, ge=0, description="Population count")
    elevation: Optional[float] = Field(None, description="Elevation in meters")
    
    # Additional metadata
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    tags: list[str] = Field(default_factory=list, description="Location tags")


class DataQualityMetrics(BaseModel):
    """Data quality metrics for weather data."""
    
    total_records: int = Field(ge=0, description="Total number of records")
    valid_records: int = Field(ge=0, description="Number of valid records")
    missing_temperature: int = Field(ge=0, description="Records with missing temperature")
    missing_humidity: int = Field(ge=0, description="Records with missing humidity")
    outliers_detected: int = Field(ge=0, description="Number of outliers detected")
    
    # Calculated metrics
    completeness_score: float = Field(ge=0.0, le=1.0, description="Data completeness score")
    quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality score")
    
    # Timestamps
    from datetime import timezone
    assessment_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data_time_range_start: Optional[datetime] = Field(None)
    data_time_range_end: Optional[datetime] = Field(None)
    
    @model_validator(mode='after')
    def valid_records_not_exceed_total(self) -> 'DataQualityMetrics':
        """Ensure valid records don't exceed total records."""
        if self.valid_records > self.total_records:
            raise ValueError("Valid records cannot exceed total records")
        return self
