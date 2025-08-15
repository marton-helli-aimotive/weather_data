"""Weather Data Pipeline - A comprehensive weather data engineering solution."""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Weather Pipeline Team"
__email__ = "team@example.com"

# Core imports
from .config import Settings, get_settings, settings
from .core import configure_logging, get_logger
from .models import (
    Coordinates,
    DataQualityMetrics,
    LocationInfo,
    WeatherAlert,
    WeatherDataPoint,
    WeatherProvider,
)

__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    
    # Configuration
    "Settings",
    "get_settings",
    "settings",
    
    # Logging
    "configure_logging",
    "get_logger",
    
    # Models
    "Coordinates",
    "DataQualityMetrics",
    "LocationInfo", 
    "WeatherAlert",
    "WeatherDataPoint",
    "WeatherProvider",
]
