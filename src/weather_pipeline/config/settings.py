"""Configuration management for the weather pipeline."""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    """Available log levels."""

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class Environment(str, Enum):
    """Available environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class APISettings(BaseSettings):
    """API-related settings."""

    model_config = SettingsConfigDict(
        env_prefix="WEATHER_API_",
        case_sensitive=False,
    )

    # API Keys
    openweather_api_key: Optional[str] = Field(
        default=None,
        description="OpenWeatherMap API key"
    )
    weatherapi_key: Optional[str] = Field(
        default=None,
        description="WeatherAPI key"
    )
    
    # Rate limiting
    rate_limit_requests: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Max concurrent requests"
    )
    request_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Request timeout in seconds"
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retry attempts"
    )
    retry_backoff_factor: float = Field(
        default=0.3,
        ge=0.0,
        le=5.0,
        description="Backoff factor for retries"
    )


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        case_sensitive=False,
    )

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(default="weather_data", description="Database name")
    user: str = Field(default="weather_user", description="Database user")
    password: str = Field(default="", description="Database password")
    
    @property
    def url(self) -> str:
        """Get database URL."""
        if self.password:
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        return f"postgresql://{self.user}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis configuration for caching."""

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        case_sensitive=False,
    )

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    ttl: int = Field(default=3600, ge=60, description="Default TTL in seconds")
    
    @property
    def url(self) -> str:
        """Get Redis URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        case_sensitive=False,
    )

    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    format: str = Field(
        default="json",
        description="Log format: 'json' or 'text'"
    )
    file_path: Optional[Path] = Field(
        default=None,
        description="Log file path (optional)"
    )
    max_file_size: str = Field(
        default="10MB",
        description="Maximum log file size"
    )
    backup_count: int = Field(
        default=5,
        ge=0,
        description="Number of backup log files to keep"
    )

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate log format."""
        if v not in ("json", "text"):
            raise ValueError("Log format must be 'json' or 'text'")
        return v


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # Application
    app_name: str = Field(
        default="Weather Pipeline",
        description="Application name"
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version"
    )
    
    # Paths
    data_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "data",
        description="Data directory path"
    )
    
    # Sub-configurations
    api: APISettings = Field(default_factory=APISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: Any) -> Environment:
        """Validate and convert environment."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION

    def is_testing(self) -> bool:
        """Check if running in testing."""
        return self.environment == Environment.TESTING


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function for accessing settings
settings = get_settings()
