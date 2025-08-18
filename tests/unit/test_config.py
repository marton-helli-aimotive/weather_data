"""Unit tests for configuration management."""

import pytest
import os
from unittest.mock import patch
from decimal import Decimal

from src.weather_pipeline.config.settings import (
    Settings, APISettings, DatabaseSettings, RedisSettings,
    LoggingSettings, Environment, LogLevel, get_settings
)


class TestSettings:
    """Test main Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            
            assert settings.app_name == "Weather Pipeline"
            assert settings.app_version == "0.1.0"
            assert settings.debug is False
            assert settings.environment == Environment.DEVELOPMENT

    def test_environment_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {
            'APP_NAME': 'Test Weather App',
            'DEBUG': 'true',
            'ENVIRONMENT': 'development'
        }):
            settings = Settings()
            
            assert settings.app_name == "Test Weather App"
            assert settings.debug is True
            assert settings.environment == "development"

    def test_nested_settings_initialization(self):
        """Test that nested settings are properly initialized."""
        settings = Settings()
        
        assert isinstance(settings.api, APISettings)
        assert isinstance(settings.database, DatabaseSettings)
        assert isinstance(settings.redis, RedisSettings)
        assert isinstance(settings.logging, LoggingSettings)

    def test_model_config(self):
        """Test model configuration."""
        settings = Settings()
        
        # Test that the model config is properly set
        assert 'env_file' in settings.model_config
        assert settings.model_config['env_file'] == '.env'


class TestAPISettings:
    """Test API-related settings."""

    def test_default_api_settings(self):
        """Test default API settings."""
        with patch.dict(os.environ, {}, clear=True):
            settings = APISettings()
            
            assert settings.openweather_api_key is None
            assert settings.weatherapi_key is None
            assert settings.rate_limit_requests == 10
            assert settings.request_timeout == 30
            assert settings.retry_attempts == 3
            assert settings.retry_backoff_factor == 0.3

    def test_api_settings_with_env_vars(self):
        """Test API settings with environment variables."""
        with patch.dict(os.environ, {
            'WEATHER_API_OPENWEATHER_API_KEY': 'openweather_test_key',
            'WEATHER_API_WEATHERAPI_KEY': 'weatherapi_test_key',
            'WEATHER_API_RATE_LIMIT_REQUESTS': '20',
            'WEATHER_API_REQUEST_TIMEOUT': '60',
            'WEATHER_API_RETRY_ATTEMPTS': '5'
        }):
            settings = APISettings()
            
            assert settings.openweather_api_key == "openweather_test_key"
            assert settings.weatherapi_key == "weatherapi_test_key"
            assert settings.rate_limit_requests == 20
            assert settings.request_timeout == 60
            assert settings.retry_attempts == 5

    def test_api_settings_validation(self):
        """Test API settings validation."""
        # Test rate limit validation
        with pytest.raises(ValueError):
            APISettings(rate_limit_requests=0)  # Below minimum
        
        with pytest.raises(ValueError):
            APISettings(rate_limit_requests=101)  # Above maximum
        
        # Test timeout validation
        with pytest.raises(ValueError):
            APISettings(request_timeout=0)  # Below minimum
        
        with pytest.raises(ValueError):
            APISettings(request_timeout=301)  # Above maximum

    def test_retry_config_validation(self):
        """Test retry configuration validation."""
        # Test retry attempts validation
        with pytest.raises(ValueError):
            APISettings(retry_attempts=-1)  # Below minimum
        
        with pytest.raises(ValueError):
            APISettings(retry_attempts=11)  # Above maximum
        
        # Test backoff factor validation
        with pytest.raises(ValueError):
            APISettings(retry_backoff_factor=-0.1)  # Below minimum
        
        with pytest.raises(ValueError):
            APISettings(retry_backoff_factor=5.1)  # Above maximum


class TestDatabaseSettings:
    """Test database-related settings."""

    def test_default_database_settings(self):
        """Test default database settings."""
        with patch.dict(os.environ, {}, clear=True):
            settings = DatabaseSettings()
            
            assert settings.host == "localhost"
            assert settings.port == 5432
            assert settings.name == "weather_data"
            assert settings.user == "weather_user"
            assert settings.password == ""

    def test_database_settings_with_env_vars(self):
        """Test database settings with environment variables."""
        with patch.dict(os.environ, {
            'DB_HOST': 'db.example.com',
            'DB_PORT': '5433',
            'DB_NAME': 'weather_test',
            'DB_USER': 'testuser',
            'DB_PASSWORD': 'testpass'
        }):
            settings = DatabaseSettings()
            
            assert settings.host == "db.example.com"
            assert settings.port == 5433
            assert settings.name == "weather_test"
            assert settings.user == "testuser"
            assert settings.password == "testpass"

    def test_database_url_construction(self):
        """Test database URL construction."""
        settings = DatabaseSettings(
            host="localhost",
            port=5432,
            name="weather_test",
            user="testuser",
            password="testpass"
        )
        
        expected_url = "postgresql://testuser:testpass@localhost:5432/weather_test"
        assert settings.url == expected_url

    def test_database_url_without_credentials(self):
        """Test database URL construction without credentials."""
        settings = DatabaseSettings(
            host="localhost",
            port=5432,
            name="weather_test"
        )
        
        expected_url = "postgresql://weather_user@localhost:5432/weather_test"
        assert settings.url == expected_url

    def test_port_validation(self):
        """Test port validation."""
        with pytest.raises(ValueError):
            DatabaseSettings(port=0)  # Below minimum
        
        with pytest.raises(ValueError):
            DatabaseSettings(port=65536)  # Above maximum


class TestRedisSettings:
    """Test cache-related settings."""

    def test_default_cache_settings(self):
        """Test default cache settings."""
        with patch.dict(os.environ, {}, clear=True):
            settings = RedisSettings()
            
            assert settings.host == "localhost"
            assert settings.port == 6379
            assert settings.password is None
            assert settings.db == 0
            assert settings.ttl == 3600

    def test_cache_settings_with_env_vars(self):
        """Test cache settings with environment variables."""
        with patch.dict(os.environ, {
            'REDIS_HOST': 'redis.example.com',
            'REDIS_PORT': '6380',
            'REDIS_PASSWORD': 'redis_pass',
            'REDIS_DB': '1',
            'REDIS_TTL': '7200'
        }):
            settings = RedisSettings()
            
            assert settings.host == "redis.example.com"
            assert settings.port == 6380
            assert settings.password == "redis_pass"
            assert settings.db == 1
            assert settings.ttl == 7200

    def test_redis_url_construction(self):
        """Test Redis URL construction."""
        settings = RedisSettings(
            host="localhost",
            port=6379,
            password="testpass",
            db=1
        )
        
        expected_url = "redis://:testpass@localhost:6379/1"
        assert settings.url == expected_url

    def test_redis_url_without_password(self):
        """Test Redis URL construction without password."""
        settings = RedisSettings(
            host="localhost",
            port=6379,
            password=None,
            db=0
        )
        
        expected_url = "redis://localhost:6379/0"
        assert settings.url == expected_url


class TestLoggingSettings:
    """Test logging-related settings."""

    def test_default_logging_settings(self):
        """Test default logging settings."""
        with patch.dict(os.environ, {}, clear=True):
            settings = LoggingSettings()
            
            assert settings.level == LogLevel.INFO
            assert settings.format == "json"
            assert settings.file_path is None
            assert settings.max_file_size == "10MB"
            assert settings.backup_count == 5

    def test_logging_settings_with_env_vars(self):
        """Test logging settings with environment variables."""
        with patch.dict(os.environ, {
            'LOG_LEVEL': 'DEBUG',
            'LOG_FORMAT': 'text',
            'LOG_FILE_PATH': '/var/log/weather.log',
            'LOG_MAX_FILE_SIZE': '50MB',
            'LOG_BACKUP_COUNT': '10'
        }):
            settings = LoggingSettings()
            
            assert settings.level == LogLevel.DEBUG
            assert settings.format == "text"
            assert str(settings.file_path) == "\\var\\log\\weather.log"
            assert settings.max_file_size == "50MB"
            assert settings.backup_count == 10

    def test_log_level_validation(self):
        """Test log level validation."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in valid_levels:
            settings = LoggingSettings(level=level)
            assert settings.level == level

        with pytest.raises(ValueError):
            LoggingSettings(level="INVALID")

    def test_log_format_validation(self):
        """Test log format validation."""
        valid_formats = ["json", "text"]
        
        for format_type in valid_formats:
            settings = LoggingSettings(format=format_type)
            assert settings.format == format_type

        with pytest.raises(ValueError):
            LoggingSettings(format="invalid")


class TestGetSettings:
    """Test the get_settings function."""

    def test_get_settings_singleton(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2

    def test_get_settings_with_custom_env_file(self, tmp_path):
        """Test get_settings works correctly."""
        # Clear the cache first
        get_settings.cache_clear()
        
        with patch.dict(os.environ, {
            'APP_NAME': 'Custom App',
            'DEBUG': 'true'
        }):
            settings = get_settings()
            
            assert settings.app_name == "Custom App"
            assert settings.debug is True

    def test_get_settings_env_override(self):
        """Test that environment variables override .env file."""
        # Clear the cache first
        get_settings.cache_clear()
        
        with patch.dict(os.environ, {
            'APP_NAME': 'Env Override App',
            'DEBUG': 'false'
        }):
            settings = get_settings()
            
            assert settings.app_name == "Env Override App"
            assert settings.debug is False


class TestSettingsIntegration:
    """Test settings integration scenarios."""

    def test_complete_settings_with_all_env_vars(self):
        """Test complete settings configuration with all environment variables."""
        env_vars = {
            # Main settings
            'APP_NAME': 'Test Weather Pipeline',
            'DEBUG': 'true',
            'ENVIRONMENT': 'testing',
            
            # API settings
            'WEATHER_API_OPENWEATHER_API_KEY': 'ow_test_key',
            'WEATHER_API_WEATHERAPI_KEY': 'wa_test_key',
            'WEATHER_API_RATE_LIMIT_REQUESTS': '15',
            
            # Database settings
            'DB_HOST': 'test-db.example.com',
            'DB_PORT': '5433',
            'DB_NAME': 'test_weather',
            'DB_USER': 'test_user',
            'DB_PASSWORD': 'test_pass',
            
            # Cache settings
            'REDIS_HOST': 'test-redis.example.com',
            'REDIS_PORT': '6380',
            'REDIS_TTL': '7200',
            
            # Logging settings
            'LOG_LEVEL': 'DEBUG',
            'LOG_FORMAT': 'text'
        }
        
        # Clear the cache first to ensure clean state
        get_settings.cache_clear()
        
        with patch.dict(os.environ, env_vars):
            settings = get_settings()
            
            # Verify main settings
            assert settings.app_name == "Test Weather Pipeline"
            assert settings.debug is True
            assert settings.environment == Environment.TESTING
            
            # Verify API settings
            assert settings.api.openweather_api_key == "ow_test_key"
            assert settings.api.weatherapi_key == "wa_test_key"
            assert settings.api.rate_limit_requests == 15
            
            # Verify database settings
            assert settings.database.host == "test-db.example.com"
            assert settings.database.port == 5433
            assert settings.database.name == "test_weather"
            
            # Verify cache settings
            assert settings.redis.host == "test-redis.example.com"
            assert settings.redis.port == 6380
            assert settings.redis.ttl == 7200
            
            # Verify logging settings
            assert settings.logging.level == LogLevel.DEBUG
            assert settings.logging.format == "text"

    def test_settings_validation_edge_cases(self):
        """Test settings validation with edge cases."""
        # Test minimum valid values
        api_settings = APISettings(
            rate_limit_requests=1,
            request_timeout=1,
            retry_attempts=0,
            retry_backoff_factor=0.0
        )
        assert api_settings.rate_limit_requests == 1
        
        # Test maximum valid values
        api_settings = APISettings(
            rate_limit_requests=100,
            request_timeout=300,
            retry_attempts=10,
            retry_backoff_factor=5.0
        )
        assert api_settings.rate_limit_requests == 100
