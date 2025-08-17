-- Weather Data Pipeline Database Initialization
-- This script sets up the initial database schema for the weather pipeline

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;

-- Create weather data table with time-series optimization
CREATE TABLE IF NOT EXISTS weather_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL,
    location_name VARCHAR(255) NOT NULL,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    source VARCHAR(50) NOT NULL,
    temperature DECIMAL(5, 2),
    humidity DECIMAL(5, 2),
    pressure DECIMAL(7, 2),
    wind_speed DECIMAL(5, 2),
    wind_direction DECIMAL(5, 2),
    visibility DECIMAL(6, 2),
    uv_index DECIMAL(4, 2),
    precipitation DECIMAL(6, 2),
    cloud_cover DECIMAL(5, 2),
    condition_text VARCHAR(255),
    condition_code VARCHAR(50),
    data_quality_score DECIMAL(3, 2),
    raw_data JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create hypertable for time-series data (if TimescaleDB is available)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('weather_data', 'timestamp', if_not_exists => TRUE);
    END IF;
END $$;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_weather_data_timestamp ON weather_data (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_weather_data_location ON weather_data (location_name);
CREATE INDEX IF NOT EXISTS idx_weather_data_source ON weather_data (source);
CREATE INDEX IF NOT EXISTS idx_weather_data_coordinates ON weather_data (latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_weather_data_created_at ON weather_data (created_at);

-- Create data quality monitoring table
CREATE TABLE IF NOT EXISTS data_quality_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    check_type VARCHAR(50) NOT NULL,
    data_source VARCHAR(50),
    location VARCHAR(255),
    quality_score DECIMAL(3, 2),
    issues_found INTEGER DEFAULT 0,
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dq_reports_timestamp ON data_quality_reports (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_dq_reports_type ON data_quality_reports (check_type);

-- Create API usage tracking table
CREATE TABLE IF NOT EXISTS api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    api_source VARCHAR(50) NOT NULL,
    endpoint VARCHAR(255),
    response_time_ms INTEGER,
    status_code INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    rate_limit_remaining INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_api_usage_source ON api_usage (api_source);
CREATE INDEX IF NOT EXISTS idx_api_usage_success ON api_usage (success);

-- Create cache invalidation table
CREATE TABLE IF NOT EXISTS cache_invalidation (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cache_key VARCHAR(255) NOT NULL,
    invalidated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reason VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_cache_invalidation_key ON cache_invalidation (cache_key);
CREATE INDEX IF NOT EXISTS idx_cache_invalidation_timestamp ON cache_invalidation (invalidated_at DESC);

-- Create materialized view for recent weather summary
CREATE MATERIALIZED VIEW IF NOT EXISTS recent_weather_summary AS
SELECT 
    location_name,
    latitude,
    longitude,
    source,
    AVG(temperature) as avg_temperature,
    AVG(humidity) as avg_humidity,
    AVG(pressure) as avg_pressure,
    MAX(timestamp) as last_updated,
    COUNT(*) as data_points
FROM weather_data 
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY location_name, latitude, longitude, source;

-- Create unique index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_recent_summary_unique 
ON recent_weather_summary (location_name, source);

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_weather_summary()
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY recent_weather_summary;
END;
$$;

-- Create update trigger for updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_weather_data_updated_at 
    BEFORE UPDATE ON weather_data 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO weather_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO weather_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO weather_user;
