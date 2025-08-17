# Weather Pipeline User Guide

## Table of Contents

- [Getting Started](#getting-started)
- [Installation Guide](#installation-guide)
- [CLI Usage](#cli-usage)
- [Dashboard Usage](#dashboard-usage)
- [API Usage](#api-usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Getting Started

The Weather Pipeline is a comprehensive system for collecting, processing, and visualizing weather data. This guide will help you get started with all the major features.

### Quick Start

1. **Install the package**:
```bash
pip install -e .
```

2. **Initialize the system**:
```bash
weather-pipeline init
```

3. **Check system health**:
```bash
weather-pipeline check
```

4. **Launch the dashboard**:
```bash
weather-pipeline dashboard
```

5. **Access the dashboard**: Open http://127.0.0.1:8050 in your browser

## Installation Guide

### Prerequisites

- Python 3.10 or higher
- Git (for development installation)
- PostgreSQL (for production deployment)
- Redis (optional, for caching)

### Development Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd weather_data
```

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -e .
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env file with your API keys
```

5. **Initialize the system**:
```bash
weather-pipeline init
```

### Production Installation

For production deployment, see the [Deployment Guide](../../DEPLOYMENT.md).

### API Keys Setup

You'll need API keys from weather providers:

1. **WeatherAPI.com** (Required):
   - Sign up at: https://www.weatherapi.com/signup.aspx
   - Add to `.env`: `WEATHER_API_WEATHERAPI_KEY=your_key_here`

2. **OpenWeatherMap** (Optional):
   - Sign up at: https://openweathermap.org/api
   - Add to `.env`: `WEATHER_API_OPENWEATHER_API_KEY=your_key_here`

3. **7Timer** (No key required):
   - Free service, no registration needed

## CLI Usage

The CLI provides powerful command-line tools for managing the weather pipeline.

### Basic Commands

#### System Management

```bash
# Initialize the system
weather-pipeline init

# Check system health
weather-pipeline check

# Show system status
weather-pipeline status

# View configuration
weather-pipeline config
```

#### Data Collection

```bash
# Collect weather data for a city
weather-pipeline collect --city "London" --provider weatherapi

# Collect data for multiple cities
weather-pipeline collect --cities "London,Paris,Tokyo" --provider auto

# Collect with specific coordinates
weather-pipeline collect --lat 40.7128 --lon -74.0060 --city "New York"

# Collect forecast data
weather-pipeline collect --city "London" --forecast --days 5
```

#### Dashboard Management

```bash
# Start the dashboard
weather-pipeline dashboard

# Start with custom host/port
weather-pipeline dashboard --host 0.0.0.0 --port 8080

# Start in debug mode
weather-pipeline dashboard --debug

# Start with specific configuration
weather-pipeline dashboard --config /path/to/config.yaml
```

#### Performance Analysis

```bash
# Run performance benchmarks
weather-pipeline benchmark

# Run specific benchmark
weather-pipeline benchmark --operation groupby

# Run with custom data size
weather-pipeline benchmark --rows 10000

# Generate performance report
weather-pipeline benchmark --report --output performance_report.json
```

#### Export and Reports

```bash
# Export weather data
weather-pipeline export --format csv --output weather_data.csv

# Export with date range
weather-pipeline export --format excel --start-date 2024-01-01 --end-date 2024-01-31

# Generate summary report
weather-pipeline report --type summary --cities "London,Paris"

# Generate performance report
weather-pipeline report --type performance --output performance.pdf
```

### Advanced CLI Usage

#### Using Configuration Files

Create a configuration file `config.yaml`:

```yaml
# Weather API Configuration
api:
  weatherapi:
    key: ${WEATHER_API_WEATHERAPI_KEY}
    timeout: 30
    rate_limit: 100
  
  openweather:
    key: ${WEATHER_API_OPENWEATHER_API_KEY}
    units: metric

# Database Configuration
database:
  host: localhost
  port: 5432
  name: weather_data
  user: weather_user

# Dashboard Configuration
dashboard:
  host: 127.0.0.1
  port: 8050
  debug: false
  update_interval: 30
```

Use the configuration file:

```bash
weather-pipeline --config config.yaml dashboard
```

#### Batch Operations

Create a batch script `collect_cities.sh`:

```bash
#!/bin/bash

cities=("London" "Paris" "Tokyo" "New York" "Sydney")

for city in "${cities[@]}"; do
    echo "Collecting data for $city..."
    weather-pipeline collect --city "$city" --provider auto
    sleep 2  # Respect rate limits
done

echo "Data collection complete!"
```

#### Automation with Cron

Set up automated data collection:

```bash
# Edit crontab
crontab -e

# Add entries for automated collection
# Collect data every hour
0 * * * * /path/to/venv/bin/weather-pipeline collect --cities "London,Paris,Tokyo"

# Run health check every 15 minutes
*/15 * * * * /path/to/venv/bin/weather-pipeline check --quiet

# Generate daily reports
0 6 * * * /path/to/venv/bin/weather-pipeline report --type daily --email admin@company.com
```

## Dashboard Usage

The Weather Dashboard provides an interactive web interface for visualizing and analyzing weather data.

### Accessing the Dashboard

1. **Start the dashboard**:
```bash
weather-pipeline dashboard
```

2. **Open in browser**: http://127.0.0.1:8050

3. **Login with demo credentials**:
   - Username: `demo`
   - Password: `demo123`

### Dashboard Overview

The dashboard consists of several main sections:

#### Header
- **Title**: "Weather Data Pipeline Dashboard"
- **Status Indicator**: Shows connection status
- **User Menu**: Login/logout and user settings

#### Sidebar Controls
- **City Selection**: Multi-select dropdown for cities
- **Date Range**: Date picker for historical data
- **Parameters**: Checkboxes for weather variables
- **Export Options**: PDF and Excel download buttons

#### Main Content Tabs

### Tab 1: Time Series Analysis

**Purpose**: Visualize weather trends over time

**Features**:
- Line charts for temperature, humidity, pressure, wind speed
- Multi-city comparison with color-coded lines
- Interactive zoom, pan, and hover
- Parameter selection checkboxes

**How to Use**:
1. Select cities from the sidebar dropdown
2. Choose date range with the date picker
3. Select parameters to display (temperature, humidity, etc.)
4. Use mouse to zoom and pan on charts
5. Hover over data points for details

**Example Use Cases**:
- Compare temperature trends between cities
- Analyze seasonal patterns
- Identify weather anomalies
- Track changes over time

### Tab 2: Geographic Map

**Purpose**: View weather data on an interactive world map

**Features**:
- World map with weather station markers
- Temperature heatmap overlay
- Click markers for detailed information
- Zoom and pan controls

**How to Use**:
1. View default city markers on the map
2. Click on any marker to see current weather
3. Use zoom controls to focus on regions
4. Hover over markers for quick information

**Example Use Cases**:
- Visualize global weather patterns
- Compare regional temperatures
- Identify weather systems
- Plan travel based on weather

### Tab 3: 3D Analysis

**Purpose**: Explore relationships between weather parameters

**Features**:
- 3D surface plots showing parameter relationships
- Temperature-Pressure-Humidity correlations
- Interactive rotation and zoom
- Scatter point overlay for actual data

**How to Use**:
1. Select weather parameters for X, Y, Z axes
2. Rotate the 3D plot by dragging
3. Zoom with mouse wheel
4. Observe patterns and correlations

**Example Use Cases**:
- Analyze pressure-temperature relationships
- Identify atmospheric patterns
- Study humidity correlations
- Understand weather system dynamics

### Tab 4: Animation

**Purpose**: Watch weather evolution over time

**Features**:
- Animated time-lapse of weather patterns
- Play/pause controls
- Frame-by-frame navigation
- Speed adjustment

**How to Use**:
1. Click play to start animation
2. Use pause to stop at interesting frames
3. Drag the timeline slider for manual navigation
4. Adjust speed with the speed control

**Example Use Cases**:
- Watch storm systems develop
- Observe seasonal transitions
- Study weather pattern movements
- Create weather presentations

### User Authentication

#### User Roles

1. **Admin** (`admin` / `admin123`):
   - Full access to all features
   - Can export data
   - Can access system settings

2. **Viewer** (`viewer` / `viewer123`):
   - Read-only access
   - Can view all visualizations
   - Cannot export data

3. **Demo** (`demo` / `demo123`):
   - Limited demo access
   - Sample data only
   - Basic visualization features

#### Session Management

- Sessions expire after 2 hours of inactivity
- Automatic logout on session expiration
- Remember login option (for development)

### Export Features

#### PDF Reports

Generate comprehensive PDF reports containing:
- Executive summary
- Key weather metrics
- Chart placeholders (for manual insertion)
- Data quality assessment

**How to Export**:
1. Select cities and date range
2. Click "Export PDF" in sidebar
3. Choose report template
4. Download generated PDF

#### Excel Exports

Export detailed data in Excel format with:
- Main weather data sheet
- Summary statistics sheet
- City comparison sheet
- Professional formatting

**How to Export**:
1. Configure data selection
2. Click "Export Excel" in sidebar
3. Choose data format options
4. Download Excel file

### Real-time Features

#### Live Updates

- Dashboard updates every 30 seconds
- New data automatically refreshed
- Connection status indicator
- Real-time performance metrics

#### Weather Alerts

- Extreme weather condition notifications
- Temperature threshold alerts
- Severe weather warnings
- System health alerts

### Performance Tips

#### Optimizing Dashboard Performance

1. **Limit Data Range**: Use shorter date ranges for faster loading
2. **Select Fewer Cities**: Too many cities can slow rendering
3. **Choose Specific Parameters**: Only display needed weather parameters
4. **Use Caching**: Data is cached for 30 minutes by default

#### Browser Compatibility

- **Recommended**: Chrome, Firefox, Safari (latest versions)
- **Minimum**: Internet Explorer 11, Edge
- **Mobile**: Responsive design works on tablets and phones

## API Usage

### Python API

The Weather Pipeline provides a comprehensive Python API for programmatic access.

#### Basic Usage

```python
import asyncio
from weather_pipeline.api import WeatherAPIClient
from weather_pipeline.models import Coordinates

async def main():
    # Initialize client
    client = WeatherAPIClient(api_key="your_api_key")
    
    # Define location
    location = Coordinates(latitude=40.7128, longitude=-74.0060)
    
    # Get current weather
    weather_data = await client.get_current_weather(location)
    
    print(f"Temperature: {weather_data.temperature}°C")
    print(f"Humidity: {weather_data.humidity}%")
    print(f"Condition: {weather_data.condition}")

# Run the example
asyncio.run(main())
```

#### Advanced Usage

```python
from weather_pipeline.api import MultiProviderClient
from weather_pipeline.processing import WeatherDataProcessor

async def advanced_example():
    # Multi-provider client with fallback
    client = MultiProviderClient(
        primary_provider="weatherapi",
        fallback_providers=["7timer"]
    )
    
    # Data processor
    processor = WeatherDataProcessor()
    
    # Collect data for multiple cities
    cities = {
        "London": Coordinates(51.5074, -0.1278),
        "Paris": Coordinates(48.8566, 2.3522),
        "Tokyo": Coordinates(35.6762, 139.6503)
    }
    
    weather_data = []
    for city_name, coords in cities.items():
        data = await client.get_weather_data(coords)
        weather_data.append(data)
    
    # Process and analyze data
    analysis = processor.analyze_multi_city(weather_data)
    
    print(f"Average temperature: {analysis.avg_temperature:.1f}°C")
    print(f"Warmest city: {analysis.warmest_city}")
    print(f"Coolest city: {analysis.coolest_city}")

asyncio.run(advanced_example())
```

### REST API (Future Enhancement)

Future versions will include REST API endpoints:

```bash
# Get current weather
curl "http://localhost:8000/api/v1/weather/current?lat=40.7128&lon=-74.0060"

# Get weather forecast
curl "http://localhost:8000/api/v1/weather/forecast?city=London&days=5"

# Get historical data
curl "http://localhost:8000/api/v1/weather/history?city=Paris&start=2024-01-01&end=2024-01-31"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys
WEATHER_API_WEATHERAPI_KEY=your_weatherapi_key_here
WEATHER_API_OPENWEATHER_API_KEY=your_openweather_key_here

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=weather_data
DB_USER=weather_user
DB_PASSWORD=your_password_here

# Cache Configuration
CACHE_BACKEND=redis  # redis, memory, or disabled
CACHE_TTL=300        # Cache time-to-live in seconds
REDIS_HOST=localhost
REDIS_PORT=6379

# Dashboard Configuration
DASHBOARD_HOST=127.0.0.1
DASHBOARD_PORT=8050
DASHBOARD_DEBUG=false

# Logging Configuration
LOG_LEVEL=INFO       # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json      # json or standard
LOG_FILE=logs/weather-pipeline.log

# Performance Configuration
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30
RATE_LIMIT_CALLS=100
RATE_LIMIT_PERIOD=60

# Security Configuration
SESSION_TIMEOUT=7200  # Session timeout in seconds
MAX_LOGIN_ATTEMPTS=5
REQUIRE_HTTPS=false   # Set to true in production
```

### Configuration File

Create `config.yaml` for advanced configuration:

```yaml
# API Configuration
api:
  providers:
    weatherapi:
      key: ${WEATHER_API_WEATHERAPI_KEY}
      timeout: 30
      rate_limit:
        calls: 100
        period: 60
      retry:
        max_attempts: 3
        backoff_factor: 2
    
    openweather:
      key: ${WEATHER_API_OPENWEATHER_API_KEY}
      units: metric
      timeout: 30
    
    7timer:
      timeout: 30
      product: civil

# Database Configuration
database:
  host: ${DB_HOST:localhost}
  port: ${DB_PORT:5432}
  name: ${DB_NAME:weather_data}
  user: ${DB_USER:weather_user}
  password: ${DB_PASSWORD}
  pool_size: 10
  max_overflow: 20

# Cache Configuration
cache:
  backend: ${CACHE_BACKEND:memory}
  ttl: ${CACHE_TTL:300}
  redis:
    host: ${REDIS_HOST:localhost}
    port: ${REDIS_PORT:6379}
    db: 0

# Dashboard Configuration
dashboard:
  host: ${DASHBOARD_HOST:127.0.0.1}
  port: ${DASHBOARD_PORT:8050}
  debug: ${DASHBOARD_DEBUG:false}
  auto_reload: true
  update_interval: 30
  max_cities: 10
  
  auth:
    session_timeout: ${SESSION_TIMEOUT:7200}
    max_login_attempts: ${MAX_LOGIN_ATTEMPTS:5}
    require_https: ${REQUIRE_HTTPS:false}

# Logging Configuration
logging:
  level: ${LOG_LEVEL:INFO}
  format: ${LOG_FORMAT:json}
  file: ${LOG_FILE:logs/weather-pipeline.log}
  rotation:
    max_size: 10MB
    backup_count: 5

# Performance Configuration
performance:
  max_concurrent_requests: ${MAX_CONCURRENT_REQUESTS:10}
  request_timeout: ${REQUEST_TIMEOUT:30}
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 60
  
  benchmarking:
    default_rows: 5000
    iterations: 3
    memory_profiling: true
```

### City Configuration

Add custom cities by editing the city configuration:

```python
# In src/weather_pipeline/dashboard/data_manager.py
CUSTOM_CITIES = {
    'my_city': {
        'name': 'My City, Country',
        'coordinates': Coordinates(latitude=XX.XXXX, longitude=YY.YYYY)
    }
}
```

## Troubleshooting

### Common Issues

#### 1. Installation Problems

**Issue**: Package installation fails
```bash
ERROR: Could not install packages due to an EnvironmentError
```

**Solutions**:
- Use virtual environment: `python -m venv .venv`
- Update pip: `pip install --upgrade pip`
- Install in development mode: `pip install -e .`

#### 2. API Key Issues

**Issue**: Authentication errors
```
APIError: Invalid API key
```

**Solutions**:
- Verify API key in `.env` file
- Check key validity on provider website
- Ensure correct environment variable name
- Restart application after changing keys

#### 3. Dashboard Not Loading

**Issue**: Dashboard shows blank page or errors

**Solutions**:
- Check if port 8050 is available
- Verify Python dependencies installed
- Check browser console for JavaScript errors
- Clear browser cache and cookies

#### 4. Database Connection Issues

**Issue**: Cannot connect to database
```
OperationalError: could not connect to server
```

**Solutions**:
- Verify PostgreSQL is running
- Check database credentials in `.env`
- Ensure database exists: `createdb weather_data`
- Check firewall settings

#### 5. Performance Issues

**Issue**: Slow dashboard or API responses

**Solutions**:
- Reduce date range in dashboard
- Limit number of cities selected
- Check system resources (CPU, memory)
- Enable caching with Redis
- Optimize database queries

#### 6. Memory Usage Issues

**Issue**: High memory consumption

**Solutions**:
- Reduce batch sizes for data processing
- Enable garbage collection
- Use Polars for large datasets
- Monitor memory usage with tools

### Debug Mode

Enable debug mode for detailed error information:

```bash
# CLI debug mode
weather-pipeline --debug dashboard

# Dashboard debug mode
weather-pipeline dashboard --debug

# Environment variable
export WEATHER_DEBUG=true
```

### Log Analysis

Check logs for troubleshooting:

```bash
# View recent logs
tail -f logs/weather-pipeline.log

# Search for errors
grep ERROR logs/weather-pipeline.log

# View specific component logs
grep "dashboard" logs/weather-pipeline.log
```

### Health Checks

Use built-in health checks:

```bash
# Basic health check
weather-pipeline check

# Detailed health check
weather-pipeline check --verbose

# Check specific components
weather-pipeline check --component database
weather-pipeline check --component cache
weather-pipeline check --component api
```

## FAQ

### General Questions

**Q: What weather providers are supported?**
A: WeatherAPI.com (primary), 7Timer.info (free backup), and OpenWeatherMap (optional).

**Q: Do I need API keys?**
A: WeatherAPI.com key is required. OpenWeatherMap is optional. 7Timer requires no key.

**Q: Can I use this commercially?**
A: Yes, but check the license terms and API provider usage policies.

**Q: What Python versions are supported?**
A: Python 3.10 and higher.

### Technical Questions

**Q: How often is data updated?**
A: Dashboard updates every 30 seconds by default. Configurable via settings.

**Q: How much data can I store?**
A: Limited by your database storage capacity. PostgreSQL can handle terabytes.

**Q: Can I add custom weather providers?**
A: Yes, implement the `BaseWeatherClient` interface and register with the factory.

**Q: Is real-time streaming supported?**
A: Currently polling-based. True streaming is planned for future versions.

### Dashboard Questions

**Q: Why can't I export data?**
A: Check user permissions. Only admin and viewer roles can export.

**Q: Can I customize the dashboard?**
A: Yes, modify components in `src/weather_pipeline/dashboard/components.py`.

**Q: How do I add new cities?**
A: Edit the city configuration in `data_manager.py` or request via API.

**Q: Can I embed charts in other applications?**
A: Charts can be exported as images or HTML. Full embedding requires integration work.

### Performance Questions

**Q: Which is faster, pandas or Polars?**
A: Generally Polars is faster for large datasets. Run benchmarks to compare.

**Q: How can I improve dashboard performance?**
A: Use shorter date ranges, fewer cities, enable caching, and optimize queries.

**Q: What's the recommended server specification?**
A: Minimum: 2 CPU cores, 4GB RAM. Recommended: 4+ cores, 8GB+ RAM.

### Deployment Questions

**Q: Can I deploy to cloud platforms?**
A: Yes, supports Docker/Podman deployment on any cloud platform.

**Q: Do I need Kubernetes?**
A: Not required. Can run with simple Docker Compose or Podman Compose.

**Q: How do I scale the application?**
A: Use load balancers, database replication, and horizontal pod autoscaling.

**Q: Is HTTPS supported?**
A: Yes, configure SSL certificates in the Nginx reverse proxy.

### Data Questions

**Q: How long is data retained?**
A: Configurable. Default is unlimited retention with optional archiving.

**Q: Can I import historical data?**
A: Yes, use CLI commands or API endpoints to import CSV/JSON data.

**Q: What's the data update frequency?**
A: Real-time for current conditions, hourly for detailed data.

**Q: Can I export data in different formats?**
A: Supports CSV, Excel, JSON, and PDF exports.

### Security Questions

**Q: How are passwords stored?**
A: Hashed using bcrypt with salt.

**Q: Is API rate limiting implemented?**
A: Yes, configurable rate limiting for all API endpoints.

**Q: Can I use external authentication?**
A: Currently basic auth only. OAuth/LDAP planned for future versions.

**Q: Are there any security best practices?**
A: Use HTTPS, strong passwords, regular updates, and network security.

## Getting Help

### Support Channels

1. **Documentation**: Check this guide and API documentation
2. **Issues**: Create GitHub issues for bugs and feature requests
3. **Discussions**: Use GitHub discussions for questions
4. **Community**: Join the weather data engineering community

### Contributing

We welcome contributions! See the [contributing guide](../../CONTRIBUTING.md) for details.

### Reporting Bugs

When reporting bugs, include:
- Python version and OS
- Complete error messages
- Steps to reproduce
- Expected vs actual behavior
- Log files if relevant

### Feature Requests

For feature requests, describe:
- Use case and problem to solve
- Proposed solution
- Alternative solutions considered
- Additional context or examples
