# Weather Pipeline Architecture Documentation

## Overview

The Weather Pipeline is a production-ready data engineering system built with modern Python practices. It implements a microservices-inspired architecture with clear separation of concerns, dependency injection, and robust error handling.

## Table of Contents

- [System Architecture](#system-architecture)
- [Component Overview](#component-overview)
- [Data Flow](#data-flow)
- [API Integration](#api-integration)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Dashboard Architecture](#dashboard-architecture)
- [Security Architecture](#security-architecture)
- [Deployment Architecture](#deployment-architecture)
- [Performance Considerations](#performance-considerations)
- [Monitoring and Observability](#monitoring-and-observability)

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Weather Pipeline System                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Web Dashboard │    │   CLI Interface │    │   API Endpoints │         │
│  │   (Dash/Plotly) │    │   (Click-based) │    │   (FastAPI)     │         │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘         │
│            │                      │                      │                 │
│            └──────────────────────┼──────────────────────┘                 │
│                                   │                                        │
│  ┌─────────────────────────────────┼─────────────────────────────────────┐  │
│  │                    Core Application Layer                             │  │
│  │                                 │                                     │  │
│  │  ┌─────────────┐  ┌─────────────┴─────────────┐  ┌─────────────────┐   │  │
│  │  │    Auth     │  │    Dependency Injection   │  │   Configuration │   │  │
│  │  │   Manager   │  │        Container          │  │     Manager     │   │  │
│  │  └─────────────┘  └─────────────┬─────────────┘  └─────────────────┘   │  │
│  └─────────────────────────────────┼─────────────────────────────────────┘  │
│                                    │                                        │
│  ┌─────────────────────────────────┼─────────────────────────────────────┐  │
│  │                    Business Logic Layer                               │  │
│  │                                 │                                     │  │
│  │  ┌─────────────┐  ┌─────────────┴─────────────┐  ┌─────────────────┐   │  │
│  │  │   Weather   │  │      Data Processing      │  │   Performance   │   │  │
│  │  │   Clients   │  │    & Analytics Engine     │  │   Monitoring    │   │  │
│  │  │             │  │                           │  │                 │   │  │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ ┌─────────────┐│  │ ┌─────────────┐ │   │  │
│  │  │ │WeatherAP│ │  │ │Time     │ │Feature      ││  │ │Benchmarking │ │   │  │
│  │  │ │Client   │ │  │ │Series   │ │Engineering  ││  │ │& Metrics    │ │   │  │
│  │  │ │         │ │  │ │Analysis │ │             ││  │ │             │ │   │  │
│  │  │ ├─────────┤ │  │ ├─────────┤ ├─────────────┤│  │ ├─────────────┤ │   │  │
│  │  │ │7Timer   │ │  │ │Geospatial│ │Data Quality ││  │ │Cache        │ │   │  │
│  │  │ │Client   │ │  │ │Analysis │ │Validation   ││  │ │Performance  │ │   │  │
│  │  │ │         │ │  │ │         │ │             ││  │ │             │ │   │  │
│  │  │ ├─────────┤ │  │ ├─────────┤ ├─────────────┤│  │ ├─────────────┤ │   │  │
│  │  │ │OpenWeath│ │  │ │Streaming│ │Data         ││  │ │Memory       │ │   │  │
│  │  │ │erMap    │ │  │ │Processing│ │Lineage      ││  │ │Profiling    │ │   │  │
│  │  │ │Client   │ │  │ │         │ │             ││  │ │             │ │   │  │
│  │  │ └─────────┘ │  │ └─────────┘ └─────────────┘│  │ └─────────────┘ │   │  │
│  │  └─────────────┘  └───────────────────────────┘  └─────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                           Data Layer                                   │  │
│  │                                                                         │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │  │
│  │  │   PostgreSQL│  │    Redis    │  │   File      │  │   External APIs │ │  │
│  │  │   Database  │  │    Cache    │  │   Storage   │  │                 │ │  │
│  │  │             │  │             │  │             │  │ ┌─────────────┐ │ │  │
│  │  │ Weather     │  │ Session     │  │ Logs &      │  │ │WeatherAPI   │ │ │  │
│  │  │ Historical  │  │ Cache       │  │ Reports     │  │ │             │ │ │  │
│  │  │ Data        │  │             │  │             │  │ ├─────────────┤ │ │  │
│  │  │             │  │ Weather     │  │ Export      │  │ │7Timer       │ │ │  │
│  │  │ User        │  │ Data Cache  │  │ Files       │  │ │             │ │ │  │
│  │  │ Sessions    │  │             │  │             │  │ ├─────────────┤ │ │  │
│  │  │             │  │ Performance │  │ Performance │  │ │OpenWeather  │ │ │  │
│  │  │ Metadata    │  │ Metrics     │  │ Reports     │  │ │Map          │ │ │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │ └─────────────┘ │ │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Overview

### 1. Presentation Layer

#### Web Dashboard (Dash/Plotly)
- **Purpose**: Interactive web interface for data visualization
- **Technology**: Dash framework with Plotly charts
- **Features**:
  - Real-time weather data visualization
  - Interactive maps and charts
  - User authentication and session management
  - Export capabilities (PDF, Excel)
  - Responsive design

#### CLI Interface (Click)
- **Purpose**: Command-line interface for automation and scripting
- **Technology**: Click framework
- **Features**:
  - Data collection commands
  - System health checks
  - Performance benchmarking
  - Configuration management

#### API Endpoints (FastAPI)
- **Purpose**: RESTful API for external integrations
- **Technology**: FastAPI with automatic documentation
- **Features**:
  - Weather data endpoints
  - Authentication endpoints
  - Health check endpoints
  - Rate limiting and validation

### 2. Core Application Layer

#### Dependency Injection Container
```python
# Located in: src/weather_pipeline/core/container.py

class DIContainer:
    """Centralized dependency injection container."""
    
    def __init__(self):
        self._services = {}
        self._singletons = {}
    
    def register(self, interface: Type[T], implementation: Type[T], singleton: bool = False):
        """Register a service implementation."""
        
    def get(self, service_type: Type[T]) -> T:
        """Resolve a service dependency."""
```

#### Configuration Manager
```python
# Located in: src/weather_pipeline/config/settings.py

class WeatherConfig(BaseSettings):
    """Application configuration with environment variable support."""
    
    # API Configuration
    weatherapi_key: str = Field(..., env="WEATHER_API_WEATHERAPI_KEY")
    openweather_key: Optional[str] = Field(None, env="WEATHER_API_OPENWEATHER_API_KEY")
    
    # Database Configuration
    db_host: str = Field("localhost", env="DB_HOST")
    db_port: int = Field(5432, env="DB_PORT")
    db_name: str = Field("weather_data", env="DB_NAME")
    
    # Cache Configuration
    cache_backend: str = Field("memory", env="CACHE_BACKEND")
    cache_ttl: int = Field(300, env="CACHE_TTL")
```

#### Authentication Manager
```python
# Located in: src/weather_pipeline/dashboard/auth.py

class AuthManager:
    """Handles user authentication and session management."""
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user credentials."""
        
    def create_session(self, username: str) -> str:
        """Create user session and return session token."""
        
    def validate_session(self, session_token: str) -> Optional[str]:
        """Validate session and return username if valid."""
```

### 3. Business Logic Layer

#### Weather Clients Architecture

```python
# Base client interface
class BaseWeatherClient(ABC):
    """Abstract base class for weather API clients."""
    
    @abstractmethod
    async def get_current_weather(self, location: Coordinates) -> WeatherDataPoint:
        """Get current weather data."""
        
    @abstractmethod
    async def get_forecast(self, location: Coordinates, days: int) -> List[WeatherDataPoint]:
        """Get weather forecast."""
```

**Client Implementations:**

1. **WeatherAPIClient**: Primary weather data provider
2. **SevenTimerClient**: Free backup provider
3. **OpenWeatherMapClient**: Alternative commercial provider
4. **MultiProviderClient**: Intelligent provider management with fallback

#### Circuit Breaker Pattern
```python
# Located in: src/weather_pipeline/core/resilience.py

class CircuitBreaker:
    """Implements circuit breaker pattern for resilient API calls."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
```

#### Data Processing Pipeline

```python
# Data processing architecture
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Weather   │───▶│   Validation    │───▶│   Enrichment    │
│     Data        │    │   & Cleaning    │    │   & Features    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│   Aggregated    │◀───│   Analysis      │◀────────────┘
│     Results     │    │   & Insights    │
└─────────────────┘    └─────────────────┘
```

**Processing Components:**

1. **TimeSeriesAnalysis**: Trend detection and forecasting
2. **GeospatialAnalysis**: Location-based processing
3. **FeatureEngineering**: Derived metrics and rolling windows
4. **DataQualityValidation**: Quality checks and scoring
5. **StreamingProcessor**: Real-time data processing

### 4. Data Layer

#### Database Schema (PostgreSQL)

```sql
-- Weather data table
CREATE TABLE weather_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    latitude DECIMAL(10, 7) NOT NULL,
    longitude DECIMAL(10, 7) NOT NULL,
    temperature DECIMAL(5, 2),
    humidity INTEGER,
    pressure DECIMAL(7, 2),
    wind_speed DECIMAL(5, 2),
    wind_direction INTEGER,
    precipitation DECIMAL(5, 2),
    visibility INTEGER,
    uv_index DECIMAL(3, 1),
    condition VARCHAR(100),
    city VARCHAR(100),
    country VARCHAR(10),
    data_source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User sessions table
CREATE TABLE user_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

-- Performance metrics table
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10, 4),
    metric_unit VARCHAR(20),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);
```

#### Cache Layer (Redis)

```python
# Cache strategy implementation
class CacheManager:
    """Manages distributed caching with Redis backend."""
    
    def __init__(self, backend: str = "redis"):
        self.backend = backend
        self.cache_strategies = {
            "weather_data": CacheStrategy(ttl=300, compression=True),
            "user_sessions": CacheStrategy(ttl=7200, persistent=True),
            "performance_metrics": CacheStrategy(ttl=60, aggregation=True)
        }
    
    async def get(self, key: str, strategy: str = "default") -> Optional[Any]:
        """Retrieve cached data with strategy-specific handling."""
        
    async def set(self, key: str, value: Any, strategy: str = "default") -> bool:
        """Store data with strategy-specific handling."""
```

## Data Flow

### 1. Weather Data Collection Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Dashboard    │    │       CLI       │    │    Scheduler    │
│   User Request  │    │     Command     │    │  Cron/Timer Job │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
          ┌─────────────────────▼─────────────────────┐
          │          DIContainer.get_client()        │
          └─────────────────────┬─────────────────────┘
                                │
          ┌─────────────────────▼─────────────────────┐
          │        MultiProviderClient               │
          │     (Circuit Breaker + Fallback)         │
          └─────────┬─────────────────────┬─────────┘
                    │                     │
    ┌───────────────▼─────────────┐       ▼─────────────┐
    │     WeatherAPIClient       │     7TimerClient   │
    │   (Primary Provider)       │   (Backup Provider) │
    └───────────┬─────────────────┘       ┬─────────────┘
                │                         │
                └─────────────┬───────────┘
                              │
          ┌─────────────────────▼─────────────────────┐
          │           Data Validation                │
          │        (Pydantic Models)                 │
          └─────────────────────┬─────────────────────┘
                                │
          ┌─────────────────────▼─────────────────────┐
          │          Data Processing                 │
          │    (Quality Checks + Enrichment)         │
          └─────────┬─────────────────────┬─────────┘
                    │                     │
          ┌─────────▼───────┐   ┌─────────▼─────────┐
          │  Cache Storage  │   │ Database Storage  │
          │    (Redis)      │   │   (PostgreSQL)    │
          └─────────────────┘   └───────────────────┘
```

### 2. Dashboard Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     User        │    │    Browser      │    │     Dash        │
│   Interaction   │───▶│   JavaScript    │───▶│   Application   │
└─────────────────┘    └─────────────────┘    └─────────┬───────┘
                                                        │
                          ┌─────────────────────────────▼─────────────────────────────┐
                          │                  Callback Handler                        │
                          │              (Authentication Check)                      │
                          └─────────────────────────────┬─────────────────────────────┘
                                                        │
                          ┌─────────────────────────────▼─────────────────────────────┐
                          │                DataManager.get_weather_data()            │
                          └─────────┬─────────────────────────────┬─────────────────────┘
                                    │                             │
                    ┌───────────────▼─────────────┐    ┌─────────▼─────────┐
                    │        Cache Check          │    │   Real-time       │
                    │        (30 min TTL)         │    │   API Call        │
                    └───────────────┬─────────────┘    └─────────┬─────────┘
                                    │                             │
                                    └─────────────┬───────────────┘
                                                  │
                          ┌─────────────────────▼─────────────────────┐
                          │          Components.create_plot()        │
                          │       (Plotly Chart Generation)          │
                          └─────────────────────┬─────────────────────┘
                                                │
                          ┌─────────────────────▼─────────────────────┐
                          │            JSON Response                 │
                          │        (Chart Data + Layout)             │
                          └─────────────────────┬─────────────────────┘
                                                │
┌─────────────────┐    ┌─────────────────┐    ┌▼─────────────────┐
│   Updated       │◀───│    Browser      │◀───│     Client      │
│ Visualization   │    │   Rendering     │    │    Response     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 3. Performance Monitoring Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   Performance   │    │   Benchmark     │
│   Operations    │───▶│   Monitoring    │───▶│   Analysis      │
└─────────────────┘    └─────────────────┘    └─────────┬───────┘
                                                        │
                          ┌─────────────────────────────▼─────────────────────────────┐
                          │              PerformanceComparator                      │
                          │           (pandas vs Polars benchmarks)                 │
                          └─────────────────────┬─────────────────────┬─────────────────┘
                                                │                     │
                    ┌─────────────────────────▼─────────────┐    ┌───▼─────────────────┐
                    │        Memory Profiling             │    │    Time Profiling   │
                    │      (psutil monitoring)            │    │   (perf_counter)    │
                    └─────────────────────┬─────────────────┘    └───┬─────────────────┘
                                          │                          │
                                          └─────────┬────────────────┘
                                                    │
                          ┌─────────────────────────▼─────────────────────────────┐
                          │               Report Generation                      │
                          │         (Performance recommendations)                │
                          └─────────────────────┬─────────────────────────────────┘
                                                │
                    ┌─────────────────────┬─────▼────┬─────────────────────────┐
                    │                     │          │                         │
          ┌─────────▼───────┐   ┌─────────▼─────────┐   ┌─────────▼─────────┐
          │    Database     │   │       Cache       │   │      File         │
          │    Storage      │   │     Storage       │   │     Export        │
          └─────────────────┘   └───────────────────┘   └───────────────────┘
```

## API Integration

### External API Integration Architecture

```python
# API client factory pattern
class WeatherClientFactory:
    """Factory for creating weather API clients."""
    
    @staticmethod
    def create_client(provider: str, config: WeatherConfig) -> BaseWeatherClient:
        """Create appropriate client based on provider."""
        
        clients = {
            "weatherapi": WeatherAPIClient,
            "7timer": SevenTimerClient,
            "openweather": OpenWeatherMapClient
        }
        
        client_class = clients.get(provider)
        if not client_class:
            raise ValueError(f"Unknown provider: {provider}")
            
        return client_class(config)
```

### Rate Limiting Strategy

```python
# Rate limiting implementation
class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, calls_per_period: int, period_seconds: int):
        self.calls_per_period = calls_per_period
        self.period_seconds = period_seconds
        self.tokens = calls_per_period
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire a token for API call."""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Refill tokens based on elapsed time
            tokens_to_add = elapsed * (self.calls_per_period / self.period_seconds)
            self.tokens = min(self.calls_per_period, self.tokens + tokens_to_add)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            return False
```

## Dashboard Architecture

### Component-Based Architecture

```python
# Dashboard component structure
src/weather_pipeline/dashboard/
├── app.py                 # Main Dash application
├── components.py          # Reusable visualization components
├── auth.py               # Authentication management
├── data_manager.py       # Data fetching and caching
├── exports.py            # Report generation
├── realtime.py           # Live updates and metrics
└── layouts/
    ├── header.py         # Header component
    ├── sidebar.py        # Sidebar controls
    └── main.py           # Main content area
```

### State Management

```python
# Client-side state management
class DashboardState:
    """Manages dashboard application state."""
    
    def __init__(self):
        self.current_user: Optional[str] = None
        self.selected_cities: List[str] = []
        self.selected_parameters: List[str] = []
        self.date_range: Tuple[datetime, datetime] = (
            datetime.now() - timedelta(days=7),
            datetime.now()
        )
        self.refresh_interval: int = 30  # seconds
    
    def update_state(self, **kwargs):
        """Update dashboard state with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
```

### Real-time Updates

```python
# Real-time update mechanism
class RealTimeUpdater:
    """Handles real-time dashboard updates."""
    
    def __init__(self, update_interval: int = 30):
        self.update_interval = update_interval
        self.active_sessions: Set[str] = set()
        self.metrics = DashboardMetrics()
    
    async def start_updates(self):
        """Start the real-time update loop."""
        while True:
            try:
                await self.update_all_sessions()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Real-time update error: {e}")
    
    async def update_all_sessions(self):
        """Update data for all active sessions."""
        for session_id in self.active_sessions:
            await self.update_session_data(session_id)
```

## Security Architecture

### Authentication & Authorization

```python
# Security implementation
class SecurityManager:
    """Centralized security management."""
    
    def __init__(self):
        self.password_hasher = bcrypt.BCrypt()
        self.session_manager = SessionManager()
        self.rate_limiter = RateLimiter(calls_per_period=100, period_seconds=60)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self.password_hasher.hash(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return self.password_hasher.verify(password, hashed)
    
    def create_session_token(self, username: str) -> str:
        """Create secure session token."""
        payload = {
            "username": username,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=2)).isoformat()
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
```

### Data Protection

1. **Input Validation**: All inputs validated using Pydantic models
2. **SQL Injection Prevention**: Parameterized queries with SQLAlchemy
3. **XSS Protection**: Output encoding in dashboard templates
4. **CSRF Protection**: Built-in Dash CSRF protection
5. **Rate Limiting**: API and dashboard request limiting

## Deployment Architecture

### Container Architecture

```yaml
# Docker/Podman compose structure
version: '3.8'
services:
  weather-app:
    build: .
    ports:
      - "8050:8050"
    depends_on:
      - postgres
      - redis
    environment:
      - DB_HOST=postgres
      - CACHE_BACKEND=redis
    
  postgres:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=weather_data
      - POSTGRES_USER=weather_user
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    
  redis:
    image: redis:6-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./docker/nginx/ssl:/etc/nginx/ssl
    depends_on:
      - weather-app
```

### Production Deployment

```bash
# Production deployment flow
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Development   │    │      Build      │    │   Production    │
│   Environment   │───▶│    Pipeline     │───▶│   Deployment    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
    ┌─────▼─────┐           ┌─────▼─────┐           ┌─────▼─────┐
    │   Code    │           │  Docker   │           │   Podman  │
    │ Repository│           │   Build   │           │  Cluster  │
    └───────────┘           └───────────┘           └───────────┘
```

## Performance Considerations

### Database Optimization

```sql
-- Optimized indexes for weather data
CREATE INDEX idx_weather_data_timestamp ON weather_data(timestamp DESC);
CREATE INDEX idx_weather_data_location ON weather_data(latitude, longitude);
CREATE INDEX idx_weather_data_city ON weather_data(city);
CREATE INDEX idx_weather_data_source ON weather_data(data_source);

-- Composite index for common queries
CREATE INDEX idx_weather_data_composite ON weather_data(city, timestamp DESC, data_source);
```

### Caching Strategy

```python
# Multi-level caching
class CacheHierarchy:
    """Implements multi-level caching strategy."""
    
    def __init__(self):
        self.l1_cache = {}  # In-memory cache (fastest)
        self.l2_cache = RedisCache()  # Distributed cache (fast)
        self.l3_cache = DatabaseCache()  # Persistent cache (slowest)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get data with cache hierarchy fallback."""
        # Try L1 cache first
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # Try L2 cache
        data = await self.l2_cache.get(key)
        if data:
            self.l1_cache[key] = data  # Promote to L1
            return data
        
        # Try L3 cache
        data = await self.l3_cache.get(key)
        if data:
            await self.l2_cache.set(key, data)  # Promote to L2
            self.l1_cache[key] = data  # Promote to L1
            return data
        
        return None
```

### Async Processing

```python
# Concurrent data processing
class AsyncProcessor:
    """Handles concurrent weather data processing."""
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = aiohttp.ClientSession()
    
    async def process_locations(self, locations: List[Coordinates]) -> List[WeatherDataPoint]:
        """Process multiple locations concurrently."""
        tasks = []
        for location in locations:
            task = self.process_single_location(location)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, WeatherDataPoint)]
    
    async def process_single_location(self, location: Coordinates) -> WeatherDataPoint:
        """Process single location with semaphore limiting."""
        async with self.semaphore:
            return await self.fetch_weather_data(location)
```

## Monitoring and Observability

### Metrics Collection

```python
# Metrics collection system
class MetricsCollector:
    """Collects application metrics for monitoring."""
    
    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.timers = defaultdict(list)
    
    def increment_counter(self, metric_name: str, value: int = 1):
        """Increment a counter metric."""
        self.counters[metric_name] += value
    
    def set_gauge(self, metric_name: str, value: float):
        """Set a gauge metric value."""
        self.gauges[metric_name] = value
    
    def record_timing(self, metric_name: str, duration: float):
        """Record a timing measurement."""
        self.timers[metric_name].append(duration)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "timers": {
                name: {
                    "count": len(times),
                    "avg": sum(times) / len(times) if times else 0,
                    "max": max(times) if times else 0,
                    "min": min(times) if times else 0
                }
                for name, times in self.timers.items()
            }
        }
```

### Health Checks

```python
# Health check system
class HealthChecker:
    """Performs system health checks."""
    
    def __init__(self, container: DIContainer):
        self.container = container
        self.checks = [
            self.check_database_connection,
            self.check_cache_connection,
            self.check_api_availability,
            self.check_disk_space,
            self.check_memory_usage
        ]
    
    async def check_health(self) -> HealthStatus:
        """Perform all health checks."""
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for check in self.checks:
            try:
                check_name = check.__name__.replace("check_", "")
                result = await check()
                results[check_name] = result
                
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status != HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                results[check.__name__] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {e}"
                )
                overall_status = HealthStatus.UNHEALTHY
        
        return HealthReport(overall_status=overall_status, checks=results)
```

### Logging Architecture

```python
# Structured logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        },
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filename": "logs/weather-pipeline.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "weather_pipeline": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False
        }
    }
}
```

## Technology Stack Summary

### Core Technologies
- **Python 3.11+**: Primary programming language
- **asyncio/aiohttp**: Asynchronous programming
- **Pydantic v2**: Data validation and serialization
- **FastAPI**: Web framework for APIs
- **Dash/Plotly**: Interactive dashboard framework

### Data Technologies
- **PostgreSQL**: Primary database
- **Redis**: Caching and session storage
- **Polars**: High-performance data processing
- **pandas**: Data analysis and manipulation
- **Apache Arrow**: Columnar data format

### Infrastructure Technologies
- **Podman/Docker**: Containerization
- **Nginx**: Reverse proxy and load balancing
- **prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

### Development Technologies
- **pytest**: Testing framework
- **mypy**: Static type checking
- **ruff**: Fast Python linter
- **pre-commit**: Git hooks for code quality

This architecture provides a solid foundation for a production-ready weather data pipeline with excellent scalability, maintainability, and observability characteristics.
