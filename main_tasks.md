# Advanced Weather Data Engineering Challenge

## Overview
Build a comprehensive weather data pipeline using modern Python practices and data engineering patterns. This project will demonstrate your ability to work with APIs, process data at scale, implement quality controls, and create production-ready code with proper testing and deployment strategies.

## Core Requirements

### 1. Async Data Collection & Processing
- **Upgrade the API client** to use `asyncio` and `aiohttp` for concurrent data fetching
- Implement **rate limiting** and **retry logic** with exponential backoff
- Add support for **multiple weather APIs** (7timer, OpenWeatherMap, WeatherAPI) with a unified interface
- Create a **data source factory pattern** for easy API switching
- Handle **API failures gracefully** with circuit breaker pattern

### 2. Data Validation & Quality Assurance
- Implement **Pydantic models** for strict data validation and serialization
- Add **data quality checks**: missing values, outliers, temporal consistency
- Create **data lineage tracking** to monitor data flow and transformations
- Implement **schema evolution** handling for API changes
- Add **data freshness monitoring** with configurable thresholds

### 3. Modern Python Architecture
- Use **type hints throughout** with `mypy` compliance
- Implement **context managers** for resource management
- Apply **dependency injection** patterns for testability
- Use **dataclasses/Pydantic** for configuration management
- Follow **SOLID principles** in class design

### 4. Advanced Data Processing
- Extend beyond basic statistics to include:
  - **Time series analysis**: trend detection, seasonality, anomaly detection
  - **Geospatial analysis**: weather pattern clustering, interpolation
  - **Feature engineering**: derived metrics, rolling windows, lag features
- Use **Polars** alongside pandas for performance comparisons
- Implement **streaming data processing** patterns for real-time scenarios
- Add **data caching strategies** with Redis integration

### 5. Interactive Visualizations & Dashboard
- Create a **Streamlit/Dash dashboard** with real-time updates
- Implement **interactive plots** using Plotly with:
  - Time series with zoom/pan capabilities
  - Geographic heatmaps and choropleth maps
  - 3D surface plots for temperature/pressure relationships
  - Animated visualizations showing weather evolution
- Add **export functionality** for reports (PDF, Excel)
- Implement **user authentication** and **session management**

### 6. Comprehensive Testing Strategy
- **Unit tests** with >90% coverage using pytest
- **Integration tests** for API endpoints with mocking
- **Property-based testing** using Hypothesis
- **Performance benchmarks** and **load testing**
- **Contract testing** for API responses
- **End-to-end testing** of the complete pipeline

### 7. Code Quality & Documentation
- Set up **pre-commit hooks** with black, isort, flake8, mypy
- Generate **API documentation** with FastAPI/Sphinx
- Implement **logging** with structured JSON format
- Add **monitoring and alerting** with Prometheus metrics
- Create comprehensive **README** with setup instructions

### 8. Containerization & Deployment
- Create **multi-stage Containerfile** (Podman-compatible, Dockerfile syntax) with optimized Python images
- Set up **podman-compose** for local development environment
- Implement **health checks** and **graceful shutdown**
- Add **environment-specific configurations**
- Create **CI/CD pipeline** example with GitHub Actions

## Technical Stack Recommendations

### Core Libraries
- **httpx/aiohttp**: Async HTTP clients
- **Pydantic v2**: Data validation and settings
- **Polars**: High-performance DataFrame library
- **FastAPI**: API framework (if building web service)
- **Streamlit/Panel**: Interactive dashboard
- **Plotly/Altair**: Modern visualization libraries

### Data & Analytics
- **DuckDB**: In-process analytical database
- **Apache Arrow**: Columnar data format
- **Scikit-learn**: Machine learning features
- **Statsmodels**: Time series analysis
- **GeoPandas**: Geospatial data processing

### Infrastructure & DevOps
- **uv**: Dependency management
- **Podman**: Containerization (Docker-compatible CLI)
- **pytest + pytest-asyncio**: Testing framework
- **pre-commit**: Code quality hooks
- **Ruff**: Fast Python linter/formatter

## Deliverables

1. **Source Code**: Well-structured, documented, and tested codebase
2. **Interactive Dashboard**: Deployed application with live data
3. **Documentation**: Architecture decisions, API docs, deployment guide
4. **Performance Report**: Benchmarks, scalability analysis, optimization insights
5. **Testing Suite**: Comprehensive test coverage with CI/CD integration
6. **Podman Setup**: Complete containerized environment

## Evaluation Criteria

- **Code Quality**: Clean, maintainable, well-documented code
- **Architecture**: Scalable, testable, and extensible design
- **Performance**: Efficient data processing and API usage
- **Testing**: Comprehensive test coverage and quality
- **Innovation**: Creative use of modern tools and techniques
- **Production Readiness**: Monitoring, error handling, deployment strategy
