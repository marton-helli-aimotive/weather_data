# Weather Data Pipeline

A comprehensive weather data engineering pipeline built with modern Python practices.

## Overview

This project implements a production-ready weather data pipeline that:

- Collects data from multiple weather APIs asynchronously
- Validates and processes data using modern Python libraries
- Provides interactive visualizations through a web dashboard
- Implements comprehensive testing, monitoring, and deployment strategies

## Features

- **Async Data Collection**: Concurrent API calls with rate limiting and retry logic
- **Data Validation**: Strict Pydantic models with comprehensive validation
- **Modern Architecture**: Dependency injection, type hints, and SOLID principles
- **Quality Assurance**: >90% test coverage with multiple testing strategies
- **Production Ready**: Containerized deployment with monitoring and logging

## Quick Start

### Prerequisites

- Python 3.10+
- uv (package manager)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd weather_data
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

4. Initialize the environment:
```bash
weather-pipeline init
```

5. Check system health:
```bash
weather-pipeline check
```

## Project Structure

```
src/weather_pipeline/
├── api/           # API clients and integrations
├── config/        # Configuration management
├── core/          # Core utilities and DI container
├── models/        # Data models and schemas
├── processing/    # Data processing and analytics
└── cli.py         # Command-line interface
```

## Configuration

The application uses environment variables and `.env` files for configuration. Key settings include:

- `WEATHER_API_WEATHERAPI_KEY`: WeatherAPI key
- `WEATHER_API_OPENWEATHER_API_KEY`: OpenWeatherMap API key (optional)
- `DB_HOST`: Database host (default: localhost)
- `LOG_LEVEL`: Logging level (default: INFO)

## Development

### Code Quality

This project uses several tools to maintain code quality:

- **ruff**: Fast Python linter and formatter
- **mypy**: Static type checking
- **black**: Code formatting
- **pre-commit**: Git hooks for code quality

Install pre-commit hooks:
```bash
pre-commit install
```

### Testing

Run tests with coverage:
```bash
pytest --cov=weather_pipeline
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and quality checks
5. Submit a pull request

## Architecture

The pipeline follows modern software engineering practices:

- **Dependency Injection**: Testable and modular components
- **Async-First**: Non-blocking I/O operations
- **Type Safety**: Complete type hints with mypy compliance
- **Configuration Management**: Environment-based settings
- **Structured Logging**: JSON logging with context
- **Error Handling**: Circuit breakers and retry logic

For detailed architecture documentation, see the `docs/` directory.
