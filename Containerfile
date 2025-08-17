# Multi-stage Containerfile for Weather Data Pipeline
# Stage 1: Build dependencies and compile requirements
FROM python:3.11-slim as builder

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Create app directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Install dependencies to a virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install -e .

# Stage 2: Runtime image
FROM python:3.11-slim as runtime

# Set environment variables for runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    WEATHER_ENV=production \
    LOG_FORMAT=json

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r weather && useradd -r -g weather weather

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app directory and set ownership
WORKDIR /app
RUN chown -R weather:weather /app

# Copy application code
COPY --chown=weather:weather src/ ./src/
COPY --chown=weather:weather pyproject.toml ./
COPY --chown=weather:weather README.md ./

# Create data directory
RUN mkdir -p /app/data && chown -R weather:weather /app/data

# Switch to non-root user
USER weather

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -m src.weather_pipeline.core.health || exit 1

# Expose ports
EXPOSE 8050 8000

# Default command
CMD ["python", "-m", "src.weather_pipeline", "dashboard", "--host", "0.0.0.0", "--port", "8050"]
