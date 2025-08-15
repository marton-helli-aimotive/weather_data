"""Structured logging configuration for the weather pipeline."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.types import FilteringBoundLogger

from ..config.settings import LogLevel, get_settings


def configure_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.logging.level.value),
    )
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    if settings.logging.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.logging.level.value)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure file logging if path is specified
    if settings.logging.file_path:
        setup_file_logging(settings.logging.file_path)


def setup_file_logging(log_file: Path) -> None:
    """Set up rotating file handler for logging."""
    settings = get_settings()
    
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=_parse_size(settings.logging.max_file_size),
        backupCount=settings.logging.backup_count,
        encoding="utf-8",
    )
    
    # Set formatter for file handler
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


def _parse_size(size_str: str) -> int:
    """Parse size string like '10MB' to bytes."""
    size_str = size_str.upper()
    
    if size_str.endswith("KB"):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith("MB"):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith("GB"):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


# Context managers for logging contexts
class LogContext:
    """Context manager for adding structured logging context."""
    
    def __init__(self, **context: Any) -> None:
        self.context = context
    
    def __enter__(self) -> None:
        structlog.contextvars.bind_contextvars(**self.context)
    
    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_function_call(func_name: str, **kwargs: Any) -> LogContext:
    """Log context for function calls."""
    return LogContext(
        function=func_name,
        **kwargs
    )


def log_api_request(
    provider: str,
    endpoint: str,
    method: str = "GET",
    **kwargs: Any
) -> LogContext:
    """Log context for API requests."""
    return LogContext(
        api_provider=provider,
        endpoint=endpoint,
        method=method,
        **kwargs
    )


def log_data_processing(
    operation: str,
    record_count: Optional[int] = None,
    **kwargs: Any
) -> LogContext:
    """Log context for data processing operations."""
    context = {
        "operation": operation,
        **kwargs
    }
    if record_count is not None:
        context["record_count"] = record_count
    
    return LogContext(**context)
