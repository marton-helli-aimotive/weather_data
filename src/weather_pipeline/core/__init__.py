"""Core utilities and dependency injection for the weather pipeline."""

from .container import (
    DIContainer,
    DependencyInjectionError,
    get_container,
    get_optional_service,
    get_service,
    register_factory,
    register_singleton,
    register_transient,
)
from .logging import configure_logging, get_logger, LogContext, log_api_request, log_data_processing, log_function_call
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerState,
    circuit_breaker_registry,
)
from .resilience import (
    RateLimiter,
    RateLimiterConfig,
    RetryConfig,
    RetryHandler,
    ResilientAPIClient,
    rate_limiter_registry,
)

__all__ = [
    # Logging
    "configure_logging",
    "get_logger", 
    "LogContext",
    "log_api_request",
    "log_data_processing",
    "log_function_call",
    
    # Dependency Injection
    "DIContainer",
    "DependencyInjectionError",
    "get_container",
    "get_optional_service", 
    "get_service",
    "register_factory",
    "register_singleton",
    "register_transient",
    
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig", 
    "CircuitBreakerError",
    "CircuitBreakerState",
    "circuit_breaker_registry",
    
    # Resilience
    "RateLimiter",
    "RateLimiterConfig",
    "RetryConfig",
    "RetryHandler",
    "ResilientAPIClient",
    "rate_limiter_registry",
]
