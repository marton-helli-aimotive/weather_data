"""Weather data core package.

Exposes HTTP reliability primitives (M02) and provider interfaces.
"""

from .http.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from .http.interfaces import AsyncHttpClient
from .http.reliability import (
    AsyncRateLimiter,
    ConcurrencyLimiter,
    RetryDecision,
    RetryPolicy,
    backoff_exponential_jitter,
)
from .http.reliable_client import MetricsHook, ReliableHttpClient
from .http.types import HttpError, HttpRequest, HttpResponse, RetryError, TimeoutError

__all__ = [
    "HttpRequest",
    "HttpResponse",
    "HttpError",
    "TimeoutError",
    "RetryError",
    "AsyncHttpClient",
    "RetryPolicy",
    "RetryDecision",
    "backoff_exponential_jitter",
    "AsyncRateLimiter",
    "ConcurrencyLimiter",
    "CircuitBreaker",
    "CircuitState",
    "CircuitOpenError",
    "ReliableHttpClient",
    "MetricsHook",
]
