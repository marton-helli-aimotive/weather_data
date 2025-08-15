"""Circuit breaker pattern implementation for API resilience."""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker."""
    
    failure_threshold: int = Field(default=5, description="Number of failures to open circuit")
    success_threshold: int = Field(default=3, description="Number of successes to close circuit from half-open")
    timeout: float = Field(default=60.0, description="Timeout in seconds before trying half-open")
    expected_exception: type = Field(default=Exception, description="Exception type that triggers circuit breaker")


class CircuitBreakerMetrics(BaseModel):
    """Metrics for circuit breaker monitoring."""
    
    failure_count: int = Field(default=0, description="Current consecutive failure count")
    success_count: int = Field(default=0, description="Current consecutive success count")
    total_requests: int = Field(default=0, description="Total requests made")
    total_failures: int = Field(default=0, description="Total failures")
    total_successes: int = Field(default=0, description="Total successes")
    last_failure_time: Optional[float] = Field(default=None, description="Timestamp of last failure")
    state_changed_time: float = Field(default_factory=time.time, description="When state last changed")


class CircuitBreaker:
    """Circuit breaker implementation for resilient API calls."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            await self._check_state()
            
            if self.state == CircuitBreakerState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Last failure: {self.metrics.last_failure_time}"
                )
        
        # Execute the function
        self.metrics.total_requests += 1
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await self._on_success()
            return result
            
        except self.config.expected_exception as e:
            await self._on_failure(e)
            raise
        except Exception as e:
            # Unexpected exceptions don't affect circuit breaker
            raise
    
    async def _check_state(self) -> None:
        """Check and update circuit breaker state."""
        current_time = time.time()
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if timeout has passed to try half-open
            if (self.metrics.last_failure_time and 
                current_time - self.metrics.last_failure_time >= self.config.timeout):
                self.state = CircuitBreakerState.HALF_OPEN
                self.metrics.success_count = 0
                self.metrics.state_changed_time = current_time
    
    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self.metrics.failure_count = 0
            self.metrics.total_successes += 1
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.metrics.success_count += 1
                
                # Close circuit if enough successes
                if self.metrics.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.metrics.state_changed_time = time.time()
    
    async def _on_failure(self, exception: Exception) -> None:
        """Handle failed call."""
        async with self._lock:
            self.metrics.failure_count += 1
            self.metrics.total_failures += 1
            self.metrics.last_failure_time = time.time()
            
            # Reset success count on any failure
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.metrics.success_count = 0
                self.state = CircuitBreakerState.OPEN
                self.metrics.state_changed_time = time.time()
            
            # Open circuit if failure threshold reached
            elif (self.state == CircuitBreakerState.CLOSED and 
                  self.metrics.failure_count >= self.config.failure_threshold):
                self.state = CircuitBreakerState.OPEN
                self.metrics.state_changed_time = time.time()
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self.metrics.model_copy()
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()


class CircuitBreakerRegistry:
    """Registry to manage multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]
    
    def get_all_breakers(self) -> dict[str, CircuitBreaker]:
        """Get all registered circuit breakers."""
        return self._breakers.copy()
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()
