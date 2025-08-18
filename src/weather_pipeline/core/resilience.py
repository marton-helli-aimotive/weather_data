"""Rate limiting and retry logic for API calls."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Callable, NoReturn, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class RateLimiterConfig(BaseModel):
    """Configuration for rate limiter."""
    
    max_requests: int = Field(default=10, description="Maximum requests per window")
    window_size: float = Field(default=60.0, description="Time window in seconds")
    burst_size: Optional[int] = Field(default=None, description="Maximum burst size (defaults to max_requests)")


class RetryConfig(BaseModel):
    """Configuration for retry logic."""
    
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    base_delay: float = Field(default=1.0, description="Base delay in seconds")
    max_delay: float = Field(default=60.0, description="Maximum delay in seconds")
    exponential_base: float = Field(default=2.0, description="Exponential backoff base")
    jitter: bool = Field(default=True, description="Add random jitter to delays")
    retry_exceptions: tuple[type[Exception], ...] = Field(default=(Exception,), description="Exception types to retry on")


class RateLimiter:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, name: str, config: RateLimiterConfig):
        self.name = name
        self.config = config
        self.burst_size = config.burst_size or config.max_requests
        
        # Token bucket state
        self.tokens = float(self.burst_size)
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens from the bucket, waiting if necessary."""
        async with self._lock:
            await self._refill_tokens()
            
            while self.tokens < tokens:
                # Calculate wait time for next token
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed * (self.config.window_size / self.config.max_requests)
                
                # Release lock during wait
                self._lock.release()
                await asyncio.sleep(wait_time)
                await self._lock.acquire()
                
                await self._refill_tokens()
            
            self.tokens -= tokens
    
    async def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        
        # Calculate tokens to add
        tokens_to_add = elapsed * (self.config.max_requests / self.config.window_size)
        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
        self.last_update = now
    
    def get_available_tokens(self) -> float:
        """Get currently available tokens (approximate)."""
        return self.tokens


class RetryHandler:
    """Handles retry logic with exponential backoff."""
    
    def __init__(self, name: str, config: RetryConfig):
        self.name = name
        self.config = config
    
    async def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                # Check if this exception type should trigger retry
                if not isinstance(e, self.config.retry_exceptions):
                    raise
                
                # Don't retry on last attempt
                if attempt >= self.config.max_retries:
                    break
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.config.base_delay * (self.config.exponential_base ** attempt),
                    self.config.max_delay
                )
                
                # Add jitter if enabled
                if self.config.jitter:
                    delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
                
                await asyncio.sleep(delay)
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Retry handler '{self.name}' failed without exception")


class ResilientAPIClient:
    """Base class combining rate limiting, retries, and circuit breaking."""
    
    def __init__(
        self,
        name: str,
        rate_limiter_config: Optional[RateLimiterConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.name = name
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            name=f"{name}_rate_limiter",
            config=rate_limiter_config or RateLimiterConfig()
        )
        
        # Initialize retry handler
        self.retry_handler = RetryHandler(
            name=f"{name}_retry_handler",
            config=retry_config or RetryConfig()
        )
    
    async def execute_with_resilience(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """Execute function with rate limiting and retry logic."""
        # Acquire rate limit token
        await self.rate_limiter.acquire()
        
        # Execute with retry logic
        return await self.retry_handler.execute(func, *args, **kwargs)


class RateLimiterRegistry:
    """Registry to manage multiple rate limiters."""
    
    def __init__(self) -> None:
        self._limiters: dict[str, RateLimiter] = {}
    
    def get_limiter(self, name: str, config: RateLimiterConfig) -> RateLimiter:
        """Get or create a rate limiter."""
        if name not in self._limiters:
            self._limiters[name] = RateLimiter(name, config)
        return self._limiters[name]
    
    def get_all_limiters(self) -> dict[str, RateLimiter]:
        """Get all registered rate limiters."""
        return self._limiters.copy()


# Global registry instance
rate_limiter_registry = RateLimiterRegistry()
