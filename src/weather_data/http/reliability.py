from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass
from types import TracebackType


@dataclass(slots=True)
class RetryDecision:
    should_retry: bool
    delay: float = 0.0


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.2
    max_delay: float = 5.0
    jitter: float = 0.1
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504)

    def decide(
        self,
        attempt: int,
        status: int | None,
        error: BaseException | None,
    ) -> RetryDecision:
        if attempt >= self.max_attempts:
            return RetryDecision(False, 0.0)
        if error is not None:
            return RetryDecision(
                True,
                backoff_exponential_jitter(
                    attempt,
                    self.base_delay,
                    self.max_delay,
                    self.jitter,
                ),
            )
        if status is not None and status in self.retry_on_status:
            return RetryDecision(
                True,
                backoff_exponential_jitter(
                    attempt,
                    self.base_delay,
                    self.max_delay,
                    self.jitter,
                ),
            )
        return RetryDecision(False, 0.0)


def backoff_exponential_jitter(attempt: int, base: float, max_delay: float, jitter: float) -> float:
    expo = base * (2 ** (attempt - 1))
    delay = min(expo, max_delay)
    if jitter:
        jitter_amount = delay * jitter
        delay = random.uniform(max(0.0, delay - jitter_amount), delay + jitter_amount)
    return delay


class AsyncRateLimiter:
    def __init__(self, rate: float, burst: int | None = None) -> None:
        self._rate = float(rate)
        self._capacity = float(burst if burst is not None else max(1, int(math.ceil(rate))))
        self._tokens = self._capacity
        self._updated = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._updated
            self._updated = now
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                now2 = asyncio.get_event_loop().time()
                elapsed2 = now2 - self._updated
                self._updated = now2
                self._tokens = min(self._capacity, self._tokens + elapsed2 * self._rate)
            self._tokens -= 1.0


class ConcurrencyLimiter:
    def __init__(self, max_concurrent: int) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self) -> ConcurrencyLimiter:
        await self._sem.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self._sem.release()
        return False
