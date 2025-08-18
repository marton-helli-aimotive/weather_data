from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from .circuit_breaker import CircuitBreaker, CircuitOpenError
from .interfaces import AsyncHttpClient
from .reliability import AsyncRateLimiter, ConcurrencyLimiter, RetryPolicy
from .types import HttpError, HttpRequest, HttpResponse, RetryError

MetricsHook = Callable[[str, dict], None]


@dataclass(slots=True)
class ReliableHttpClient:
    inner: AsyncHttpClient
    retry: RetryPolicy
    rate_limiter: AsyncRateLimiter | None = None
    concurrency: ConcurrencyLimiter | None = None
    breaker: CircuitBreaker | None = None
    metrics: MetricsHook | None = None

    async def send(self, request: HttpRequest) -> HttpResponse:
        attempt = 0
        last_err: BaseException | None = None
        while True:
            attempt += 1
            if self.metrics:
                self.metrics("attempt", {"attempt": attempt})
            if self.rate_limiter:
                await self.rate_limiter.acquire()
            if self.concurrency:
                async with self.concurrency:
                    response, last_err = await self._try_once(request, attempt)
            else:
                response, last_err = await self._try_once(request, attempt)

            if response is not None:
                return response

            decision = self.retry.decide(attempt, None, last_err)
            if not decision.should_retry:
                if isinstance(last_err, CircuitOpenError):
                    raise last_err
                raise RetryError(str(last_err)) from last_err
            await asyncio.sleep(decision.delay)

    async def _try_once(
        self, request: HttpRequest, attempt: int
    ) -> tuple[HttpResponse | None, BaseException | None]:
        try:
            if self.breaker:
                await self.breaker.before(asyncio.get_running_loop().time())
            resp = await self.inner.send(request)
            if self.breaker:
                if 200 <= resp.status < 400:
                    await self.breaker.record_success()
                else:
                    await self.breaker.record_failure(asyncio.get_running_loop().time())
            # retry on status codes
            if resp.status in self.retry.retry_on_status:
                # Signal retry by surfacing an error; outer loop decides backoff
                return None, HttpError(f"retryable status {resp.status}")
            return resp, None
        except Exception as e:  # noqa: BLE001 - bubble through retry policy
            if self.breaker:
                await self.breaker.record_failure(asyncio.get_running_loop().time())
            return None, e
