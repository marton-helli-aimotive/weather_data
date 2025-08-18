import asyncio

import pytest

from weather_data.http.circuit_breaker import CircuitBreaker, CircuitOpenError
from weather_data.http.reliability import (
    AsyncRateLimiter,
    ConcurrencyLimiter,
    RetryPolicy,
)
from weather_data.http.reliable_client import ReliableHttpClient
from weather_data.http.types import HttpRequest, RetryError
from weather_data.providers.mock import MockAsyncHttpClient


@pytest.mark.asyncio
async def test_retry_on_500_then_success():
    inner = MockAsyncHttpClient(status_sequence=[500, 200])
    client = ReliableHttpClient(
        inner=inner,
        retry=RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.02),
    )
    resp = await client.send(HttpRequest(method="GET", url="https://x"))
    assert resp.status == 200
    assert inner.calls == 2


@pytest.mark.asyncio
async def test_rate_limiter_allows_burst_then_slows():
    rl = AsyncRateLimiter(rate=5, burst=2)
    t0 = asyncio.get_event_loop().time()
    for _ in range(3):
        await rl.acquire()
    t1 = asyncio.get_event_loop().time()
    # Third acquire should have waited at least ~0.2s at rate 5/sec with burst 2
    assert (t1 - t0) >= 0.15


@pytest.mark.asyncio
async def test_circuit_breaker_opens_and_half_opens():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05, half_open_max_success=1)
    inner = MockAsyncHttpClient(status_sequence=[500, 500, 200])
    client = ReliableHttpClient(inner=inner, retry=RetryPolicy(max_attempts=1), breaker=cb)

    # first failure
    with pytest.raises(RetryError):
        await client.send(HttpRequest(method="GET", url="https://x"))
    # second failure triggers open
    with pytest.raises(RetryError):
        await client.send(HttpRequest(method="GET", url="https://x"))
    assert cb.state.name == "OPEN"

    # calls while open should raise CircuitOpenError via before()
    with pytest.raises(CircuitOpenError):
        await client.send(HttpRequest(method="GET", url="https://x"))

    await asyncio.sleep(0.06)
    # next attempt in half-open should succeed and close
    inner.status_sequence.append(200)
    resp = await client.send(HttpRequest(method="GET", url="https://x"))
    assert resp.status == 200


@pytest.mark.asyncio
async def test_concurrency_limiter():
    sem = ConcurrencyLimiter(2)
    order = []

    async def worker(i: int):
        async with sem:
            order.append((i, "start"))
            await asyncio.sleep(0.05)
            order.append((i, "end"))

    await asyncio.gather(*(worker(i) for i in range(4)))
    starts = [t for t in order if t[1] == "start"]
    # Ensure first two started before others
    assert starts[0][0] in (0, 1)
    assert starts[1][0] in (0, 1)
