from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    pass


@dataclass(slots=True)
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 10.0
    half_open_max_success: int = 2

    # internal state (must be declared for slots=True)
    _state: CircuitState = field(init=False, default=CircuitState.CLOSED)
    _failures: int = field(init=False, default=0)
    _half_open_success: int = field(init=False, default=0)
    _opened_at: float | None = field(init=False, default=None)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    async def before(self, now: float) -> None:
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._opened_at is not None and now - self._opened_at >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_success = 0
                else:
                    raise CircuitOpenError("circuit is open")

    async def record_success(self) -> None:
        async with self._lock:
            self._failures = 0
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_success += 1
                if self._half_open_success >= self.half_open_max_success:
                    self._state = CircuitState.CLOSED

    async def record_failure(self, now: float) -> None:
        async with self._lock:
            self._failures += 1
            if self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                if self._failures >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = now
                    self._half_open_success = 0

    @property
    def state(self) -> CircuitState:
        return self._state
