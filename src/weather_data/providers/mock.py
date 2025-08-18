from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from ..http.interfaces import AsyncHttpClient
from ..http.types import HttpRequest, HttpResponse
from .base import WeatherProvider


class MockAsyncHttpClient(AsyncHttpClient):
    def __init__(self, *, status_sequence: list[int] | None = None, delay: float = 0.0) -> None:
        self.status_sequence = status_sequence or [200]
        self.delay = delay
        self.calls = 0

    async def send(self, request: HttpRequest) -> HttpResponse:
        await asyncio.sleep(self.delay)
        status = self.status_sequence[min(self.calls, len(self.status_sequence) - 1)]
        self.calls += 1
        return HttpResponse(
            status=status,
            url=request.url,
            headers={},
            body=b"{}",
            elapsed=self.delay,
        )


class MockProvider(WeatherProvider):
    async def build_request(self, **kwargs: Any) -> HttpRequest:
        return HttpRequest(method="GET", url="https://example.test/mock")

    async def parse_response(self, response: HttpResponse) -> Mapping[str, Any]:
        return {"status": response.status}
