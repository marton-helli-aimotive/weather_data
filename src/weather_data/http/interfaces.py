from __future__ import annotations

from typing import Protocol

from .types import HttpRequest, HttpResponse


class AsyncHttpClient(Protocol):
    async def send(self, request: HttpRequest) -> HttpResponse:  # pragma: no cover - interface
        ...
