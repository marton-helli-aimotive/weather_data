from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class HttpRequest:
    method: str
    url: str
    headers: MutableMapping[str, str] = field(default_factory=dict)
    params: MutableMapping[str, Any] = field(default_factory=dict)
    body: Any | None = None
    timeout: float | None = None


@dataclass(slots=True)
class HttpResponse:
    status: int
    url: str
    headers: Mapping[str, str]
    body: bytes
    elapsed: float

    def json(self) -> Any:
        import json

        return json.loads(self.body.decode("utf-8"))


class HttpError(Exception):
    pass


class TimeoutError(HttpError):
    pass


class RetryError(HttpError):
    pass
