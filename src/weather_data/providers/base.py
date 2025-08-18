from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import Any

from ..http.types import HttpRequest, HttpResponse


class WeatherProvider(abc.ABC):
    """Unified provider interface for M02. Data models arrive in M04.

    Implementations should translate domain requests to HttpRequest and back.
    """

    @abc.abstractmethod
    async def build_request(self, **kwargs: Any) -> HttpRequest: ...

    @abc.abstractmethod
    async def parse_response(self, response: HttpResponse) -> Mapping[str, Any]: ...
