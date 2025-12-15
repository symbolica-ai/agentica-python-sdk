"""
Exa.ai web search client with rate limiting and ephemeral key management.

Environment variables:
- EXA_API_KEY: Regular API key
- EXA_SERVICE_API_KEY: Service key for creating ephemeral keys
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from agentica_internal.warpc import forbidden
from exa_py import AsyncExa

__all__ = ["SearchResult", "ExaClient", "ExaAdmin", "web_search", "web_fetch", "RateLimitError"]


class RateLimitError(Exception):
    """Exa rate limit exceeded."""

    def __init__(self, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded{f' (retry after {retry_after:.1f}s)' if retry_after else ''}"
        )


@dataclass
class SearchResult:
    """A single search result from Exa."""

    title: str
    url: str
    content_lines: list[str]
    score: float | None = None

    def __repr__(self) -> str:
        return f"SearchResult(title={self.title!r}, url={self.url!r}, score={self.score}, num_lines={self.num_lines})"

    @property
    def num_lines(self) -> int:
        return len(self.content_lines)

    def content_with_line_numbers(self, start: int = 1, end: int | None = None) -> str:
        """
        Return the content of the search result with the lines numbers that have been used.
        Optionally, specify the start and end lines to return (1-indexed, inclusive).
        """
        start = max(start, 1)
        lines = self.content_lines[start - 1 : end]
        return "\n".join(f"{start + idx}: {line}" for idx, line in enumerate(lines))

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content_lines": self.content_lines,
            "score": self.score,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "SearchResult":
        data = json.loads(Path(path).read_text())
        return cls(
            title=data["title"],
            url=data["url"],
            content_lines=data["content_lines"],
            score=data.get("score"),
        )

    @classmethod
    def _from_exa(cls, r: Any) -> "SearchResult":
        return cls(
            title=r.title or "<no title>",
            url=r.url,
            content_lines=getattr(r, "text", "").splitlines(),
            score=getattr(r, "score", None),
        )


class _RateLimiter:
    """N requests per second window."""

    def __init__(self, max_requests: int = 5, window: float = 1.0):
        self.max_requests = max_requests
        self.window = window
        self._count = 0
        self._window_start = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()

            if now - self._window_start >= self.window:
                self._window_start = now
                self._count = 0

            if self._count >= self.max_requests:
                sleep_time = self.window - (now - self._window_start)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self._window_start = time.monotonic()
                self._count = 0

            self._count += 1


_API_KEY_PREFIX = 'agentica_web'


class ExaClient:
    """Rate-limited Exa client with retry on rate limit errors."""

    def __init__(self, api_key: str | None = None, *, max_retries: int = 3, rate_limit: int = 5):
        key = api_key or os.getenv("EXA_API_KEY")
        if not key:
            raise ValueError("No API key. Set EXA_API_KEY or pass api_key=")
        self._api_key: str = key
        self._exa = AsyncExa(api_key=key)
        self._max_retries = max_retries
        self._rate_limiter = _RateLimiter(max_requests=rate_limit, window=1.0)

    async def search(
        self,
        query: str,
        *,
        num_results: int = 5,
        include_text: bool = True,
    ) -> list[SearchResult]:
        for attempt in range(self._max_retries + 1):
            try:
                await self._rate_limiter.acquire()

                if include_text:
                    response = await self._exa.search_and_contents(
                        query=query, num_results=num_results, text=True
                    )
                else:
                    response = await self._exa.search(query=query, num_results=num_results)

                return [SearchResult._from_exa(r) for r in response.results]

            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "429" in err or "limit" in err:
                    if attempt < self._max_retries:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                    raise RateLimitError() from e
                raise

        raise RateLimitError()

    async def fetch(self, url: str) -> SearchResult:
        for attempt in range(self._max_retries + 1):
            try:
                await self._rate_limiter.acquire()
                response = await self._exa.get_contents([url], text=True)
                if not response.results:
                    raise ValueError(f"No content for URL: {url}")
                r = response.results[0]
                return SearchResult._from_exa(r)
            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "429" in err or "limit" in err:
                    if attempt < self._max_retries:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                    raise RateLimitError() from e
                raise

        raise RateLimitError()

    async def close(self) -> None:
        """
        Best-effort cleanup for any underlying HTTP resources held by the Exa client.
        (The upstream SDK may or may not expose an explicit close method.)
        """
        exa = getattr(self, "_exa", None)
        for method_name in ("aclose", "close"):
            try:
                method = getattr(exa, method_name, None)
                if callable(method):
                    res = method()
                    if asyncio.iscoroutine(res):
                        await res
                    break
            except Exception:
                pass


class ExaAdmin:
    """
    Exa Admin API for managing API keys.

    Requires EXA_SERVICE_API_KEY with permissions to create/delete keys.
    See: https://docs.exa.ai/reference/team-management/create-api-key
    """

    BASE_URL = "https://admin-api.exa.ai/team-management/api-keys"

    def __init__(self, service_key: str | None = None):
        key = service_key or os.getenv("EXA_SERVICE_API_KEY")
        if not key:
            raise ValueError("No service key. Set EXA_SERVICE_API_KEY or pass service_key=")
        self._service_key: str = key
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={"x-api-key": self._service_key, "Content-Type": "application/json"},
                timeout=30,
            )
        return self._client

    async def create_key(self, name: str | None = None) -> str:
        name = name or f"{_API_KEY_PREFIX}_{datetime.now():%Y%m%d%H%M%S}"
        resp = await self.client.post("", json={"name": name})
        if resp.status_code >= 400:
            raise ValueError(f"Failed to create key: {resp.status_code} {resp.text}")
        api_key = resp.json().get("apiKey", {}).get("id")
        if not api_key:
            raise ValueError(f"No key in response: {resp.json()}")
        return api_key

    async def list_keys(self) -> list[dict]:
        resp = await self.client.get("")
        if resp.status_code >= 400:
            raise ValueError(f"Failed to list keys: {resp.status_code} {resp.text}")
        return resp.json().get("apiKeys", [])

    async def delete_key(self, key_id: str) -> bool:
        return (await self.client.delete(f"/{key_id}")).is_success

    async def prune_keys(self, prefix: str = _API_KEY_PREFIX) -> int:
        keys = await self.list_keys()
        deleted = 0
        for key in keys:
            if key.get("name", "").startswith(prefix) and (key_id := key.get("id")):
                if await self.delete_key(key_id):
                    deleted += 1
        return deleted

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "ExaAdmin":
        # Ensure the client exists.
        _ = self.client
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


# Convenience functions using a default client

_default_client: ExaClient | None = None


def _get_default_client() -> ExaClient:
    global _default_client
    if _default_client is None:
        _default_client = ExaClient()
    return _default_client


async def web_search(query: str, num_results: int = 5) -> list[SearchResult]:
    """Search the web. Uses EXA_API_KEY."""
    return await _get_default_client().search(query, num_results=num_results)


async def web_fetch(url: str) -> SearchResult:
    """Fetch content from a URL. Uses EXA_API_KEY."""
    return await _get_default_client().fetch(url)


forbidden.whitelist_modules("agentica.std.web.")
forbidden.whitelist_objects(
    SearchResult,
    web_search,
    web_fetch,
)
