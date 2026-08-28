"""Brave Search API adapter returning compact, normalized web evidence."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Any
from urllib.parse import urlparse

import httpx

from research_radar.exceptions import ExternalServiceError, InvalidResponseError
from research_radar.schemas import SearchBatch, SourceType, WebResult
from research_radar.utils.retry import RetryPolicy, call_with_retry

logger = logging.getLogger(__name__)


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


def _parse_optional_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def source_quality_score(url: str) -> float:
    """Approximate source priority without claiming authority from a snippet."""

    hostname = (urlparse(url).hostname or "").lower().removeprefix("www.")
    path = urlparse(url).path.lower()
    if hostname in {"arxiv.org", "export.arxiv.org"}:
        return 0.98
    if hostname == "github.com":
        return 0.95
    if hostname.startswith("docs.") or hostname.startswith("developer."):
        return 0.95
    if "/docs/" in path or "/documentation/" in path:
        return 0.90
    if hostname.endswith(".edu") or hostname.endswith(".gov"):
        return 0.90
    if "blog" in hostname or "/blog/" in path:
        return 0.75
    return 0.55


def _freshness_value(start: date | None, end: date | None) -> str | None:
    if start is None and end is None:
        return None
    if start is None or end is None:
        raise ValueError("web freshness requires both start_date and end_date")
    if start > end:
        raise ValueError("start_date must not be after end_date")
    return f"{start.isoformat()}to{end.isoformat()}"


class BraveWebSearchTool:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.search.brave.com/res/v1/web/search",
        timeout_seconds: float = 30,
        concurrency: int = 5,
        retry_policy: RetryPolicy | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Brave Search API key must not be empty")
        self.api_key = api_key
        self.base_url = base_url
        self.retry_policy = retry_policy or RetryPolicy()
        self._semaphore = asyncio.Semaphore(concurrency)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        start_date: date | None = None,
        end_date: date | None = None,
        country: str = "US",
        search_language: str = "en",
    ) -> SearchBatch[WebResult]:
        normalized = " ".join(query.split())
        if not normalized:
            raise ValueError("web query must not be empty")
        if len(normalized) > 400 or len(normalized.split()) > 50:
            raise ValueError("web query exceeds Brave's 400-character or 50-word limit")
        if not 1 <= limit <= 20:
            raise ValueError("web result limit must be between 1 and 20")
        params: dict[str, Any] = {
            "q": normalized,
            "count": limit,
            "country": country,
            "search_lang": search_language,
            "safesearch": "moderate",
            "extra_snippets": True,
        }
        freshness = _freshness_value(start_date, end_date)
        if freshness:
            params["freshness"] = freshness

        async def operation() -> httpx.Response:
            async with self._semaphore:
                response = await self._client.get(
                    self.base_url,
                    params=params,
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": self.api_key,
                        "User-Agent": "autonomous-ai-research-radar/0.2",
                    },
                )
            response.raise_for_status()
            return response

        try:
            response = await call_with_retry(
                operation,
                policy=self.retry_policy,
                should_retry=_retryable,
                on_retry=lambda exc, attempt: logger.warning(
                    "Brave Search retry",
                    extra={
                        "event": "tool_retry",
                        "tool": "web",
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                    },
                ),
            )
        except Exception as exc:
            status_code = (
                exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            )
            raise ExternalServiceError(
                "brave_search",
                "Web search sedang tidak dapat diakses. Silakan coba lagi.",
                status_code=status_code,
                retryable=_retryable(exc),
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        try:
            data = response.json()
            raw_results = (data.get("web") or {}).get("results") or []
            results = []
            for item in raw_results:
                profile = item.get("profile") or {}
                results.append(
                    WebResult(
                        title=item["title"],
                        url=item["url"],
                        description=item.get("description") or "",
                        published_at=_parse_optional_datetime(item.get("page_age")),
                        age=item.get("age"),
                        source_name=profile.get("long_name"),
                        source_quality_score=source_quality_score(item["url"]),
                        extra_snippets=(item.get("extra_snippets") or [])[:3],
                    )
                )
        except (ValueError, KeyError, TypeError) as exc:
            raise InvalidResponseError(
                "brave_search",
                "Web search mengembalikan data yang tidak valid.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        query_meta = data.get("query") or {}
        return SearchBatch[WebResult](
            query=query,
            source=SourceType.WEB,
            items=results,
            total_available=None,
            partial=bool(query_meta.get("more_results_available")) and len(results) < limit,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
