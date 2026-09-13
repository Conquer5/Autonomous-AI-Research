"""Async arXiv Atom API search with category/date filtering."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime

import httpx
from pydantic import HttpUrl

from research_radar.exceptions import InvalidResponseError
from research_radar.retrieval import (
    RetrievalError,
    RetrievalFailure,
    observed_search,
    request_with_retry,
    retry_after,
)
from research_radar.schemas import PaperResult, SearchBatch, SourceType
from research_radar.utils.retry import RetryPolicy

logger = logging.getLogger(__name__)
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
SUPPORTED_CATEGORIES = {"cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.SE", "cs.IR"}


def _clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def _parse_time(value: str | None) -> datetime:
    if not value:
        raise ValueError("timestamp missing")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class ArxivSearchTool:
    def __init__(
        self,
        *,
        base_url: str = "https://export.arxiv.org/api/query",
        timeout_seconds: float = 60,
        min_interval_seconds: float = 3,
        rate_limit_cooldown_seconds: float = 60,
        retry_policy: RetryPolicy | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url
        self.min_interval_seconds = min_interval_seconds
        self.rate_limit_cooldown_seconds = rate_limit_cooldown_seconds
        self.retry_policy = retry_policy or RetryPolicy()
        self.timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
        self._rate_lock = asyncio.Lock()
        self._search_lock = asyncio.Lock()
        self._last_request_started = 0.0
        self._blocked_until = 0.0
        self._last_rate_failure: RetrievalError | None = None

    async def _respect_rate_limit(self) -> None:
        async with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_started
            delay = max(
                self.min_interval_seconds - elapsed,
                self._blocked_until - time.monotonic(),
            )
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_request_started = time.monotonic()

    @staticmethod
    def _query(query: str, categories: list[str] | None, published_after: date | None) -> str:
        escaped = query.strip().replace('"', "")
        terms = re.findall(r"[\w.+-]+", escaped)[:12]
        if not terms:
            raise ValueError("arXiv query has no searchable terms")
        pieces = ["(" + " AND ".join(f'all:"{term}"' for term in terms) + ")"]
        if categories:
            invalid = set(categories) - SUPPORTED_CATEGORIES
            if invalid:
                raise ValueError(f"unsupported arXiv categories: {sorted(invalid)}")
            category_query = " OR ".join(f"cat:{category}" for category in categories)
            pieces.append(f"({category_query})")
        if published_after:
            start = published_after.strftime("%Y%m%d0000")
            end = datetime.now(UTC).strftime("%Y%m%d2359")
            pieces.append(f"submittedDate:[{start} TO {end}]")
        return " AND ".join(pieces)

    @observed_search("arxiv")
    async def search(
        self,
        query: str,
        *,
        categories: list[str] | None = None,
        published_after: date | None = None,
        limit: int = 10,
    ) -> SearchBatch[PaperResult]:
        if not query.strip():
            raise ValueError("arXiv query must not be empty")
        if not 1 <= limit <= 100:
            raise ValueError("arXiv result limit must be between 1 and 100")
        search_query = self._query(query, categories, published_after)

        async def operation() -> httpx.Response:
            response = await self._client.get(
                self.base_url,
                params={
                    "search_query": search_query,
                    "start": 0,
                    "max_results": limit,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
                headers={"User-Agent": "autonomous-ai-research-radar/0.2"},
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if response.status_code == 429:
                    self._blocked_until = time.monotonic() + max(
                        self.rate_limit_cooldown_seconds, retry_after(exc)
                    )
                raise
            return response

        # Hold one connection across retries. Other queries must not keep hitting
        # an already rate-limited provider after the retry budget is exhausted.
        async with self._search_lock:
            if self._last_rate_failure and time.monotonic() < self._blocked_until:
                raise self._last_rate_failure
            self._last_rate_failure = None
            try:
                response = await request_with_retry(
                    "arxiv",
                    operation,
                    policy=self.retry_policy,
                    timeout_seconds=self.timeout_seconds,
                    before_attempt=self._respect_rate_limit,
                )
            except RetrievalError as exc:
                if exc.failure.failure_category == "http_429":
                    self._last_rate_failure = exc
                raise
        if response.status_code == 204:
            return SearchBatch[PaperResult](query=query, source=SourceType.ARXIV, items=[])
        return self._parse_feed(response.content, query, published_after)

    @staticmethod
    def _parse_feed(
        payload: bytes, query: str, published_after: date | None
    ) -> SearchBatch[PaperResult]:
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise InvalidResponseError(
                "arxiv", "arXiv mengembalikan XML yang tidak valid.", detail=str(exc)
            ) from exc

        if root.tag != f"{ATOM}feed":
            raise TypeError("unexpected Atom root")
        if any(
            "api/errors" in (entry.findtext(f"{ATOM}id") or "")
            for entry in root.findall(f"{ATOM}entry")
        ):
            raise ValueError("arXiv rejected query")
        total_element = root.find(f"{OPENSEARCH}totalResults")
        try:
            total = (
                int(total_element.text)
                if total_element is not None and total_element.text
                else None
            )
        except ValueError as exc:
            raise TypeError("invalid Atom totalResults") from exc
        papers: list[PaperResult] = []
        warnings: list[str] = []
        for entry in root.findall(f"{ATOM}entry"):
            try:
                raw_id = _clean_text(entry.findtext(f"{ATOM}id"))
                arxiv_id = raw_id.rstrip("/").rsplit("/", 1)[-1]
                categories = [
                    category.attrib["term"]
                    for category in entry.findall(f"{ATOM}category")
                    if category.attrib.get("term")
                ]
                primary = entry.find(f"{ARXIV}primary_category")
                pdf_url = next(
                    (
                        link.attrib.get("href")
                        for link in entry.findall(f"{ATOM}link")
                        if link.attrib.get("type") == "application/pdf"
                    ),
                    None,
                )
                published = _parse_time(entry.findtext(f"{ATOM}published"))
                if published_after and published.date() < published_after:
                    continue
                if not _clean_text(entry.findtext(f"{ATOM}title")):
                    raise ValueError("missing title")
                papers.append(
                    PaperResult(
                        arxiv_id=arxiv_id,
                        title=_clean_text(entry.findtext(f"{ATOM}title")),
                        authors=[
                            _clean_text(author.findtext(f"{ATOM}name"))
                            for author in entry.findall(f"{ATOM}author")
                        ],
                        abstract=_clean_text(entry.findtext(f"{ATOM}summary")),
                        categories=categories,
                        primary_category=primary.attrib.get("term")
                        if primary is not None
                        else None,
                        published_at=published,
                        updated_at=_parse_time(entry.findtext(f"{ATOM}updated")),
                        url=HttpUrl(raw_id.replace("http://", "https://", 1)),
                        pdf_url=HttpUrl(pdf_url.replace("http://", "https://", 1))
                        if pdf_url
                        else None,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                warnings.append(
                    f"Satu paper dilewati karena metadata tidak valid: {type(exc).__name__}"
                )
        return SearchBatch[PaperResult](
            query=query,
            source=SourceType.ARXIV,
            items=list({str(item.url): item for item in papers}.values()),
            failures=[
                RetrievalFailure(
                    provider="arxiv", failure_category="schema_drift", error_code="INVALID_ENTRY"
                )
                for _ in warnings
            ],
            total_available=total,
            partial=bool(warnings),
            warnings=warnings,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
