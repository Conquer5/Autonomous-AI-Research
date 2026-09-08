"""Small, dependency-free RSS/Atom reader for free first-party technology news."""

from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx
from pydantic import HttpUrl

from research_radar.evidence.canonicalizer import canonicalize_url
from research_radar.retrieval import (
    RetrievalFailure,
    classify_failure,
    observed_search,
    operation_deadline,
    request_with_retry,
)
from research_radar.schemas import NewsResult, SearchBatch, SourceType
from research_radar.utils.retry import RetryPolicy

logger = logging.getLogger(__name__)
_WORD_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_IGNORED_TOPIC_WORDS = {
    "and",
    "for",
    "the",
    "with",
    "tools",
    "models",
    "frameworks",
    "only",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _clean_markup(value: str | None) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html.unescape(value or ""))
    except (ValueError, TypeError):
        return ""
    return " ".join(" ".join(parser.parts).split())


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child(element: ET.Element, *names: str) -> ET.Element | None:
    expected = set(names)
    return next((item for item in element if _local_name(item.tag) in expected), None)


def _child_text(element: ET.Element, *names: str) -> str:
    item = _child(element, *names)
    return "" if item is None else "".join(item.itertext()).strip()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _topic_tokens(topics: tuple[str, ...]) -> set[str]:
    return {
        token
        for topic in topics
        for token in _WORD_PATTERN.findall(topic.lower())
        if token not in _IGNORED_TOPIC_WORDS
    }


def _relevance(title: str, description: str, tokens: set[str]) -> float:
    if not tokens:
        return 0.5
    title_tokens = set(_WORD_PATTERN.findall(title.lower()))
    body_tokens = set(_WORD_PATTERN.findall(description.lower()))
    title_hits = len(tokens & title_tokens)
    body_hits = len(tokens & body_tokens)
    return min(1.0, 0.15 + title_hits * 0.18 + body_hits * 0.06)


def news_rank_key(item: NewsResult) -> tuple[datetime, float]:
    # Fresh announcements must not lose to older keyword-heavy articles.
    return (item.published_at or datetime.min.replace(tzinfo=UTC), item.relevance_score)


def diverse_news(items: list[NewsResult]) -> list[NewsResult]:
    """Interleave publishers, keeping the freshest articles first within each."""
    publishers: dict[str, list[NewsResult]] = {}
    for item in sorted(items, key=news_rank_key, reverse=True):
        publisher = (urlsplit(str(item.url)).hostname or "").removeprefix("www.")
        if publisher == "news.google.com":
            publisher = (item.source_name or publisher).lower()
        publishers.setdefault(publisher, []).append(item)
    ordered: list[NewsResult] = []
    while publishers:
        for publisher in list(publishers):
            ordered.append(publishers[publisher].pop(0))
            if not publishers[publisher]:
                del publishers[publisher]
    return ordered


def _entry_link(entry: ET.Element, feed_url: str) -> str:
    for item in entry:
        if _local_name(item.tag) != "link":
            continue
        href = item.attrib.get("href")
        if href and item.attrib.get("rel", "alternate") == "alternate":
            return urljoin(feed_url, href)
        if item.text and item.text.strip():
            return urljoin(feed_url, item.text.strip())
    guid = _child_text(entry, "guid", "id")
    return urljoin(feed_url, guid) if guid.startswith(("http://", "https://", "/")) else ""


class RssNewsTool:
    """Fetch configured feeds concurrently and normalize RSS/Atom entries."""

    def __init__(
        self,
        feed_urls: tuple[str, ...],
        *,
        timeout_seconds: float = 30,
        concurrency: int = 3,
        client: httpx.AsyncClient | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.feed_urls = tuple(dict.fromkeys(feed_urls))
        self.retry_policy = retry_policy or RetryPolicy()
        self.timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(concurrency)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))

    async def _fetch(self, url: str) -> bytes:
        async def operation() -> httpx.Response:
            async with self._semaphore:
                response = await self._client.get(
                    url,
                    headers={"User-Agent": "autonomous-ai-research-radar/0.3"},
                    follow_redirects=True,
                )
            response.raise_for_status()
            return response

        deadline = operation_deadline.get()
        remaining = (
            self.timeout_seconds
            if deadline is None
            else min(self.timeout_seconds, deadline - time.perf_counter())
        )
        # Finish each feed before the enclosing search deadline so healthy feeds survive.
        async with asyncio.timeout(max(0, remaining * 0.9)):
            response = await request_with_retry(
                "news",
                operation,
                policy=self.retry_policy,
                timeout_seconds=self.timeout_seconds,
                resource_id=hashlib.sha256(url.encode()).hexdigest()[:16],
            )
        return response.content if response.status_code != 204 else b"<rss><channel/></rss>"

    @staticmethod
    def _parse_feed(
        payload: bytes,
        feed_url: str,
        *,
        tokens: set[str],
        published_after: date | None,
        failures: list[RetrievalFailure] | None = None,
    ) -> list[NewsResult]:
        root = ET.fromstring(payload)
        if _local_name(root.tag) not in {"rss", "feed", "rdf"}:
            raise TypeError("unexpected feed root")
        parsed_channel = _child(root, "channel")
        channel = parsed_channel if parsed_channel is not None else root
        source_name = _clean_markup(_child_text(channel, "title")) or feed_url
        entries = [item for item in channel.iter() if _local_name(item.tag) in {"item", "entry"}]
        results: list[NewsResult] = []
        for entry in entries:
            title = _clean_markup(_child_text(entry, "title"))
            description = _clean_markup(_child_text(entry, "summary", "description", "content"))
            link = _entry_link(entry, feed_url)
            published_at = _parse_datetime(
                _child_text(entry, "published", "pubdate", "updated", "date")
            )
            relevance = _relevance(title, description, tokens)
            try:
                if not title or not link:
                    raise ValueError("missing entry field")
                valid_url = HttpUrl(link)
            except ValueError:
                if failures is not None:
                    failures.append(
                        RetrievalFailure(
                            provider="news",
                            failure_category="schema_drift",
                            error_code="INVALID_ENTRY",
                        )
                    )
                continue
            if tokens and relevance <= 0.15:
                continue
            if published_after and published_at and published_at.date() < published_after:
                continue
            results.append(
                NewsResult(
                    title=title,
                    url=valid_url,
                    description=description,
                    source_name=_clean_markup(_child_text(entry, "source")) or source_name,
                    published_at=published_at,
                    relevance_score=relevance,
                )
            )
        return results

    @observed_search("news")
    async def search(
        self,
        topics: tuple[str, ...],
        *,
        published_after: date | None = None,
        limit: int = 10,
    ) -> SearchBatch[NewsResult]:
        if not 1 <= limit <= 50:
            raise ValueError("news result limit must be between 1 and 50")
        if not self.feed_urls:
            from research_radar.exceptions import ConfigurationError

            raise ConfigurationError("No feeds configured")
        tokens = _topic_tokens(topics)
        responses = await asyncio.gather(
            *(self._fetch(url) for url in self.feed_urls), return_exceptions=True
        )
        warnings: list[str] = []
        failures: list[RetrievalFailure] = []
        results: list[NewsResult] = []
        for url, response in zip(self.feed_urls, responses, strict=True):
            if isinstance(response, BaseException):
                failure = classify_failure("news", response)
                failure.operation = "feed:" + hashlib.sha256(url.encode()).hexdigest()[:16]
                failures.append(failure)
                warnings.append(f"Feed failed: {failure.failure_category}")
                logger.warning(
                    "RSS feed failed",
                    extra={"event": "rss_feed_failed", "error_type": type(response).__name__},
                )
                continue
            try:
                results.extend(
                    self._parse_feed(
                        response,
                        url,
                        tokens=tokens,
                        published_after=published_after,
                        failures=failures,
                    )
                )
            except (ET.ParseError, TypeError, ValueError) as exc:
                failure = classify_failure("news", exc)
                failure.operation = "feed:" + hashlib.sha256(url.encode()).hexdigest()[:16]
                failures.append(failure)
                warnings.append("Invalid feed skipped")

        unique: dict[str, NewsResult] = {}
        for item in results:
            key = canonicalize_url(str(item.url))
            previous = unique.get(key)
            if previous is None or item.relevance_score > previous.relevance_score:
                unique[key] = item
        ordered = diverse_news(list(unique.values()))
        return SearchBatch[NewsResult](
            query=", ".join(topics),
            source=SourceType.NEWS,
            items=ordered[:limit],
            total_available=len(ordered),
            partial=bool(warnings or failures),
            failures=failures,
            warnings=warnings,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
