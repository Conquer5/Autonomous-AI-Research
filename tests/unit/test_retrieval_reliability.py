"""Failure contracts exercised through the real HTTP adapters."""

from __future__ import annotations

import asyncio
import logging
import socket
from contextlib import contextmanager
from unittest.mock import AsyncMock

import httpx
import pytest

from research_radar.observability.logging import JsonFormatter, current_run_id
from research_radar.retrieval import (
    RetrievalError,
    classify_failure,
    operation_sink,
    request_with_retry,
)
from research_radar.tools.arxiv import ArxivSearchTool
from research_radar.tools.github import GitHubClient, GitHubSearchTool
from research_radar.tools.news import RssNewsTool
from research_radar.tools.web import BraveWebSearchTool
from research_radar.utils.retry import RetryPolicy

RETRY = RetryPolicy(attempts=3, base_delay_seconds=0, jitter_seconds=0)
RSS = (
    b"<rss><channel><item><title>AI agent</title>"
    b"<link>https://example.com/Agent?utm_source=x</link>"
    b"<description>AI agent research</description></item></channel></rss>"
)
ATOM = (
    b'<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
    b"<id>https://arxiv.org/abs/2608.12345v1</id><title>AI agent</title>"
    b"<summary>AI agent research</summary><published>2026-08-01T00:00:00Z</published>"
    b"<updated>2026-08-01T00:00:00Z</updated></entry></feed>"
)
REPO = {
    "full_name": "org/agent",
    "html_url": "https://github.com/org/agent",
    "created_at": "2026-08-01T00:00:00Z",
    "updated_at": "2026-08-01T00:00:00Z",
}
WEB = {"title": "AI agent", "url": "https://example.com/agent"}


def adapter(name, http):
    if name == "github":
        return GitHubSearchTool(GitHubClient(client=http, retry_policy=RETRY)).search
    if name == "arxiv":
        return ArxivSearchTool(client=http, min_interval_seconds=0, retry_policy=RETRY).search
    if name == "web":
        return BraveWebSearchTool(api_key="private-key", client=http, retry_policy=RETRY).search
    tool = RssNewsTool(("https://example.com/feed",), client=http, retry_policy=RETRY)
    return lambda query: tool.search((query,))


def success(name):
    if name == "github":
        return httpx.Response(200, json={"items": [REPO], "total_count": 1})
    if name == "web":
        return httpx.Response(200, json={"web": {"results": [WEB]}})
    return httpx.Response(200, content=RSS if name == "news" else ATOM)


@pytest.mark.parametrize("provider", ["github", "arxiv", "news", "web"])
@pytest.mark.parametrize(
    "status,category,attempts",
    [
        (400, "invalid_query", 1),
        (401, "authentication_failure", 1),
        (403, "authorization_failure", 1),
        (404, "http_4xx", 1),
        (429, "http_429", 3),
        (500, "http_5xx", 3),
        (502, "http_5xx", 3),
        (503, "http_5xx", 3),
    ],
)
async def test_http_failure_matrix(provider, status, category, attempts, caplog):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"message": "sensitive-provider-payload"})

    traces = []
    token = operation_sink.set(traces)
    try:
        with caplog.at_level(logging.INFO), current_run("test-run"):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                search = adapter(provider, http)
                if provider == "news":
                    batch = await search("agent")
                    assert batch.partial and not batch.items
                    failure = batch.failures[0]
                else:
                    with pytest.raises(RetrievalError) as caught:
                        await search("agent")
                    failure = caught.value.failure
        assert calls == attempts
        assert failure.failure_category == category
        assert failure.http_status_class == f"{status // 100}xx"
        assert traces[0].attempts == attempts
        assert traces[0].retries == attempts - 1
        assert traces[0].run_id == "test-run"
        assert traces[0].latency_ms >= 0
        rendered = " ".join(JsonFormatter().format(r) for r in caplog.records)
        assert "sensitive-provider-payload" not in rendered
        assert "private-key" not in rendered
        assert "provider_request" in rendered
    finally:
        operation_sink.reset(token)


@contextmanager
def current_run(run_id):
    token = current_run_id.set(run_id)
    try:
        yield
    finally:
        current_run_id.reset(token)


@pytest.mark.parametrize("provider", ["github", "arxiv", "news", "web"])
@pytest.mark.parametrize("status", [200, 204])
async def test_success_and_no_content(provider, status):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: success(provider) if status == 200 else httpx.Response(204)
        )
    ) as http:
        batch = await adapter(provider, http)("agent")
    assert len(batch.items) == (1 if status == 200 else 0)
    assert not batch.partial


@pytest.mark.parametrize(
    "kind,category",
    [("timeout", "timeout"), ("connect", "connection_failure"), ("dns", "dns_failure")],
)
@pytest.mark.parametrize("provider", ["github", "arxiv", "news", "web"])
async def test_network_failures(provider, kind, category):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if kind == "timeout":
            raise httpx.ReadTimeout("private-payload", request=request)
        if kind == "dns":
            raise httpx.ConnectError("private-payload", request=request) from socket.gaierror(
                -2, "dns"
            )
        raise httpx.ConnectError("private-payload", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        if provider == "news":
            batch = await adapter(provider, http)("agent")
            failure = batch.failures[0]
        else:
            with pytest.raises(RetrievalError) as caught:
                await adapter(provider, http)("agent")
            failure = caught.value.failure
    assert failure.failure_category == category
    assert calls == 3


@pytest.mark.parametrize("provider", ["github", "arxiv", "news", "web"])
async def test_malformed_response_not_retried(provider):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"{broken <")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        if provider == "news":
            batch = await adapter(provider, http)("agent")
            failure = batch.failures[0]
        else:
            with pytest.raises(RetrievalError) as caught:
                await adapter(provider, http)("agent")
            failure = caught.value.failure
    assert failure.failure_category in {"malformed_response", "parser_failure"}
    assert calls == 1


@pytest.mark.parametrize("provider", ["github", "arxiv", "news", "web"])
async def test_schema_drift_and_duplicate_entries_keep_valid_evidence(provider):
    if provider == "github":
        response = httpx.Response(200, json={"items": [{}, REPO, REPO]})
    elif provider == "web":
        response = httpx.Response(200, json={"web": {"results": [{}, WEB, WEB]}})
    elif provider == "arxiv":
        response = httpx.Response(
            200,
            content=ATOM.replace(
                b"</feed>",
                b"<entry/>"
                + ATOM.split(b"<entry>")[1].split(b"</feed>")[0].join([b"<entry>", b""])
                + b"</feed>",
            ),
        )
    else:
        response = httpx.Response(
            200,
            content=RSS.replace(
                b"</channel>",
                b"<item/>"
                + RSS.split(b"<item>")[1].split(b"</channel>")[0].join([b"<item>", b""])
                + b"</channel>",
            ),
        )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as http:
        batch = await adapter(provider, http)("agent")
    assert batch.partial
    assert len(batch.items) == 1
    assert batch.failures[0].failure_category == "schema_drift"


@pytest.mark.parametrize(
    "provider,payload", [("github", []), ("github", {}), ("web", []), ("web", {"web": []})]
)
async def test_changed_envelope_is_not_empty_success(provider, payload):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as http:
        with pytest.raises(RetrievalError) as caught:
            await adapter(provider, http)("agent")
    assert caught.value.failure.failure_category == "schema_drift"


async def test_feed_isolation_and_canonical_case_sensitive_dedup():
    def handler(request):
        if request.url.path == "/bad":
            return httpx.Response(503)
        return httpx.Response(200, content=RSS)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        tool = RssNewsTool(
            ("https://example.com/bad", "https://example.com/good"), client=http, retry_policy=RETRY
        )
        batch = await tool.search(("agent",))
    assert len(batch.items) == 1
    assert batch.partial and batch.failures[0].failure_category == "http_5xx"


async def test_retry_after_long_delay_does_not_retry_early():
    response = httpx.Response(
        403,
        request=httpx.Request("GET", "https://api.github.com"),
        headers={"x-ratelimit-remaining": "0", "retry-after": "3600"},
    )
    operation = AsyncMock(
        side_effect=httpx.HTTPStatusError("hidden", request=response.request, response=response)
    )
    with pytest.raises(RetrievalError) as caught:
        await request_with_retry("github", operation, policy=RETRY)
    assert operation.await_count == 1
    assert caught.value.failure.failure_category == "rate_limiting"
    assert caught.value.retryable


async def test_retry_recovery_is_observed():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(502) if calls == 1 else success("github")

    traces = []
    token = operation_sink.set(traces)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            batch = await adapter("github", http)("agent")
    finally:
        operation_sink.reset(token)
    assert len(batch.items) == 1 and not batch.partial
    assert traces[0].retries == 1 and traces[0].attempts == 2


async def test_per_request_deadline_and_cancellation():
    async def slow():
        await asyncio.sleep(10)

    with pytest.raises(RetrievalError) as caught:
        await request_with_retry(
            "github", slow, policy=RetryPolicy(attempts=1), timeout_seconds=0.01
        )
    assert caught.value.failure.failure_category == "timeout"
    task = asyncio.create_task(request_with_retry("github", slow, policy=RETRY))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_quota_is_permanent_and_query_is_provider_specific():
    response = httpx.Response(
        429,
        json={"error": {"code": "QUOTA_LIMITED"}},
        request=httpx.Request("GET", "https://example.com"),
    )
    failure = classify_failure(
        "web", httpx.HTTPStatusError("hidden", request=response.request, response=response)
    )
    assert failure.failure_category == "quota_exhaustion" and not failure.retryable
    query = ArxivSearchTool._query("local AI agent benchmarks", ["cs.AI"], None)
    assert 'all:"local" AND all:"AI"' in query
    assert 'all:"local AI agent benchmarks"' not in query
    assert "cat:cs.AI" in query


async def test_slow_feed_preserves_other_feed_within_query_deadline():
    import time

    from research_radar.retrieval import operation_deadline

    async def handler(request):
        if request.url.path == "/slow":
            await asyncio.sleep(10)
        return httpx.Response(200, content=RSS)

    token = operation_deadline.set(time.perf_counter() + 0.1)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            tool = RssNewsTool(
                ("https://example.com/slow", "https://example.com/good"), client=http
            )
            async with asyncio.timeout(0.15):
                batch = await tool.search(("agent",))
        assert len(batch.items) == 1 and batch.partial
        assert batch.failures[0].failure_category == "timeout"
        assert batch.failures[0].operation.startswith("feed:")
    finally:
        operation_deadline.reset(token)


def test_structured_failures_are_json_objects_and_secrets_redacted():
    import json

    record = logging.LogRecord("test", logging.INFO, "", 0, "diagnostics", (), None)
    record.failures = [
        {
            "failure_category": "timeout",
            "api_key": "hidden",
            "nested": {"description": "token=hidden"},
        }
    ]
    payload = json.loads(JsonFormatter().format(record))
    assert payload["failures"][0]["failure_category"] == "timeout"
    assert "hidden" not in json.dumps(payload)


async def test_decorator_preserves_keyword_query_and_topics_contracts():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: success("github"))
    ) as http:
        batch = await GitHubSearchTool(GitHubClient(client=http)).search(query="agent")
        assert batch.items
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: success("news"))
    ) as http:
        batch = await RssNewsTool(("https://example.com/feed",), client=http).search(
            topics=("agent",)
        )
        assert batch.items
