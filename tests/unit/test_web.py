from __future__ import annotations

from datetime import date

import httpx
import pytest

from research_radar.tools.web import BraveWebSearchTool, source_quality_score
from research_radar.utils.retry import RetryPolicy


async def test_brave_web_search_normalizes_results_and_authenticates() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-subscription-token"] == "brave-secret"
        assert request.url.params["freshness"] == "2026-08-20to2026-08-27"
        return httpx.Response(
            200,
            json={
                "query": {"more_results_available": False},
                "web": {
                    "results": [
                        {
                            "title": "Official docs",
                            "url": "https://docs.example.com/agents",
                            "description": "Agent documentation",
                            "page_age": "2026-08-26T10:00:00Z",
                            "profile": {"long_name": "Example"},
                            "extra_snippets": ["one", "two", "three", "four"],
                        }
                    ]
                },
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = BraveWebSearchTool(
        api_key="brave-secret",
        client=http,
        retry_policy=RetryPolicy(attempts=1),
    )
    result = await tool.search(
        "agent harness",
        start_date=date(2026, 8, 20),
        end_date=date(2026, 8, 27),
    )

    assert result.items[0].source_name == "Example"
    assert result.items[0].source_quality_score == 0.95
    assert len(result.items[0].extra_snippets) == 3
    await http.aclose()


async def test_web_query_limits_and_source_priority() -> None:
    assert source_quality_score("https://arxiv.org/abs/123") == 0.98
    assert source_quality_score("https://github.com/example/repo") == 0.95
    tool = BraveWebSearchTool(api_key="key")
    with pytest.raises(ValueError, match="must not be empty"):
        await tool.search("")
    await tool.aclose()
