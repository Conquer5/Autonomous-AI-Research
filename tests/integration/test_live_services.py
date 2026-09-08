from __future__ import annotations

import os

import pytest

from research_radar.agent.hermes_runtime import HermesHttpRuntime
from research_radar.config import AppSettings
from research_radar.llm.gemini import GeminiProvider
from research_radar.schemas import LLMRequest
from research_radar.tools.arxiv import ArxivSearchTool
from research_radar.tools.github import GitHubClient, GitHubSearchTool
from research_radar.tools.news import RssNewsTool
from research_radar.tools.web import BraveWebSearchTool

SETTINGS = AppSettings()

LIVE_ENABLED = os.getenv("RUN_LIVE_TESTS") == "1"


@pytest.mark.live
@pytest.mark.skipif(
    not LIVE_ENABLED or not SETTINGS.gemini_api_key,
    reason="set RUN_LIVE_TESTS=1 and GEMINI_API_KEY",
)
async def test_live_gemini() -> None:
    provider = GeminiProvider(
        api_key=SETTINGS.gemini_api_key.get_secret_value(),
        default_model=SETTINGS.gemini_model,
    )
    try:
        response = await provider.generate(LLMRequest(prompt="Reply with radar-ok", temperature=0))
        assert response.text
    finally:
        await provider.aclose()


@pytest.mark.live
@pytest.mark.skipif(
    not LIVE_ENABLED or not SETTINGS.hermes_api_key,
    reason="set RUN_LIVE_TESTS=1 and HERMES_API_KEY",
)
async def test_live_hermes_health() -> None:
    runtime = HermesHttpRuntime(
        base_url=str(SETTINGS.hermes_api_url),
        api_key=SETTINGS.hermes_api_key.get_secret_value(),
    )
    try:
        assert (await runtime.health()).healthy
    finally:
        await runtime.aclose()


@pytest.mark.live
@pytest.mark.skipif(not LIVE_ENABLED, reason="set RUN_LIVE_TESTS=1")
async def test_live_github_public_search() -> None:
    github = GitHubClient(
        token=SETTINGS.github_token.get_secret_value() if SETTINGS.github_token else None
    )
    try:
        result = await GitHubSearchTool(github).search("AI agent", limit=1)
        assert result.items
    finally:
        await github.aclose()


@pytest.mark.live
@pytest.mark.skipif(not LIVE_ENABLED, reason="set RUN_LIVE_TESTS=1")
async def test_live_arxiv_search() -> None:
    arxiv = ArxivSearchTool(min_interval_seconds=0)
    try:
        result = await arxiv.search("AI agent", categories=["cs.AI"], limit=1)
        assert result.items
    finally:
        await arxiv.aclose()


@pytest.mark.live
@pytest.mark.skipif(
    not LIVE_ENABLED or not SETTINGS.brave_search_api_key,
    reason="set RUN_LIVE_TESTS=1 and BRAVE_SEARCH_API_KEY",
)
async def test_live_brave_search() -> None:
    web = BraveWebSearchTool(api_key=SETTINGS.brave_search_api_key.get_secret_value())
    try:
        result = await web.search("AI agent", limit=1)
        assert result.items
    finally:
        await web.aclose()


@pytest.mark.live
@pytest.mark.skipif(not LIVE_ENABLED, reason="set RUN_LIVE_TESTS=1")
async def test_live_rss_search() -> None:
    news = RssNewsTool(SETTINGS.news_feed_urls)
    try:
        result = await news.search(("AI", "agent", "model"), limit=3)
        assert result.items
    finally:
        await news.aclose()
