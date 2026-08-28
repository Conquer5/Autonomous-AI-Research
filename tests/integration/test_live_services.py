from __future__ import annotations

import os

import pytest

from research_radar.agent.hermes_runtime import HermesHttpRuntime
from research_radar.llm.gemini import GeminiProvider
from research_radar.schemas import LLMRequest
from research_radar.tools.arxiv import ArxivSearchTool
from research_radar.tools.github import GitHubClient, GitHubSearchTool
from research_radar.tools.web import BraveWebSearchTool

LIVE_ENABLED = os.getenv("RUN_LIVE_TESTS") == "1"


@pytest.mark.live
@pytest.mark.skipif(
    not LIVE_ENABLED or not os.getenv("GEMINI_API_KEY"),
    reason="set RUN_LIVE_TESTS=1 and GEMINI_API_KEY",
)
async def test_live_gemini() -> None:
    provider = GeminiProvider(
        api_key=os.environ["GEMINI_API_KEY"],
        default_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
    )
    try:
        response = await provider.generate(LLMRequest(prompt="Reply with radar-ok", temperature=0))
        assert response.text
    finally:
        await provider.aclose()


@pytest.mark.live
@pytest.mark.skipif(
    not LIVE_ENABLED or not os.getenv("HERMES_API_KEY"),
    reason="set RUN_LIVE_TESTS=1 and HERMES_API_KEY",
)
async def test_live_hermes_health() -> None:
    runtime = HermesHttpRuntime(
        base_url=os.getenv("HERMES_API_URL", "http://127.0.0.1:8642/v1"),
        api_key=os.environ["HERMES_API_KEY"],
    )
    try:
        assert (await runtime.health()).healthy
    finally:
        await runtime.aclose()


@pytest.mark.live
@pytest.mark.skipif(not LIVE_ENABLED, reason="set RUN_LIVE_TESTS=1")
async def test_live_github_public_search() -> None:
    github = GitHubClient(token=os.getenv("GITHUB_TOKEN"))
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
    not LIVE_ENABLED or not os.getenv("BRAVE_SEARCH_API_KEY"),
    reason="set RUN_LIVE_TESTS=1 and BRAVE_SEARCH_API_KEY",
)
async def test_live_brave_search() -> None:
    web = BraveWebSearchTool(api_key=os.environ["BRAVE_SEARCH_API_KEY"])
    try:
        result = await web.search("AI agent", limit=1)
        assert result.items
    finally:
        await web.aclose()
