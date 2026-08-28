"""Opt-in live smoke checks. Missing credentials produce a clean skip."""

from __future__ import annotations

import argparse
import asyncio
import os

from research_radar.agent.hermes_runtime import HermesHttpRuntime
from research_radar.config import AppSettings
from research_radar.llm.gemini import GeminiProvider
from research_radar.schemas import LLMRequest
from research_radar.tools.arxiv import ArxivSearchTool
from research_radar.tools.github import GitHubClient, GitHubSearchTool
from research_radar.tools.web import BraveWebSearchTool


async def smoke_gemini(settings: AppSettings) -> None:
    if settings.gemini_api_key is None:
        print("SKIP gemini: GEMINI_API_KEY is not configured")
        return
    provider = GeminiProvider(
        api_key=settings.gemini_api_key.get_secret_value(),
        default_model=settings.gemini_model,
        timeout_seconds=settings.request_timeout_seconds,
    )
    try:
        response = await provider.generate(
            LLMRequest(prompt="Reply with exactly: radar-ok", temperature=0)
        )
        print(f"PASS gemini: model={response.model} tokens={response.usage.total_tokens}")
    finally:
        await provider.aclose()


async def smoke_hermes(settings: AppSettings) -> None:
    if settings.hermes_api_key is None:
        print("SKIP hermes: HERMES_API_KEY is not configured")
        return
    runtime = HermesHttpRuntime(
        base_url=str(settings.hermes_api_url),
        api_key=settings.hermes_api_key.get_secret_value(),
        model_name=settings.hermes_model_name,
    )
    try:
        health = await runtime.health()
        if not health.healthy:
            raise SystemExit(f"FAIL hermes health: {health.detail}")
        response = await runtime.run("Reply with exactly: hermes-ok", session_key="smoke:test")
        print(f"PASS hermes: model={response.model} latency_ms={response.latency_ms:.0f}")
    finally:
        await runtime.aclose()


async def smoke_tools(settings: AppSettings) -> None:
    github = GitHubClient(
        token=settings.github_token.get_secret_value() if settings.github_token else None,
        base_url=str(settings.github_api_url),
        api_version=settings.github_api_version,
        timeout_seconds=settings.request_timeout_seconds,
    )
    arxiv = ArxivSearchTool(
        base_url=str(settings.arxiv_api_url),
        timeout_seconds=settings.request_timeout_seconds,
        min_interval_seconds=settings.arxiv_min_interval_seconds,
    )
    web = (
        BraveWebSearchTool(
            api_key=settings.brave_search_api_key.get_secret_value(),
            base_url=str(settings.brave_search_api_url),
            timeout_seconds=settings.request_timeout_seconds,
        )
        if settings.brave_search_api_key
        else None
    )
    try:
        repositories = await GitHubSearchTool(github).search("AI agent", limit=1)
        print(f"PASS github: results={len(repositories.items)}")
        papers = await arxiv.search("AI agent", categories=["cs.AI"], limit=1)
        print(f"PASS arxiv: results={len(papers.items)}")
        if web is None:
            print("SKIP web: BRAVE_SEARCH_API_KEY is not configured")
        else:
            web_results = await web.search("AI agent", limit=1)
            print(f"PASS web: results={len(web_results.items)}")
    finally:
        await github.aclose()
        await arxiv.aclose()
        if web is not None:
            await web.aclose()


async def run(target: str) -> None:
    if os.getenv("RUN_LIVE_TESTS") != "1":
        print("SKIP live smoke: set RUN_LIVE_TESTS=1 to contact external services")
        return
    settings = AppSettings()
    if target in {"gemini", "all"}:
        await smoke_gemini(settings)
    if target in {"hermes", "all"}:
        await smoke_hermes(settings)
    if target in {"tools", "all"}:
        await smoke_tools(settings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        choices=["gemini", "hermes", "tools", "all"],
        default="all",
        nargs="?",
    )
    args = parser.parse_args()
    asyncio.run(run(args.target))


if __name__ == "__main__":
    main()
