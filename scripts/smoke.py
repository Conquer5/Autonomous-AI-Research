"""Opt-in live smoke checks. Missing credentials produce a clean skip."""

from __future__ import annotations

import argparse
import asyncio
import os

from research_radar.agent.hermes_runtime import HermesHttpRuntime
from research_radar.config import AppSettings
from research_radar.llm.gemini import GeminiProvider
from research_radar.retrieval import classify_failure
from research_radar.schemas import LLMRequest
from research_radar.tools.arxiv import ArxivSearchTool
from research_radar.tools.github import GitHubClient, GitHubSearchTool
from research_radar.tools.news import RssNewsTool
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
            raise RuntimeError("Hermes health failed")
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
    news = RssNewsTool(settings.news_feed_urls, timeout_seconds=settings.request_timeout_seconds)
    failed = []
    try:
        checks = [
            ("github", lambda: GitHubSearchTool(github).search("AI agent", limit=1)),
            ("arxiv", lambda: arxiv.search("AI agent", categories=["cs.AI"], limit=1)),
            ("news", lambda: news.search(("AI", "agent", "model"), limit=3)),
        ]
        if web is not None:
            checks.append(("web", lambda: web.search("AI agent", limit=1)))
        else:
            print("SKIP web: BRAVE_SEARCH_API_KEY is not configured")
        for name, check in checks:
            try:
                async with asyncio.timeout(60):
                    batch = await check()
                if not batch.items:
                    failed.append(name)
                    print(f"FAIL {name}: empty result")
                else:
                    print(f"PASS {name}: results={len(batch.items)} partial={batch.partial}")
                for failure in batch.failures:
                    print(f"DETAIL {name}: {failure.failure_category}")
            except Exception as exc:
                failed.append(name)
                print(f"FAIL {name}: {classify_failure(name, exc).failure_category}")
    finally:
        await github.aclose()
        await arxiv.aclose()
        await news.aclose()
        if web is not None:
            await web.aclose()
    if failed:
        raise RuntimeError("Some source checks failed")


async def run(target: str) -> None:
    if os.getenv("RUN_LIVE_TESTS") != "1":
        print("SKIP live smoke: set RUN_LIVE_TESTS=1 to contact external services")
        return
    settings = AppSettings()
    failed = []
    for name, check in [("gemini", smoke_gemini), ("hermes", smoke_hermes), ("tools", smoke_tools)]:
        if target not in {name, "all"}:
            continue
        try:
            async with asyncio.timeout(180):
                await check(settings)
        except (Exception, SystemExit) as exc:
            failed.append(name)
            print(f"FAIL {name}: {type(exc).__name__}")
    if failed:
        raise SystemExit(1)


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
