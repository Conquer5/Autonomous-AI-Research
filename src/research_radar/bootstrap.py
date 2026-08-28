"""Dependency assembly kept separate from handlers and business logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_radar.agent.hermes_runtime import HermesHttpRuntime
from research_radar.config import AppSettings
from research_radar.digest import DigestEngine
from research_radar.evidence.registry import EvidenceRegistry
from research_radar.exceptions import ConfigurationError
from research_radar.llm.gemini import GeminiProvider
from research_radar.llm.router import LLMRouter
from research_radar.research.orchestrator import ResearchOrchestrator
from research_radar.research.planner import ResearchPlanner
from research_radar.research.verifier import ClaimVerifier
from research_radar.security.tool_policy import validate_hermes_config
from research_radar.service import RadarService
from research_radar.state import DigestStore
from research_radar.tools.arxiv import ArxivSearchTool
from research_radar.tools.github import GitHubClient, GitHubRepositoryAnalyzer, GitHubSearchTool
from research_radar.tools.news import RssNewsTool
from research_radar.tools.registry import ToolRegistry
from research_radar.tools.web import BraveWebSearchTool
from research_radar.utils.retry import RetryPolicy


@dataclass(slots=True)
class ApplicationContainer:
    settings: AppSettings
    service: RadarService
    digest: DigestEngine
    gemini: GeminiProvider | None
    hermes: HermesHttpRuntime | None
    github: GitHubClient
    arxiv: ArxivSearchTool
    news: RssNewsTool
    web: BraveWebSearchTool | None
    orchestrator: ResearchOrchestrator | None = None

    async def aclose(self) -> None:
        if self.gemini is not None:
            await self.gemini.aclose()
        if self.hermes is not None:
            await self.hermes.aclose()
        await self.github.aclose()
        await self.arxiv.aclose()
        await self.news.aclose()
        if self.web is not None:
            await self.web.aclose()


def build_container(settings: AppSettings, *, require_runtime: bool = True) -> ApplicationContainer:
    retry = RetryPolicy(attempts=settings.max_retries)
    gemini = (
        GeminiProvider(
            api_key=settings.gemini_api_key.get_secret_value(),
            default_model=settings.gemini_model,
            fallback_models=settings.gemini_fallback_models,
            rpm_limit=settings.gemini_rpm_limit,
            timeout_seconds=settings.request_timeout_seconds,
            retry_policy=retry,
        )
        if settings.gemini_api_key
        else None
    )
    llm_router = (
        LLMRouter(
            gemini,
            default_model=settings.gemini_model,
            fast_model=settings.gemini_fast_model,
            reasoning_model=settings.gemini_reasoning_model,
        )
        if gemini is not None
        else None
    )
    hermes_key = (
        settings.require_hermes_key()
        if require_runtime
        else (settings.hermes_api_key.get_secret_value() if settings.hermes_api_key else None)
    )
    if hermes_key is not None:
        policy_violations = validate_hermes_config(
            api_key=hermes_key,
            timeout_seconds=settings.request_timeout_seconds,
            max_retries=settings.max_retries,
        )
        if policy_violations:
            raise ConfigurationError(
                f"Hermes security policy violation: {'; '.join(policy_violations)}"
            )
    hermes = (
        HermesHttpRuntime(
            base_url=str(settings.hermes_api_url),
            api_key=hermes_key,
            model_name=settings.hermes_model_name,
            timeout_seconds=max(settings.request_timeout_seconds, 120),
            retry_policy=retry,
        )
        if hermes_key
        else None
    )
    github = GitHubClient(
        token=settings.github_token.get_secret_value() if settings.github_token else None,
        base_url=str(settings.github_api_url),
        api_version=settings.github_api_version,
        timeout_seconds=settings.request_timeout_seconds,
        concurrency=settings.max_concurrency,
        retry_policy=retry,
    )
    arxiv = ArxivSearchTool(
        base_url=str(settings.arxiv_api_url),
        timeout_seconds=settings.request_timeout_seconds,
        min_interval_seconds=settings.arxiv_min_interval_seconds,
        retry_policy=retry,
    )
    news = RssNewsTool(
        settings.news_feed_urls,
        timeout_seconds=settings.request_timeout_seconds,
        concurrency=settings.max_concurrency,
    )
    web = (
        BraveWebSearchTool(
            api_key=settings.brave_search_api_key.get_secret_value(),
            base_url=str(settings.brave_search_api_url),
            timeout_seconds=settings.request_timeout_seconds,
            concurrency=settings.max_concurrency,
            retry_policy=retry,
        )
        if settings.brave_search_api_key
        else None
    )
    tools = ToolRegistry(
        github_search=GitHubSearchTool(github),
        github_analyzer=GitHubRepositoryAnalyzer(github),
        arxiv_search=arxiv,
        news_search=news,
        web_search=web,
    )
    evidence_registry = EvidenceRegistry(Path(settings.digest_state_path))
    claim_verifier = ClaimVerifier()
    planner = ResearchPlanner(llm_router=llm_router, hermes_runtime=hermes)
    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        evidence_registry=evidence_registry,
        claim_verifier=claim_verifier,
        llm_router=llm_router,
        hermes_runtime=hermes,
        planner=planner,
    )
    digest = DigestEngine(
        settings=settings,
        tools=tools,
        store=DigestStore(settings.digest_state_path),
        llm_router=llm_router,
        evidence_registry=evidence_registry,
        claim_verifier=claim_verifier,
    )
    service = RadarService(
        runtime=hermes,
        tools=tools,
        digest=digest,
        orchestrator=orchestrator,
    )
    return ApplicationContainer(
        settings=settings,
        service=service,
        digest=digest,
        gemini=gemini,
        hermes=hermes,
        github=github,
        arxiv=arxiv,
        news=news,
        web=web,
        orchestrator=orchestrator,
    )
