"""Application service coordinating the runtime, tools, and traces."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

from pydantic import BaseModel

from research_radar.agent.base import HermesRuntime
from research_radar.digest import DigestEngine
from research_radar.exceptions import ConfigurationError
from research_radar.observability.tracing import TraceStatus, TraceStore, TraceSummary
from research_radar.research.models import (
    ResearchBudget,
    ResearchMode,
    ResearchSynthesisResult,
)
from research_radar.research.orchestrator import ResearchOrchestrator
from research_radar.schemas import (
    PaperResult,
    RepositoryAnalysis,
    RepositoryResult,
    ResearchDigest,
    RuntimeResponse,
    SearchBatch,
    ToolHealth,
    WebResult,
)
from research_radar.tools.registry import ToolRegistry

RESEARCH_SYSTEM_INSTRUCTION = """You are the orchestration runtime for Autonomous AI Research Radar.
Act as an evidence-first AI research engineer. Use tools only when needed, distinguish facts from
inference, prefer primary sources, never invent citations, state uncertainty, and keep Telegram
answers concise and technically actionable. This early deployment has application-owned GitHub,
arXiv, and web adapters; do not claim those adapters ran unless their evidence is present in the
prompt or your own Hermes tools actually ran."""


class RadarStatus(BaseModel):
    runtime: ToolHealth
    tools: list[ToolHealth]
    metrics: TraceSummary


class RadarService:
    def __init__(
        self,
        *,
        runtime: HermesRuntime | None,
        tools: ToolRegistry,
        digest: DigestEngine | None = None,
        traces: TraceStore | None = None,
        orchestrator: ResearchOrchestrator | None = None,
    ) -> None:
        self.runtime = runtime
        self.tools = tools
        self.digest = digest
        self.traces = traces or TraceStore()
        self.orchestrator = orchestrator

    async def research(
        self,
        question: str,
        *,
        user_id: int,
        mode: ResearchMode = ResearchMode.QUICK,
        budget: ResearchBudget | None = None,
    ) -> ResearchSynthesisResult:
        """Conduct bounded autonomous research using the research orchestrator."""
        if self.orchestrator is not None:
            request_id = uuid4().hex
            trace = await self.traces.start(request_id, str(user_id), run_id=request_id)
            try:
                result = await self.orchestrator.conduct_research(
                    question, user_id=user_id, mode=mode, budget=budget, run_id=request_id
                )
                status = {
                    "completed": TraceStatus.SUCCESS,
                    "partial": TraceStatus.PARTIAL,
                    "failed": TraceStatus.FAILED,
                }[result.state.status.value]
                await self.traces.finish(trace.trace_id, status)
                return result
            except Exception as exc:
                await self.traces.record_error(trace.trace_id, type(exc).__name__)
                await self.traces.finish(trace.trace_id, TraceStatus.FAILED)
                raise

        # Fallback if orchestrator not initialized: delegate to Hermes
        response = await self.ask_agent(question, user_id=user_id)
        from research_radar.research.models import ConfidenceLevel, ResearchState, StopReason

        fallback_state = ResearchState(
            run_id=uuid4().hex,
            question=question,
            mode=mode,
            stop_reason=StopReason.ENOUGH_EVIDENCE,
        )
        return ResearchSynthesisResult(
            research_id=fallback_state.research_id,
            run_id=fallback_state.run_id,
            question=question,
            mode=mode,
            answer=response.text,
            key_findings=[response.text[:200]],
            confidence=ConfidenceLevel.MEDIUM,
            stop_reason=StopReason.ENOUGH_EVIDENCE,
            state=fallback_state,
        )

    async def ask_agent(self, text: str, *, user_id: int) -> RuntimeResponse:
        if self.runtime is None:
            raise ConfigurationError("Hermes belum dikonfigurasi untuk percakapan agentic.")
        task_id = uuid4().hex
        trace = await self.traces.start(task_id, str(user_id))
        try:
            result = await self.runtime.run(
                text,
                session_key=f"telegram:{user_id}",
                system_instruction=RESEARCH_SYSTEM_INSTRUCTION,
                request_id=task_id,
            )
            await self.traces.record_runtime_call(
                trace.trace_id,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            )
            await self.traces.finish(trace.trace_id, TraceStatus.SUCCESS)
            return result
        except Exception as exc:
            await self.traces.record_error(trace.trace_id, type(exc).__name__)
            await self.traces.finish(trace.trace_id, TraceStatus.FAILED)
            raise

    async def search_repositories(
        self, query: str, *, user_id: int, days: int | None = None, limit: int = 8
    ) -> SearchBatch[RepositoryResult]:
        trace = await self.traces.start(uuid4().hex, str(user_id))
        try:
            created_after = date.today() - timedelta(days=days) if days else None
            result = await self.tools.github_search.search(
                query,
                created_after=created_after,
                limit=limit,
                sort="stars",
            )
            await self.traces.record_tool_call(trace.trace_id, "github_search")
            await self.traces.finish(
                trace.trace_id, TraceStatus.PARTIAL if result.partial else TraceStatus.SUCCESS
            )
            return result
        except Exception as exc:
            await self.traces.record_error(trace.trace_id, type(exc).__name__)
            await self.traces.finish(trace.trace_id, TraceStatus.FAILED)
            raise

    async def analyze_repository(self, reference: str, *, user_id: int) -> RepositoryAnalysis:
        trace = await self.traces.start(uuid4().hex, str(user_id))
        try:
            result = await self.tools.github_analyzer.analyze(reference)
            await self.traces.record_tool_call(trace.trace_id, "github_analyzer")
            await self.traces.finish(
                trace.trace_id, TraceStatus.PARTIAL if result.warnings else TraceStatus.SUCCESS
            )
            return result
        except Exception as exc:
            await self.traces.record_error(trace.trace_id, type(exc).__name__)
            await self.traces.finish(trace.trace_id, TraceStatus.FAILED)
            raise

    async def search_papers(
        self,
        query: str,
        *,
        user_id: int,
        days: int | None = 30,
        limit: int = 8,
    ) -> SearchBatch[PaperResult]:
        trace = await self.traces.start(uuid4().hex, str(user_id))
        try:
            published_after = date.today() - timedelta(days=days) if days else None
            result = await self.tools.arxiv_search.search(
                query,
                categories=["cs.AI", "cs.LG", "cs.CL", "cs.SE"],
                published_after=published_after,
                limit=limit,
            )
            await self.traces.record_tool_call(trace.trace_id, "arxiv_search")
            await self.traces.finish(
                trace.trace_id, TraceStatus.PARTIAL if result.partial else TraceStatus.SUCCESS
            )
            return result
        except Exception as exc:
            await self.traces.record_error(trace.trace_id, type(exc).__name__)
            await self.traces.finish(trace.trace_id, TraceStatus.FAILED)
            raise

    async def search_web(
        self,
        query: str,
        *,
        user_id: int,
        days: int | None = 7,
        limit: int = 8,
    ) -> SearchBatch[WebResult]:
        if self.tools.web_search is None:
            raise ConfigurationError("Web search belum dikonfigurasi. Isi BRAVE_SEARCH_API_KEY.")
        trace = await self.traces.start(uuid4().hex, str(user_id))
        try:
            end = datetime.now(UTC).date() if days else None
            start = end - timedelta(days=days) if end and days else None
            result = await self.tools.web_search.search(
                query, limit=limit, start_date=start, end_date=end
            )
            await self.traces.record_tool_call(trace.trace_id, "web_search")
            await self.traces.finish(
                trace.trace_id, TraceStatus.PARTIAL if result.partial else TraceStatus.SUCCESS
            )
            return result
        except Exception as exc:
            await self.traces.record_error(trace.trace_id, type(exc).__name__)
            await self.traces.finish(trace.trace_id, TraceStatus.FAILED)
            raise

    async def status(self) -> RadarStatus:
        runtime_health = (
            await self.runtime.health()
            if self.runtime is not None
            else ToolHealth(name="hermes", configured=False, healthy=None)
        )
        return RadarStatus(
            runtime=runtime_health,
            tools=self.tools.status(),
            metrics=await self.traces.summary(),
        )

    async def generate_digest(self, *, force: bool = False) -> ResearchDigest:
        if self.digest is None:
            raise ConfigurationError("Pipeline digest belum dikonfigurasi.")
        return await self.digest.generate(force=force)

    async def digest_is_due(self) -> bool:
        return self.digest is not None and await self.digest.is_due()

    async def record_digest_sent(self, digest: ResearchDigest) -> None:
        if self.digest is None:
            raise ConfigurationError("Pipeline digest belum dikonfigurasi.")
        await self.digest.record_sent(digest)
