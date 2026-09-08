"""P1A Autonomous Research Orchestrator Core."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from pydantic import BaseModel, Field

from research_radar.agent.base import HermesRuntime
from research_radar.config import AppSettings
from research_radar.consensus.clustering import ClaimClusterer
from research_radar.consensus.contradiction import ContradictionDetector
from research_radar.consensus.engine import ConsensusEngine
from research_radar.consensus.extractor import ClaimExtractor
from research_radar.consensus.models import (
    ConsensusLevel,
    ConsensusReport,
    ContradictionSeverity,
)
from research_radar.consensus.synthesis import ConsensusSynthesizer
from research_radar.evidence.canonicalizer import content_fingerprint
from research_radar.evidence.models import (
    ClaimType,
    EvidenceStatus,
    StructuredClaim,
)
from research_radar.evidence.quality import classify_source_authority
from research_radar.evidence.registry import EvidenceRegistry
from research_radar.exceptions import ConfigurationError
from research_radar.llm.base import LLMRequest
from research_radar.llm.router import LLMRouter, Workload
from research_radar.observability.logging import current_run_id, redact_text
from research_radar.research.models import (
    ConfidenceLevel,
    QueryExecutionRecord,
    ResearchBudget,
    ResearchMode,
    ResearchState,
    ResearchStatus,
    ResearchSynthesisResult,
    SearchStep,
    StopReason,
)
from research_radar.research.persistence import ResearchStore
from research_radar.research.planner import ResearchPlanner, fallback_query
from research_radar.research.verifier import ClaimVerifier, VerificationReport
from research_radar.retrieval import (
    RetrievalFailure,
    classify_failure,
    current_call_id,
    current_iteration,
    operation_deadline,
    operation_sink,
)
from research_radar.security.untrusted_content import (
    EVIDENCE_BOUNDARY_INSTRUCTION,
    wrap_evidence_for_llm,
)
from research_radar.tools.registry import ToolRegistry
from research_radar.utils.deadline import workflow_deadline

logger = logging.getLogger(__name__)

ALLOWED_RESEARCH_TOOLS = {"github", "arxiv", "news", "web"}


def _normalize_query_key(tool: str, query: str) -> str:
    """Normalize tool and query for deterministic duplicate detection."""
    clean_tool = tool.strip().lower()
    clean_query = " ".join(query.strip().lower().split())
    return f"{clean_tool}:{clean_query}"


class LLMSynthesisSchema(BaseModel):
    answer: str
    key_findings: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    claims: list[dict[str, str]] = Field(default_factory=list)


class ResearchOrchestrator:
    """Bounded iterative research orchestrator integrating consensus and P0 verification."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        tools: ToolRegistry,
        evidence_registry: EvidenceRegistry,
        claim_verifier: ClaimVerifier,
        llm_router: LLMRouter | None = None,
        hermes_runtime: HermesRuntime | None = None,
        planner: ResearchPlanner | None = None,
        store: ResearchStore | None = None,
        extractor: ClaimExtractor | None = None,
        clusterer: ClaimClusterer | None = None,
        contradiction_detector: ContradictionDetector | None = None,
        consensus_engine: ConsensusEngine | None = None,
        consensus_synthesizer: ConsensusSynthesizer | None = None,
    ) -> None:
        self.settings = settings
        self.tools = tools
        self.evidence_registry = evidence_registry
        self.claim_verifier = claim_verifier
        self.llm_router = llm_router
        self.hermes_runtime = hermes_runtime
        self.planner = planner or ResearchPlanner(
            llm_router=llm_router, hermes_runtime=hermes_runtime
        )
        self.store = store or ResearchStore(evidence_registry.path)
        self.extractor = extractor or ClaimExtractor(llm_router=llm_router)
        self.clusterer = clusterer or ClaimClusterer()
        self.contradiction_detector = contradiction_detector or ContradictionDetector()
        self.consensus_engine = consensus_engine or ConsensusEngine()
        self.consensus_synthesizer = consensus_synthesizer or ConsensusSynthesizer(
            llm_router=llm_router
        )

    async def conduct_research(
        self,
        question: str,
        *,
        user_id: int | None = None,
        mode: ResearchMode = ResearchMode.QUICK,
        budget: ResearchBudget | None = None,
        run_id: str | None = None,
    ) -> ResearchSynthesisResult:
        token = current_run_id.set(run_id or uuid4().hex)
        deadline_token = workflow_deadline.set(
            perf_counter() + (budget or ResearchBudget.for_mode(mode)).max_wall_time_seconds
        )
        try:
            return await self._conduct_research(question, user_id=user_id, mode=mode, budget=budget)
        finally:
            current_run_id.reset(token)
            workflow_deadline.reset(deadline_token)

    async def _conduct_research(
        self,
        question: str,
        *,
        user_id: int | None = None,
        mode: ResearchMode = ResearchMode.QUICK,
        budget: ResearchBudget | None = None,
    ) -> ResearchSynthesisResult:
        """Execute autonomous research with bounded iterations, gap detection, and verification."""
        run_id = current_run_id.get() or uuid4().hex
        start_wall_time = perf_counter()

        active_budget = budget or ResearchBudget.for_mode(mode)
        state = ResearchState(
            run_id=run_id,
            question=question,
            mode=mode,
            budget=active_budget,
            status=ResearchStatus.PLANNING,
        )

        logger.info(
            "Starting autonomous research",
            extra={
                "event": "research_started",
                "research_id": state.research_id,
                "run_id": run_id,
                "mode": mode.value,
                "question_hash": __import__("hashlib").sha256(question.encode()).hexdigest()[:16],
            },
        )

        # 1. Planning Phase
        deadline = start_wall_time + active_budget.max_wall_time_seconds
        try:
            if active_budget.max_llm_calls <= 0:
                plan = ResearchPlanner().fallback_plan(question, mode)
            else:
                state.llm_calls += 1
                async with asyncio.timeout(max(0, (deadline - perf_counter()) / 4)):
                    plan = await self.planner.plan(question, mode)
        except Exception as exc:
            plan = ResearchPlanner().fallback_plan(question, mode)
            state.uncertainties.append(f"Planning fallback: {type(exc).__name__}")
        state.plan = plan
        state.objective = plan.objective

        logger.info(
            "Research plan generated",
            extra={
                "event": "research_plan_created",
                "research_id": state.research_id,
                "sub_questions": len(plan.sub_questions),
                "initial_steps": len(plan.search_steps),
            },
        )

        # 2. Iteration Loop
        pending_steps = list(plan.search_steps)
        collected_evidence: dict[str, dict[str, object]] = {}
        executed_query_keys: set[str] = set()
        fallback_keys: set[str] = set()
        unavailable: set[str] = set()

        while state.iterations < active_budget.max_iterations:
            state.iterations += 1
            iteration_new_evidence = 0

            # Check time budget
            elapsed = perf_counter() - start_wall_time
            if elapsed > active_budget.max_wall_time_seconds:
                state.stop_reason = StopReason.TIME_BUDGET_EXHAUSTED
                break

            state.status = ResearchStatus.SEARCHING
            logger.info(
                "Research iteration started",
                extra={
                    "event": "research_iteration_started",
                    "research_id": state.research_id,
                    "iteration": state.iterations,
                    "steps": len(pending_steps),
                },
            )

            # Execute search steps
            for step in pending_steps:
                if perf_counter() >= deadline:
                    state.stop_reason = StopReason.TIME_BUDGET_EXHAUSTED
                    break
                if len(collected_evidence) >= active_budget.max_evidence:
                    state.stop_reason = StopReason.MAX_EVIDENCE
                    break
                # Check tool & query budget
                if state.tool_calls >= active_budget.max_tool_calls:
                    state.stop_reason = StopReason.MAX_TOOL_CALLS
                    break
                if len(state.queries_executed) >= active_budget.max_queries:
                    state.stop_reason = StopReason.MAX_QUERIES
                    break

                tool_name = step.tool.lower().strip()
                query_text = " ".join(step.query.split())
                query_key = _normalize_query_key(tool_name, query_text)

                # Enforce allowed tools boundary
                if tool_name not in ALLOWED_RESEARCH_TOOLS:
                    logger.warning(
                        "Rejected unauthorized research tool",
                        extra={"event": "tool_rejected", "tool": tool_name},
                    )
                    state.queries_executed.append(
                        QueryExecutionRecord(
                            tool=tool_name,
                            query=query_text,
                            iteration=state.iterations,
                            status="rejected",
                        )
                    )
                    continue

                # Skip duplicate queries
                if query_key in executed_query_keys:
                    continue
                executed_query_keys.add(query_key)

                record = QueryExecutionRecord(
                    tool=tool_name,
                    query=redact_text(query_text),
                    iteration=state.iterations,
                    fallback=step.reason == "deterministic_fallback",
                )
                state.tool_calls += 1
                call_token = current_call_id.set(record.call_id)
                sink_token = operation_sink.set(record.operations)
                iteration_token = current_iteration.set(state.iterations)
                call_started = perf_counter()
                deadline_token = operation_deadline.set(
                    min(deadline, call_started + self.settings.request_timeout_seconds)
                )
                try:
                    async with asyncio.timeout(
                        max(
                            0, min(deadline - perf_counter(), self.settings.request_timeout_seconds)
                        )
                    ):
                        new_ev, total_results, status_code = await self._execute_search_step(
                            tool_name,
                            query_text,
                            collected_evidence,
                            record=record,
                            max_evidence=active_budget.max_evidence,
                        )
                except TimeoutError as exc:
                    new_ev, total_results, status_code = 0, 0, "failed"
                    record.failures.append(classify_failure(tool_name, exc))
                    if perf_counter() >= deadline:
                        state.stop_reason = StopReason.TIME_BUDGET_EXHAUSTED
                finally:
                    current_call_id.reset(call_token)
                    operation_deadline.reset(deadline_token)
                    operation_sink.reset(sink_token)
                    current_iteration.reset(iteration_token)
                record.latency_ms = (perf_counter() - call_started) * 1000
                record.result_count = total_results
                record.new_evidence_count = new_ev
                record.status = status_code
                iteration_new_evidence += new_ev
                state.queries_executed.append(record)
                state.provider_failures.extend(record.failures)
                logger.info(
                    "Research query finished",
                    extra={
                        "event": "research_query",
                        "call_id": record.call_id,
                        "provider": tool_name,
                        "run_id": run_id,
                        "iteration": state.iterations,
                        "latency_ms": record.latency_ms,
                        "result_count": total_results,
                        "status": status_code,
                        "failure_categories": [f.failure_category for f in record.failures],
                        "fallback": record.fallback,
                    },
                )
                categories = {f.failure_category for f in record.failures}
                if categories - {"empty_result", "invalid_query", "query_quality_failure"}:
                    unavailable.add(tool_name)
                if total_results == 0 and not record.fallback:
                    relaxed = fallback_query(query_text)
                    query_problem = not categories or categories <= {
                        "empty_result",
                        "invalid_query",
                        "query_quality_failure",
                    }
                    target = tool_name
                    if not query_problem or relaxed.lower() == query_text.lower():
                        target = next(
                            (
                                name
                                for name in ("github", "arxiv", "news", "web")
                                if name != tool_name
                                and name not in unavailable
                                and (name != "web" or self.tools.web_search is not None)
                                and (name != "news" or self.tools.news_search is not None)
                            ),
                            "",
                        )
                    key = _normalize_query_key(target, relaxed)
                    if (
                        target
                        and relaxed
                        and key not in executed_query_keys
                        and key not in fallback_keys
                    ):
                        fallback_keys.add(key)
                        pending_steps.append(
                            SearchStep(tool=target, query=relaxed, reason="deterministic_fallback")
                        )

                if len(collected_evidence) >= active_budget.max_evidence:
                    state.stop_reason = StopReason.MAX_EVIDENCE
                    break

            if state.stop_reason in (
                StopReason.MAX_TOOL_CALLS,
                StopReason.MAX_QUERIES,
                StopReason.MAX_EVIDENCE,
                StopReason.TIME_BUDGET_EXHAUSTED,
            ):
                break

            # 3. Evaluation & Knowledge Gap Detection
            state.status = ResearchStatus.EVALUATING
            state.evidence_ids = list(collected_evidence.keys())

            sufficiency = self.planner.detect_gaps(
                plan,
                list(collected_evidence.values()),
                executed_query_keys,
                state.iterations,
                active_budget.max_iterations,
            )
            state.knowledge_gaps = sufficiency.gaps

            # Deterministic precedence check
            if sufficiency.sufficient:
                state.stop_reason = StopReason.ENOUGH_EVIDENCE
                break
            elif iteration_new_evidence == 0 and state.iterations > 1:
                state.stop_reason = StopReason.NO_NEW_EVIDENCE
                break
            elif state.iterations >= active_budget.max_iterations:
                state.stop_reason = StopReason.MAX_ITERATIONS
                break
            elif state.tool_calls >= active_budget.max_tool_calls:
                state.stop_reason = StopReason.MAX_TOOL_CALLS
                break
            elif len(state.queries_executed) >= active_budget.max_queries:
                state.stop_reason = StopReason.MAX_QUERIES
                break
            elif len(collected_evidence) >= active_budget.max_evidence:
                state.stop_reason = StopReason.MAX_EVIDENCE
                break
            elif perf_counter() - start_wall_time > active_budget.max_wall_time_seconds:
                state.stop_reason = StopReason.TIME_BUDGET_EXHAUSTED
                break

            # 4. Follow-up Steps Generation
            state.status = ResearchStatus.FOLLOW_UP
            pending_steps = [
                SearchStep(
                    tool=gap.suggested_tool,
                    query=gap.suggested_query,
                    reason=gap.description,
                )
                for gap in sufficiency.gaps
                if _normalize_query_key(gap.suggested_tool, gap.suggested_query)
                not in executed_query_keys
            ]

            if not pending_steps:
                state.stop_reason = StopReason.LOW_INFORMATION_GAIN
                break

        if state.stop_reason is None:
            if not collected_evidence:
                state.stop_reason = StopReason.FAILED
            elif sufficiency.sufficient:
                state.stop_reason = StopReason.ENOUGH_EVIDENCE
            else:
                state.stop_reason = StopReason.MAX_ITERATIONS

        state.evidence_ids = list(collected_evidence)
        state.coverage = {
            name: sum(item.get("source_type") == name for item in collected_evidence.values())
            for name in sorted(ALLOWED_RESEARCH_TOOLS)
        }
        if not collected_evidence and state.stop_reason not in {
            StopReason.TIME_BUDGET_EXHAUSTED,
            StopReason.MAX_TOOL_CALLS,
            StopReason.MAX_QUERIES,
            StopReason.MAX_LLM_CALLS,
        }:
            state.stop_reason = StopReason.FAILED

        # 4. Consensus & Contradiction Intelligence Phase
        state.status = ResearchStatus.CONSENSUS
        consensus_report = ConsensusReport()
        if collected_evidence:
            try:
                use_llm = (
                    mode == ResearchMode.DEEP
                    and self.llm_router is not None
                    and state.llm_calls < active_budget.max_llm_calls
                    and perf_counter() < deadline
                )
                if use_llm:
                    state.llm_calls += 1
                try:
                    async with asyncio.timeout(max(0, deadline - perf_counter())):
                        claims_extracted = await self.extractor.extract_claims(
                            list(collected_evidence.values()), use_llm=use_llm
                        )
                except TimeoutError:
                    state.stop_reason = StopReason.TIME_BUDGET_EXHAUSTED
                    claims_extracted = await self.extractor.extract_claims(
                        list(collected_evidence.values()), use_llm=False
                    )
                clusters = self.clusterer.cluster_claims(claims_extracted)
                contradictions = self.contradiction_detector.detect_contradictions(clusters)
                consensus_report = self.consensus_engine.assess_consensus(clusters, contradictions)
                state.consensus_report = consensus_report
            except Exception as exc:
                logger.warning(
                    "Consensus analysis encountered error; degrading gracefully",
                    extra={"event": "consensus_analysis_error", "error_type": type(exc).__name__},
                )
                consensus_report = ConsensusReport()
                state.consensus_report = consensus_report

        # 5. Synthesis Phase
        state.status = ResearchStatus.SYNTHESIZING
        force_deterministic = (
            state.llm_calls >= active_budget.max_llm_calls or perf_counter() >= deadline
        )
        if not force_deterministic and self.llm_router is not None and collected_evidence:
            state.llm_calls += 1
        try:
            if force_deterministic:
                synthesis_output, claims = await self.consensus_synthesizer.synthesize(
                    question,
                    list(collected_evidence.values()),
                    consensus_report,
                    force_deterministic=True,
                )
            else:
                async with asyncio.timeout(max(0, deadline - perf_counter())):
                    synthesis_output, claims = await self.consensus_synthesizer.synthesize(
                        question, list(collected_evidence.values()), consensus_report
                    )
        except Exception as exc:
            if isinstance(exc, TimeoutError):
                state.stop_reason = StopReason.TIME_BUDGET_EXHAUSTED
            state.uncertainties.append(f"Synthesis fallback: {type(exc).__name__}")
            synthesis_output, claims = await ConsensusSynthesizer().synthesize(
                question,
                list(collected_evidence.values()),
                consensus_report,
                force_deterministic=True,
            )
        if perf_counter() >= deadline:
            state.stop_reason = StopReason.TIME_BUDGET_EXHAUSTED

        # 6. Verification Phase
        state.status = ResearchStatus.VERIFYING
        evidence_content_map = {
            eid: str(data.get("description", ""))
            + " "
            + str(data.get("abstract", ""))
            + " "
            + str(data.get("readme_excerpt", ""))
            for eid, data in collected_evidence.items()
        }
        report = self.claim_verifier.verify(
            claims, set(collected_evidence.keys()), evidence_content_map
        )
        verified_claims = self.claim_verifier.downgrade_unsupported_claims(claims, report)

        # 7. Finalize Result
        if state.stop_reason == StopReason.ENOUGH_EVIDENCE and not state.provider_failures:
            state.status = ResearchStatus.COMPLETED
        elif not collected_evidence or state.stop_reason == StopReason.FAILED:
            state.status = ResearchStatus.FAILED
        else:
            state.status = ResearchStatus.PARTIAL
        state.completed_at = datetime.now(UTC)

        has_unresolved_high = any(g.importance == "high" for g in state.knowledge_gaps)
        confidence = self._compute_confidence(
            evidence_items=list(collected_evidence.values()),
            covered_sub_questions=len(plan.sub_questions) - len(state.knowledge_gaps),
            total_sub_questions=len(plan.sub_questions) or 1,
            report=report,
            has_unresolved_high_gaps=has_unresolved_high,
            consensus_report=consensus_report,
        )

        raw_answer = synthesis_output.get("answer")
        answer_str = str(raw_answer) if raw_answer else "Analisis riset selesai."
        raw_findings = synthesis_output.get("key_findings")
        findings_list = [str(f) for f in raw_findings] if isinstance(raw_findings, list) else []
        raw_uncertainties = synthesis_output.get("uncertainties")
        uncertainties_list = (
            [str(u) for u in raw_uncertainties]
            if isinstance(raw_uncertainties, list)
            else list(state.uncertainties)
        )
        uncertainties_list.extend(u for u in state.uncertainties if u not in uncertainties_list)
        if state.provider_failures:
            details = ", ".join(
                sorted({f"{f.provider}: {f.failure_category}" for f in state.provider_failures})
            )
            uncertainties_list.append(f"Cakupan sumber terbatas ({details}).")
            answer_str += f"\nCakupan sumber terbatas ({details})."
            confidence = (
                ConfidenceLevel.LOW
                if not collected_evidence
                else min(
                    confidence,
                    ConfidenceLevel.MEDIUM,
                    key=lambda c: {
                        ConfidenceLevel.LOW: 0,
                        ConfidenceLevel.MEDIUM: 1,
                        ConfidenceLevel.HIGH: 2,
                    }[c],
                )
            )
        state.uncertainties = uncertainties_list
        raw_agreements = synthesis_output.get("agreements")
        agreements_list = (
            [str(a) for a in raw_agreements]
            if isinstance(raw_agreements, list)
            else list(consensus_report.agreements)
        )
        raw_disagreements = synthesis_output.get("disagreements")
        disagreements_list = (
            [str(d) for d in raw_disagreements]
            if isinstance(raw_disagreements, list)
            else list(consensus_report.disagreements)
        )
        raw_dissent = synthesis_output.get("dissenting_views")
        dissent_list = (
            [str(d) for d in raw_dissent]
            if isinstance(raw_dissent, list)
            else list(consensus_report.dissenting_findings)
        )
        safe_conclusion = str(synthesis_output.get("safe_conclusion", ""))

        result = ResearchSynthesisResult(
            research_id=state.research_id,
            run_id=run_id,
            question=question,
            mode=mode,
            answer=answer_str,
            key_findings=findings_list,
            evidence_sources=list(collected_evidence.values()),
            uncertainties=uncertainties_list,
            remaining_gaps=[g.description for g in state.knowledge_gaps],
            confidence=confidence,
            stop_reason=state.stop_reason,
            claims=verified_claims,
            consensus=consensus_report.overall_consensus,
            agreements=agreements_list,
            disagreements=disagreements_list,
            dissenting_findings=dissent_list,
            safe_conclusion=safe_conclusion,
            consensus_report=consensus_report,
            state=state,
        )

        # Persist to SQLite
        try:
            self.store.save_run(result)
        except Exception as exc:
            result.uncertainties.append("Research run could not be persisted to SQLite.")
            state.uncertainties = result.uncertainties
            if state.status == ResearchStatus.COMPLETED:
                state.status = ResearchStatus.PARTIAL
            logger.warning(
                "Failed to persist research run to SQLite",
                extra={"event": "research_save_failed", "error_type": type(exc).__name__},
            )

        return result

    async def _execute_search_step(
        self,
        tool: str,
        query: str,
        collected_evidence: dict[str, dict[str, object]],
        *,
        record: QueryExecutionRecord | None = None,
        max_evidence: int = 30,
    ) -> tuple[int, int, str]:
        """Execute a safe tool query and register collected evidence."""
        new_evidence_count = 0
        total_items = 0
        status_code = "success"
        before_ids = set(collected_evidence)

        try:
            if tool == "github":
                batch = await self.tools.github_search.search(query, limit=5)
                total_items = len(batch.items)
                if record is not None:
                    record.failures.extend(batch.failures)
                    if batch.partial and not batch.failures:
                        record.failures.append(
                            RetrievalFailure(
                                provider=tool,
                                failure_category="partial_response",
                                error_code="INCOMPLETE_RESULTS",
                            )
                        )
                status_code = "partial" if batch.partial else "success" if total_items else "empty"
                for repo in batch.items:
                    if len(collected_evidence) >= max_evidence:
                        break
                    metadata = {
                        "description": repo.description or "",
                        "stars": repo.stars,
                        "language": repo.language or "",
                        "topics": repo.topics,
                    }
                    fingerprint = content_fingerprint("github", metadata)
                    url_str = str(repo.url)
                    authority, score = classify_source_authority("github", url_str, metadata)
                    ev_record, status = self.evidence_registry.register(
                        url=url_str,
                        source_type="github",
                        title=repo.full_name,
                        content_hash=fingerprint,
                        published_at=repo.created_at,
                        source_authority=authority,
                        authority_score=score,
                        metadata=metadata,
                    )
                    if status in (EvidenceStatus.NEW, EvidenceStatus.UPDATED):
                        new_evidence_count += 1
                    collected_evidence[ev_record.evidence_id] = {
                        "evidence_id": ev_record.evidence_id,
                        "source_type": "github",
                        "title": repo.full_name,
                        "url": ev_record.canonical_url,
                        "description": repo.description,
                        "authority": authority.value,
                        "stars": repo.stars,
                    }

            elif tool == "arxiv":
                batch_arxiv = await self.tools.arxiv_search.search(query, limit=5)
                total_items = len(batch_arxiv.items)
                if record is not None:
                    record.failures.extend(batch_arxiv.failures)
                    if batch_arxiv.partial and not batch_arxiv.failures:
                        record.failures.append(
                            RetrievalFailure(
                                provider=tool,
                                failure_category="partial_response",
                                error_code="INCOMPLETE_RESULTS",
                            )
                        )
                status_code = (
                    "partial" if batch_arxiv.partial else "success" if total_items else "empty"
                )
                for paper in batch_arxiv.items:
                    if len(collected_evidence) >= max_evidence:
                        break
                    metadata = {
                        "title": paper.title,
                        "abstract": paper.abstract,
                        "categories": paper.categories,
                    }
                    fingerprint = content_fingerprint("arxiv", metadata)
                    url_str = str(paper.url)
                    authority, score = classify_source_authority("arxiv", url_str, metadata)
                    ev_record, status = self.evidence_registry.register(
                        url=url_str,
                        source_type="arxiv",
                        title=paper.title,
                        content_hash=fingerprint,
                        published_at=paper.published_at,
                        source_authority=authority,
                        authority_score=score,
                        metadata=metadata,
                    )
                    if status in (EvidenceStatus.NEW, EvidenceStatus.UPDATED):
                        new_evidence_count += 1
                    collected_evidence[ev_record.evidence_id] = {
                        "evidence_id": ev_record.evidence_id,
                        "source_type": "arxiv",
                        "title": paper.title,
                        "url": ev_record.canonical_url,
                        "abstract": paper.abstract[:500],
                        "authority": authority.value,
                    }

            elif tool == "web" and self.tools.web_search is not None:
                batch_web = await self.tools.web_search.search(query, limit=5)
                total_items = len(batch_web.items)
                if record is not None:
                    record.failures.extend(batch_web.failures)
                    if batch_web.partial and not batch_web.failures:
                        record.failures.append(
                            RetrievalFailure(
                                provider=tool,
                                failure_category="partial_response",
                                error_code="INCOMPLETE_RESULTS",
                            )
                        )
                status_code = (
                    "partial" if batch_web.partial else "success" if total_items else "empty"
                )
                for web_res in batch_web.items:
                    if len(collected_evidence) >= max_evidence:
                        break
                    metadata = {"title": web_res.title, "description": web_res.description}
                    fingerprint = content_fingerprint("web", metadata)
                    url_str = str(web_res.url)
                    authority, score = classify_source_authority("web", url_str, metadata)
                    ev_record, status = self.evidence_registry.register(
                        url=url_str,
                        source_type="web",
                        title=web_res.title,
                        content_hash=fingerprint,
                        source_authority=authority,
                        authority_score=score,
                        metadata=metadata,
                    )
                    if status in (EvidenceStatus.NEW, EvidenceStatus.UPDATED):
                        new_evidence_count += 1
                    collected_evidence[ev_record.evidence_id] = {
                        "evidence_id": ev_record.evidence_id,
                        "source_type": "web",
                        "title": web_res.title,
                        "url": ev_record.canonical_url,
                        "description": web_res.description,
                        "authority": authority.value,
                    }

            elif tool == "news" and self.tools.news_search is not None:
                batch_news = await self.tools.news_search.search((query,), limit=5)
                total_items = len(batch_news.items)
                if record is not None:
                    record.failures.extend(batch_news.failures)
                    if batch_news.partial and not batch_news.failures:
                        record.failures.append(
                            RetrievalFailure(
                                provider=tool,
                                failure_category="partial_response",
                                error_code="INCOMPLETE_RESULTS",
                            )
                        )
                status_code = (
                    "partial" if batch_news.partial else "success" if total_items else "empty"
                )
                for news_res in batch_news.items:
                    if len(collected_evidence) >= max_evidence:
                        break
                    metadata = {"title": news_res.title, "description": news_res.description}
                    fingerprint = content_fingerprint("news", metadata)
                    url_str = str(news_res.url)
                    authority, score = classify_source_authority("news", url_str, metadata)
                    ev_record, status = self.evidence_registry.register(
                        url=url_str,
                        source_type="news",
                        title=news_res.title,
                        content_hash=fingerprint,
                        source_authority=authority,
                        authority_score=score,
                        metadata=metadata,
                    )
                    if status in (EvidenceStatus.NEW, EvidenceStatus.UPDATED):
                        new_evidence_count += 1
                    collected_evidence[ev_record.evidence_id] = {
                        "evidence_id": ev_record.evidence_id,
                        "source_type": "news",
                        "title": news_res.title,
                        "url": ev_record.canonical_url,
                        "description": news_res.description,
                        "authority": authority.value,
                    }

            else:
                raise ConfigurationError("Optional provider is not configured")
            if record is not None and total_items == 0 and not record.failures:
                record.failures.append(
                    RetrievalFailure(
                        provider=tool, failure_category="empty_result", error_code="EMPTY_RESULT"
                    )
                )
        except Exception as exc:
            if record is not None:
                record.failures.append(classify_failure(tool, exc))
            logger.warning(
                "Collector step failed gracefully",
                extra={
                    "event": "research_collector_error",
                    "tool": tool,
                    "failure_category": classify_failure(tool, exc).failure_category,
                },
            )
            status_code = "failed"

        new_evidence_count = len(set(collected_evidence) - before_ids)
        return new_evidence_count, total_items, status_code

    async def _synthesize(
        self,
        question: str,
        state: ResearchState,
        evidence_items: list[dict[str, object]],
        force_deterministic: bool = False,
    ) -> tuple[dict[str, object], list[StructuredClaim]]:
        """Synthesize final findings with explicit claim grounding."""
        if not evidence_items:
            return {
                "answer": "Tidak ditemukan bukti memadai dari sumber terdaftar.",
                "key_findings": [
                    "Tidak ada repository, paper, atau sumber web relevan yang ditemukan."
                ],
                "uncertainties": ["Ketiadaan bukti empiris."],
            }, []

        if self.llm_router is None or force_deterministic:
            # Deterministic synthesis fallback
            findings = [
                f"{item.get('title')}: {item.get('description', '') or item.get('abstract', '')}"[
                    :150
                ]
                for item in evidence_items[:4]
            ]
            claims = [
                StructuredClaim(
                    text=f[:120],
                    claim_type=ClaimType.FACT,
                    evidence_ids=[str(evidence_items[idx].get("evidence_id"))],
                    confidence=0.8,
                )
                for idx, f in enumerate(findings)
            ]
            answer_text = (
                f"Berdasarkan {len(evidence_items)} sumber bukti terdaftar:\n"
                + "\n".join(f"- {f}" for f in findings)
            )
            return {
                "answer": answer_text,
                "key_findings": findings,
                "uncertainties": ["Sintesis dihasilkan melalui mode deterministik tanpa LLM."],
            }, claims

        # Structured synthesis with Gemini
        evidence_json = json.dumps(evidence_items, default=str, ensure_ascii=False)
        wrapped_evidence = wrap_evidence_for_llm(evidence_json)

        prompt = (
            f"Pertanyaan Riset: '{question}'\n\n"
            f"{wrapped_evidence}\n\n"
            "Tugas: Buat sintesis riset teknis yang komprehensif dan objektif berbasis bukti.\n"
            "Instruksi:\n"
            "1. Jawab pertanyaan utama secara langsung dan terstruktur.\n"
            "2. Cantumkan 3-5 poin key_findings utama dengan mengaitkan ID bukti.\n"
            "3. Sebutkan uncertainties dan keterbatasan data secara transparan.\n"
            "4. Bedakan fakta eksplisit dari penalaran logis (gunakan penanda 'Inferensi:').\n"
            "5. Kembalikan JSON dengan format:\n"
            "{\n"
            '  "answer": "Ringkasan jawaban komprehensif",\n'
            '  "key_findings": ["Poin temuan 1", "Poin temuan 2"],\n'
            '  "uncertainties": ["Keterbatasan 1"],\n'
            '  "claims": [\n'
            '    {"text": "Pernyataan faktual", "claim_type": "fact", "evidence_id": "id"}\n'
            "  ]\n"
            "}"
        )

        try:
            response = await self.llm_router.generate_structured(
                LLMRequest(
                    prompt=prompt,
                    system_instruction=(
                        f"{EVIDENCE_BOUNDARY_INSTRUCTION} "
                        "You are an evidence-first AI research scientist. Ground all findings in "
                        "the provided evidence block. Never fabricate citations or capabilities."
                    ),
                    max_output_tokens=2048,
                ),
                LLMSynthesisSchema,
                Workload.REASONING,
            )
            data = response.data
            claims = []
            for c in data.claims:
                text = c.get("text", "")
                if not text:
                    continue
                ev_id = c.get("evidence_id")
                is_interp = c.get("claim_type") == "interpretation" or text.startswith("Inferensi:")
                claims.append(
                    StructuredClaim(
                        text=text,
                        claim_type=ClaimType.INTERPRETATION if is_interp else ClaimType.FACT,
                        evidence_ids=[str(ev_id)] if ev_id else [],
                        confidence=0.85,
                    )
                )
            return {
                "answer": data.answer,
                "key_findings": data.key_findings,
                "uncertainties": data.uncertainties,
            }, claims
        except Exception as exc:
            logger.warning(
                "LLM synthesis failed; falling back to deterministic summary",
                extra={"event": "synthesis_fallback", "error_type": type(exc).__name__},
            )
            findings = [
                (
                    f"{item.get('title', 'Sumber')}: "
                    f"{item.get('description', '') or item.get('abstract', '')}"
                )[:150]
                for item in evidence_items[:4]
            ]
            err_answer = (
                f"Sintesis terdegradasi ({len(evidence_items)} sumber bukti terkumpul):\n"
                + "\n".join(f"- {f}" for f in findings)
            )
            return {
                "answer": err_answer,
                "key_findings": findings,
                "uncertainties": [f"Sintesis AI mengalami kendala: {type(exc).__name__}"],
            }, []

    def _compute_confidence(
        self,
        evidence_items: list[dict[str, object]],
        covered_sub_questions: int,
        total_sub_questions: int,
        report: VerificationReport | object,
        has_unresolved_high_gaps: bool = False,
        consensus_report: ConsensusReport | None = None,
    ) -> ConfidenceLevel:
        """Derive conservative confidence label from empirical indicators and consensus."""
        evidence_count = len(evidence_items)
        if evidence_count == 0:
            return ConfidenceLevel.LOW

        coverage_ratio = covered_sub_questions / total_sub_questions

        distinct_sources = len(
            {str(item.get("source_type")) for item in evidence_items if item.get("source_type")}
        )
        has_authoritative = any(
            item.get("authority") in ("primary", "official", "academic") for item in evidence_items
        )
        unsupported_count = getattr(report, "unsupported_claims", 0)

        has_unresolved_high_contra = False
        weak_consensus = False
        if consensus_report is not None:
            has_unresolved_high_contra = any(
                c.severity == ContradictionSeverity.HIGH and not c.resolved_by_version
                for c in consensus_report.contradictions
            )
            weak_consensus = consensus_report.overall_consensus in (
                ConsensusLevel.MIXED,
                ConsensusLevel.WEAK,
                ConsensusLevel.INSUFFICIENT,
            )

        # High confidence requires:
        # - Zero unresolved high-importance knowledge gaps
        # - Zero unresolved high-severity contradictions
        # - Strong or moderate consensus (not mixed/weak)
        # - Strong sub-question coverage (>= 75%)
        # - At least 3 evidence items
        # - Source diversity (>= 2 distinct sources) OR at least 1 authoritative source
        # - No unsupported factual claims in verification report
        if (
            not has_unresolved_high_gaps
            and not has_unresolved_high_contra
            and not weak_consensus
            and coverage_ratio >= 0.75
            and evidence_count >= 3
            and (distinct_sources >= 2 or has_authoritative)
            and unsupported_count == 0
        ):
            return ConfidenceLevel.HIGH

        if coverage_ratio >= 0.40 and evidence_count >= 2:
            return ConfidenceLevel.MEDIUM

        return ConfidenceLevel.LOW
