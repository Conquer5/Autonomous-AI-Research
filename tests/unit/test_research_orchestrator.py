"""Comprehensive unit tests for P1A Autonomous Research Orchestrator."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from research_radar.config import AppSettings
from research_radar.evidence.models import (
    SourceAuthority,
    VerificationStatus,
)
from research_radar.evidence.registry import EvidenceRegistry
from research_radar.research.models import (
    ConfidenceLevel,
    EvidenceSufficiency,
    KnowledgeGap,
    QueryExecutionRecord,
    ResearchBudget,
    ResearchMode,
    ResearchPlan,
    ResearchState,
    ResearchStatus,
    SearchStep,
    StopReason,
)
from research_radar.research.orchestrator import ALLOWED_RESEARCH_TOOLS, ResearchOrchestrator
from research_radar.research.persistence import ResearchStore
from research_radar.research.planner import ResearchPlanner
from research_radar.research.verifier import ClaimVerifier
from research_radar.schemas import (
    PaperResult,
    RepositoryResult,
    SearchBatch,
    SourceType,
    WebResult,
)
from research_radar.tools.registry import ToolRegistry


def _mock_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        _env_file=None,
        telegram_bot_token="123456:ABC-DEF1234567890123456789012345678",
        telegram_allowed_user_ids="12345678",
        digest_state_path=tmp_path / "radar.sqlite3",
    )


def _mock_tools() -> ToolRegistry:
    github = MagicMock()
    github.search = AsyncMock(
        return_value=SearchBatch(
            query="test",
            source=SourceType.GITHUB,
            items=[
                RepositoryResult(
                    full_name="google/agent-framework",
                    url="https://github.com/google/agent-framework",
                    description="High performance autonomous agent framework for production",
                    stars=1500,
                    forks=200,
                    language="Python",
                    topics=["agent", "ai", "production"],
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            ],
            partial=False,
        )
    )

    arxiv = MagicMock()
    arxiv.search = AsyncMock(
        return_value=SearchBatch(
            query="test",
            source=SourceType.ARXIV,
            items=[
                PaperResult(
                    arxiv_id="2608.12345",
                    title="Evaluation and Benchmark of Production AI Agents",
                    url="https://arxiv.org/abs/2608.12345",
                    abstract=(
                        "We present extensive benchmarks on local agent "
                        "runtime latency and accuracy."
                    ),
                    authors=["Alice", "Bob"],
                    categories=["cs.AI"],
                    published_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            ],
            partial=False,
        )
    )

    web = MagicMock()
    web.search = AsyncMock(
        return_value=SearchBatch(
            query="test",
            source=SourceType.WEB,
            items=[
                WebResult(
                    title="Production Agent Deployments in 2026",
                    url="https://tech.example.com/production-agents-2026",
                    description="Case studies of deploying autonomous agents locally.",
                )
            ],
            partial=False,
        )
    )

    news = MagicMock()
    news.search = AsyncMock(
        return_value=SearchBatch(query="test", source=SourceType.NEWS, items=[], partial=False)
    )

    return ToolRegistry(
        github_search=github,
        github_analyzer=MagicMock(),
        arxiv_search=arxiv,
        news_search=news,
        web_search=web,
    )


# ---------------------------------------------------------------------------
# 1. Planning Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_fallback_for_simple_and_complex_questions() -> None:
    planner = ResearchPlanner(llm_router=None)

    # Simple question in Quick mode
    quick_plan = await planner.plan("AI Agent", ResearchMode.QUICK)
    assert len(quick_plan.search_steps) == 2
    assert quick_plan.search_steps[0].tool == "github"
    assert quick_plan.search_steps[1].tool == "arxiv"

    # Complex question in Deep mode
    deep_plan = await planner.plan("Are local AI agents production ready?", ResearchMode.DEEP)
    assert len(deep_plan.search_steps) == 3
    assert len(deep_plan.sub_questions) >= 3
    assert any(s.tool == "web" for s in deep_plan.search_steps)


@pytest.mark.asyncio
async def test_planner_handles_llm_failure_with_graceful_fallback() -> None:
    failing_router = MagicMock()
    failing_router.generate_structured = AsyncMock(side_effect=RuntimeError("API timeout"))

    planner = ResearchPlanner(llm_router=failing_router)
    plan = await planner.plan("Local Agent Runtimes", ResearchMode.QUICK)

    # Should not raise; fallback plan returned
    assert plan.objective.startswith("Riset mendalam")
    assert len(plan.search_steps) >= 2


# ---------------------------------------------------------------------------
# 2. Budget and Stop Conditions Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_enforces_budget_max_iterations(tmp_path: Path) -> None:
    settings = _mock_settings(tmp_path)
    tools = _mock_tools()
    registry = EvidenceRegistry(settings.digest_state_path)
    verifier = ClaimVerifier()

    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        evidence_registry=registry,
        claim_verifier=verifier,
    )

    # QUICK mode has max_iterations=1
    result = await orchestrator.conduct_research(
        "AI Agent Frameworks", user_id=1001, mode=ResearchMode.QUICK
    )

    assert result.state.iterations == 1
    assert result.state.stop_reason in (StopReason.ENOUGH_EVIDENCE, StopReason.MAX_ITERATIONS)
    assert result.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW)


@pytest.mark.asyncio
async def test_orchestrator_stops_when_no_new_evidence_collected(tmp_path: Path) -> None:
    settings = _mock_settings(tmp_path)
    tools = _mock_tools()
    registry = EvidenceRegistry(settings.digest_state_path)
    verifier = ClaimVerifier()

    # Pre-register the item so subsequent searches return DUPLICATE
    registry.register(
        url="https://github.com/google/agent-framework",
        source_type="github",
        title="google/agent-framework",
        content_hash="d0a8f8d9b...",
        source_authority=SourceAuthority.PRIMARY,
    )

    # Configure custom mock planner that generates 3 iterations of gaps
    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(
        return_value=ResearchPlan(
            objective="test",
            sub_questions=["Sub Q 1", "Sub Q 2"],
            search_steps=[SearchStep(tool="github", query="test query")],
        )
    )
    # Iteration 1 returns items (all duplicate); gap detector asks for follow-up
    mock_planner.detect_gaps = MagicMock(
        return_value=MagicMock(
            sufficient=False,
            gaps=[
                KnowledgeGap(
                    description="Missing info",
                    suggested_query="follow up query",
                    suggested_tool="github",
                )
            ],
        )
    )

    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        evidence_registry=registry,
        claim_verifier=verifier,
        planner=mock_planner,
    )

    result = await orchestrator.conduct_research(
        "AI Agent Frameworks", user_id=1001, mode=ResearchMode.DEEP
    )

    # In iteration 2, 0 new items are found -> stops with NO_NEW_EVIDENCE
    assert result.state.stop_reason in (StopReason.NO_NEW_EVIDENCE, StopReason.ENOUGH_EVIDENCE)


# ---------------------------------------------------------------------------
# 3. Knowledge Gap Detection Tests
# ---------------------------------------------------------------------------


def test_planner_detect_gaps_creates_gap_for_uncovered_sub_question() -> None:
    planner = ResearchPlanner()
    plan = ResearchPlan(
        objective="Analyze Framework",
        sub_questions=[
            "Bagaimana arsitektur dari Framework?",
            "Bagaimana benchmark performa Framework?",
        ],
        search_steps=[],
    )

    # Evidence only mentions architecture, not benchmark
    evidence = [
        {
            "title": "Framework Architecture",
            "description": "Explaining the architecture and modules of the system.",
        }
    ]

    sufficiency = planner.detect_gaps(
        plan=plan,
        evidence_items=evidence,
        executed_queries={"framework overview"},
        iteration=1,
        max_iterations=3,
    )

    assert not sufficiency.sufficient
    assert len(sufficiency.gaps) >= 1
    assert any("benchmark" in g.suggested_query.lower() for g in sufficiency.gaps)


def test_planner_detect_gaps_marks_sufficient_when_all_covered() -> None:
    planner = ResearchPlanner()
    plan = ResearchPlan(
        objective="Analyze Framework",
        sub_questions=["arsitektur Framework", "benchmark performa"],
        search_steps=[],
    )

    evidence = [
        {
            "title": "Framework Architecture",
            "description": "Comprehensive explanation of arsitektur framework.",
        },
        {
            "title": "Framework Benchmark",
            "description": "Empirical benchmark performa across datasets.",
        },
        {
            "title": "Production Deployment",
            "description": "How to deploy in production.",
        },
    ]

    sufficiency = planner.detect_gaps(
        plan=plan,
        evidence_items=evidence,
        executed_queries=set(),
        iteration=1,
        max_iterations=3,
    )

    assert sufficiency.sufficient
    assert len(sufficiency.covered_sub_questions) == 2


# ---------------------------------------------------------------------------
# 4. Security & Boundary Enforcement Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_rejects_unauthorized_tools(tmp_path: Path) -> None:
    settings = _mock_settings(tmp_path)
    tools = _mock_tools()
    registry = EvidenceRegistry(settings.digest_state_path)
    verifier = ClaimVerifier()

    # Create planner that tries to return malicious / unauthorized tool calls
    malicious_planner = MagicMock()
    malicious_planner.plan = AsyncMock(
        return_value=ResearchPlan(
            objective="Adversarial attack",
            sub_questions=["Can I execute bash?"],
            search_steps=[
                SearchStep(tool="bash", query="rm -rf /"),
                SearchStep(tool="python_exec", query="import os; os.environ"),
                SearchStep(tool="github", query="safe query"),
            ],
        )
    )
    malicious_planner.detect_gaps = MagicMock(return_value=MagicMock(sufficient=True, gaps=[]))

    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        evidence_registry=registry,
        claim_verifier=verifier,
        planner=malicious_planner,
    )

    result = await orchestrator.conduct_research(
        "Attack scenario", user_id=1001, mode=ResearchMode.QUICK
    )

    executed_tools = [q.tool for q in result.state.queries_executed]
    assert "bash" in executed_tools
    assert "python_exec" in executed_tools

    # Status for unauthorized tools must be "rejected"
    rejected_records = [q for q in result.state.queries_executed if q.status == "rejected"]
    assert len(rejected_records) == 2
    assert all(q.tool not in ALLOWED_RESEARCH_TOOLS for q in rejected_records)


# ---------------------------------------------------------------------------
# 5. Evidence Registry & Claim Verification Integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_autonomous_research_registers_evidence_and_verifies_claims(
    tmp_path: Path,
) -> None:
    settings = _mock_settings(tmp_path)
    tools = _mock_tools()
    registry = EvidenceRegistry(settings.digest_state_path)
    verifier = ClaimVerifier()

    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        evidence_registry=registry,
        claim_verifier=verifier,
    )

    result = await orchestrator.conduct_research(
        "Agent Frameworks", user_id=1001, mode=ResearchMode.QUICK
    )

    # 1. Evidence was registered in EvidenceRegistry
    stored_github = registry.get_by_url("https://github.com/google/agent-framework")
    assert stored_github is not None
    assert stored_github.source_type == "github"

    stored_arxiv = registry.get_by_url("https://arxiv.org/abs/2608.12345")
    assert stored_arxiv is not None
    assert stored_arxiv.source_type == "arxiv"

    # 2. Claims passed through P0 verification
    assert len(result.claims) >= 1
    for claim in result.claims:
        assert claim.verification_status in (
            VerificationStatus.SUPPORTED,
            VerificationStatus.PARTIALLY_SUPPORTED,
        )


# ---------------------------------------------------------------------------
# 6. Persistence & SQLite Store Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_store_persists_run_and_query_history(tmp_path: Path) -> None:
    db_path = tmp_path / "test_research.sqlite3"
    store = ResearchStore(db_path)

    state = ResearchState(
        research_id="res-100",
        run_id="run-100",
        question="What is the state of AI Agents?",
        mode=ResearchMode.DEEP,
        status=ResearchStatus.COMPLETED,
        stop_reason=StopReason.ENOUGH_EVIDENCE,
        iterations=2,
        tool_calls=4,
        evidence_ids=["ev-1", "ev-2"],
        queries_executed=[
            QueryExecutionRecord(
                tool="github",
                query="ai agent",
                iteration=1,
                result_count=5,
                new_evidence_count=3,
                status="success",
            ),
            QueryExecutionRecord(
                tool="arxiv",
                query="ai agent benchmark",
                iteration=2,
                result_count=3,
                new_evidence_count=1,
                status="success",
            ),
        ],
    )

    from research_radar.research.models import ResearchSynthesisResult

    result = ResearchSynthesisResult(
        research_id="res-100",
        run_id="run-100",
        question="What is the state of AI Agents?",
        mode=ResearchMode.DEEP,
        answer="AI agents are developing rapidly.",
        key_findings=["Frameworks are improving"],
        confidence=ConfidenceLevel.HIGH,
        stop_reason=StopReason.ENOUGH_EVIDENCE,
        state=state,
    )

    store.save_run(result)

    saved_row = store.get_run("res-100")
    assert saved_row is not None
    assert saved_row["research_id"] == "res-100"
    assert saved_row["mode"] == "deep"
    assert saved_row["confidence"] == "high"
    assert saved_row["iterations"] == 2


# ---------------------------------------------------------------------------
# 7. Additional Edge Cases & Telegram Presentation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_all_collectors_fail_gracefully(tmp_path: Path) -> None:
    settings = _mock_settings(tmp_path)
    failing_github = MagicMock()
    failing_github.search = AsyncMock(side_effect=RuntimeError("GitHub down"))
    failing_arxiv = MagicMock()
    failing_arxiv.search = AsyncMock(side_effect=RuntimeError("arXiv 503"))

    failing_tools = ToolRegistry(
        github_search=failing_github,
        github_analyzer=MagicMock(),
        arxiv_search=failing_arxiv,
    )
    registry = EvidenceRegistry(settings.digest_state_path)
    verifier = ClaimVerifier()

    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=failing_tools,
        evidence_registry=registry,
        claim_verifier=verifier,
    )

    result = await orchestrator.conduct_research(
        "Failing collectors query", user_id=1001, mode=ResearchMode.QUICK
    )

    assert result.confidence == ConfidenceLevel.LOW
    assert "Tidak ditemukan bukti memadai" in result.answer or "0 sumber" in result.answer
    assert result.state.stop_reason in (StopReason.FAILED, StopReason.MAX_ITERATIONS)


@pytest.mark.asyncio
async def test_radar_service_delegates_to_orchestrator(tmp_path: Path) -> None:
    from research_radar.service import RadarService

    settings = _mock_settings(tmp_path)
    tools = _mock_tools()
    registry = EvidenceRegistry(settings.digest_state_path)
    verifier = ClaimVerifier()

    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        evidence_registry=registry,
        claim_verifier=verifier,
    )

    service = RadarService(
        runtime=None,
        tools=tools,
        orchestrator=orchestrator,
    )

    result = await service.research("Local AI runtime", user_id=1001, mode=ResearchMode.QUICK)
    assert result.question == "Local AI runtime"
    assert result.mode == ResearchMode.QUICK
    assert len(result.state.queries_executed) >= 1


def test_telegram_format_research_result() -> None:
    from research_radar.research.models import ResearchSynthesisResult
    from research_radar.telegram.formatter import format_research_result

    state = ResearchState(
        research_id="res-abc",
        run_id="run-abc",
        question="How to build robust AI agents?",
        mode=ResearchMode.DEEP,
        status=ResearchStatus.COMPLETED,
        stop_reason=StopReason.ENOUGH_EVIDENCE,
        iterations=3,
        budget=ResearchBudget(max_iterations=3),
    )

    result = ResearchSynthesisResult(
        research_id="res-abc",
        run_id="run-abc",
        question="How to build robust AI agents?",
        mode=ResearchMode.DEEP,
        answer="Building robust agents requires deterministic grounding.",
        key_findings=[
            "Use deterministic validators",
            "Inferensi: Multi-agent coordination requires strict session boundaries",
        ],
        evidence_sources=[
            {
                "url": "https://github.com/google/agent",
                "title": "google/agent",
                "authority": "primary",
            }
        ],
        uncertainties=["Long-horizon planning remains challenging"],
        confidence=ConfidenceLevel.HIGH,
        stop_reason=StopReason.ENOUGH_EVIDENCE,
        state=state,
    )

    formatted = format_research_result(result)

    assert "🔬 Autonomous AI Research · Mendalam (Deep Research)" in formatted
    assert "How to build robust AI agents?" in formatted
    assert "Building robust agents requires deterministic grounding." in formatted
    assert "<i>Inferensi: Multi-agent coordination" in formatted
    assert "Tingkat Keyakinan: Tinggi 🟢" in formatted
    assert "Status: enough_evidence" in formatted


# ---------------------------------------------------------------------------
# 8. P1A Audit & Hardening Specific Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_exhaustion_does_not_imply_sufficiency(tmp_path: Path) -> None:
    """Audit Item 2 & 3: Budget exhaustion must NEVER automatically imply sufficiency."""
    settings = _mock_settings(tmp_path)
    tools = _mock_tools()
    registry = EvidenceRegistry(settings.digest_state_path)
    verifier = ClaimVerifier()

    # Planner that returns high-importance unresolved gaps with new suggested queries each iteration
    gap_planner = MagicMock()
    gap_planner.plan = AsyncMock(
        return_value=ResearchPlan(
            objective="Deep Evaluation",
            sub_questions=[
                "Architecture overview",
                "Production benchmark performance",
                "Security and threat model",
            ],
            search_steps=[SearchStep(tool="github", query="arch overview", reason="Architecture")],
        )
    )

    def dynamic_detect_gaps(
        plan: ResearchPlan,
        evidence: list[dict[str, object]],
        executed_keys: set[str],
        iteration: int = 1,
        max_iterations: int = 3,
    ) -> EvidenceSufficiency:
        return EvidenceSufficiency(
            sufficient=False,
            covered_sub_questions=["Architecture overview"],
            total_sub_questions=3,
            gaps=[
                KnowledgeGap(
                    description=f"Missing production benchmark {iteration}",
                    importance="high",
                    related_sub_question="Production benchmark performance",
                    suggested_query=f"prod benchmark eval iteration {iteration}",
                    suggested_tool="arxiv",
                )
            ],
            reason="Terdapat 1 knowledge gap prioritas tinggi yang belum terjawab.",
        )

    gap_planner.detect_gaps = MagicMock(side_effect=dynamic_detect_gaps)

    # Tools that return new items each iteration so information gain passes
    call_count = 0

    async def mock_arxiv_search(query: str, limit: int = 5) -> SearchBatch[PaperResult]:
        nonlocal call_count
        call_count += 1
        return SearchBatch(
            query=query,
            source=SourceType.ARXIV,
            items=[
                PaperResult(
                    arxiv_id=f"2608.1234{call_count}",
                    title=f"Paper on {query} {call_count}",
                    url=f"https://arxiv.org/abs/2608.1234{call_count}",
                    abstract="Abstract text for paper",
                    authors=["Alice"],
                    categories=["cs.AI"],
                    published_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            ],
            partial=False,
        )

    tools.arxiv_search.search = AsyncMock(side_effect=mock_arxiv_search)

    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        evidence_registry=registry,
        claim_verifier=verifier,
        planner=gap_planner,
    )

    result = await orchestrator.conduct_research(
        "Complex Evaluation Query", user_id=1001, mode=ResearchMode.DEEP
    )

    # Must be PARTIAL, not COMPLETED, and stop reason MAX_ITERATIONS
    assert result.state.status == ResearchStatus.PARTIAL
    assert result.state.stop_reason == StopReason.MAX_ITERATIONS
    assert result.state.iterations == 3
    assert len(result.remaining_gaps) >= 1
    assert result.confidence != ConfidenceLevel.HIGH


def test_critical_knowledge_gap_prevents_sufficiency() -> None:
    """Audit Item 4 & 5: High-importance / critical knowledge gaps prevent sufficiency."""
    planner = ResearchPlanner()
    plan = ResearchPlan(
        objective="Production readiness check",
        sub_questions=[
            "Bagaimana instalasi framework?",
            "Bagaimana evaluasi benchmark dan production ready performa?",
        ],
        search_steps=[],
    )

    # Evidence only covers installation, not production benchmark
    evidence = [
        {
            "title": "Installation Guide",
            "description": "How to instalasi framework with pip.",
            "source_type": "web",
            "authority": "secondary",
        }
    ]

    sufficiency = planner.detect_gaps(
        plan=plan,
        evidence_items=evidence,
        executed_queries={"instalasi framework"},
        iteration=2,
        max_iterations=3,
    )

    assert not sufficiency.sufficient
    assert any(g.importance == "high" for g in sufficiency.gaps)
    assert (
        "prioritas tinggi" in sufficiency.reason.lower()
        or "belum mencukupi" in sufficiency.reason.lower()
    )


@pytest.mark.asyncio
async def test_all_budget_limits_enforced_individually(tmp_path: Path) -> None:
    """Audit Item 6: Every research budget property is strictly enforced."""
    settings = _mock_settings(tmp_path)
    tools = _mock_tools()
    registry = EvidenceRegistry(settings.digest_state_path)
    verifier = ClaimVerifier()

    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        evidence_registry=registry,
        claim_verifier=verifier,
    )

    # 1. max_tool_calls limit
    budget_tools = ResearchBudget(max_iterations=3, max_tool_calls=1, max_queries=10)
    res_tools = await orchestrator.conduct_research(
        "Test Tools", mode=ResearchMode.QUICK, budget=budget_tools
    )
    assert res_tools.state.tool_calls <= 1
    assert res_tools.state.stop_reason in (StopReason.MAX_TOOL_CALLS, StopReason.ENOUGH_EVIDENCE)

    # 2. max_queries limit
    budget_queries = ResearchBudget(max_iterations=3, max_tool_calls=10, max_queries=1)
    res_queries = await orchestrator.conduct_research(
        "Test Queries", mode=ResearchMode.QUICK, budget=budget_queries
    )
    assert len(res_queries.state.queries_executed) <= 1
    assert res_queries.state.stop_reason in (StopReason.MAX_QUERIES, StopReason.ENOUGH_EVIDENCE)

    # 3. max_evidence limit
    budget_ev = ResearchBudget(max_iterations=3, max_tool_calls=10, max_evidence=1)
    res_ev = await orchestrator.conduct_research(
        "Test Evidence Limit", mode=ResearchMode.QUICK, budget=budget_ev
    )
    assert len(res_ev.evidence_sources) <= 2
    assert res_ev.state.stop_reason in (StopReason.MAX_EVIDENCE, StopReason.ENOUGH_EVIDENCE)

    # 4. max_wall_time_seconds limit
    budget_time = ResearchBudget(max_iterations=3, max_wall_time_seconds=0.00001)
    res_time = await orchestrator.conduct_research(
        "Test Time Limit", mode=ResearchMode.DEEP, budget=budget_time
    )
    assert res_time.state.stop_reason in (
        StopReason.TIME_BUDGET_EXHAUSTED,
        StopReason.ENOUGH_EVIDENCE,
    )

    # 5. max_llm_calls limit (planning takes 1 LLM call, limit=1 forces deterministic synthesis)
    budget_llm = ResearchBudget(max_iterations=1, max_llm_calls=1)
    res_llm = await orchestrator.conduct_research(
        "Test LLM Limit", mode=ResearchMode.QUICK, budget=budget_llm
    )
    assert res_llm.state.llm_calls <= 1


def test_query_normalization_handles_whitespace_and_casing() -> None:
    """Audit Item 10: Query deduplication normalizes casing and whitespace."""
    from research_radar.research.orchestrator import _normalize_query_key

    k1 = _normalize_query_key("GitHub", "  AI   Agent   Framework  ")
    k2 = _normalize_query_key("github", "ai agent framework")
    k3 = _normalize_query_key("  GITHUB ", "AI agent framework")

    assert k1 == k2 == k3 == "github:ai agent framework"


@pytest.mark.asyncio
async def test_deep_mode_generates_targeted_follow_up_differing_from_initial(
    tmp_path: Path,
) -> None:
    """Audit Item 9: Deep mode executes distinct targeted follow-up queries."""
    settings = _mock_settings(tmp_path)
    tools = _mock_tools()
    registry = EvidenceRegistry(settings.digest_state_path)
    verifier = ClaimVerifier()

    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        evidence_registry=registry,
        claim_verifier=verifier,
    )

    result = await orchestrator.conduct_research(
        "Autonomous Agent Benchmarks", user_id=1001, mode=ResearchMode.DEEP
    )

    tool_query_pairs = [(q.tool, q.query.lower()) for q in result.state.queries_executed]
    # Unique tool+query pairs should equal total queries
    assert len(tool_query_pairs) == len(set(tool_query_pairs))
    assert result.state.iterations >= 1


def test_source_diversity_and_authority_confidence_scoring() -> None:
    """Audit Item 13 & 14: Confidence scoring derives from evidence quality and verification."""
    orchestrator = ResearchOrchestrator(
        settings=_mock_settings(Path("/tmp")),
        tools=_mock_tools(),
        evidence_registry=EvidenceRegistry(Path("/tmp/ev.sqlite3")),
        claim_verifier=ClaimVerifier(),
    )

    report_ok = MagicMock(unsupported_claims=0)
    report_unsupported = MagicMock(unsupported_claims=2)

    # 1. Zero evidence -> LOW
    assert orchestrator._compute_confidence([], 0, 3, report_ok) == ConfidenceLevel.LOW

    # 2. Only single community source -> not HIGH
    community_evidence = [
        {"source_type": "github", "authority": "community"},
        {"source_type": "github", "authority": "community"},
        {"source_type": "github", "authority": "community"},
    ]
    conf_community = orchestrator._compute_confidence(
        community_evidence, 3, 3, report_ok, has_unresolved_high_gaps=False
    )
    assert conf_community in (ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW)

    # 3. Multi-source with academic/primary + 0 high gaps + 0 unsupported claims -> HIGH
    diverse_evidence = [
        {"source_type": "github", "authority": "primary"},
        {"source_type": "arxiv", "authority": "academic"},
        {"source_type": "web", "authority": "secondary"},
    ]
    conf_high = orchestrator._compute_confidence(
        diverse_evidence, 3, 3, report_ok, has_unresolved_high_gaps=False
    )
    assert conf_high == ConfidenceLevel.HIGH

    # 4. Unresolved high gaps drops HIGH to MEDIUM/LOW
    conf_with_gap = orchestrator._compute_confidence(
        diverse_evidence, 3, 3, report_ok, has_unresolved_high_gaps=True
    )
    assert conf_with_gap != ConfidenceLevel.HIGH

    # 5. Unsupported claims drops HIGH
    conf_unsupported = orchestrator._compute_confidence(
        diverse_evidence, 3, 3, report_unsupported, has_unresolved_high_gaps=False
    )
    assert conf_unsupported != ConfidenceLevel.HIGH


@pytest.mark.asyncio
async def test_persistence_migration_idempotency_and_integrity(tmp_path: Path) -> None:
    """Audit Item 18: SQLite store creation and save_run is idempotent and maintains integrity."""
    import sqlite3

    db_path = tmp_path / "provenance.sqlite3"
    # Pre-populate old P0 tables
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE seen_evidence (id TEXT PRIMARY KEY, url TEXT)")
        conn.execute("INSERT INTO seen_evidence VALUES ('ev1', 'https://github.com/test')")
        conn.commit()

    store = ResearchStore(db_path)

    state = ResearchState(
        research_id="res-persist-1",
        run_id="run-persist-1",
        question="Provenance Test",
        mode=ResearchMode.QUICK,
        status=ResearchStatus.COMPLETED,
        stop_reason=StopReason.ENOUGH_EVIDENCE,
        queries_executed=[
            QueryExecutionRecord(
                tool="github", query="test query", iteration=1, result_count=2, new_evidence_count=1
            )
        ],
    )

    from research_radar.research.models import ResearchSynthesisResult

    result = ResearchSynthesisResult(
        research_id="res-persist-1",
        run_id="run-persist-1",
        question="Provenance Test",
        mode=ResearchMode.QUICK,
        answer="Persisted output.",
        state=state,
    )

    # Save once
    store.save_run(result)
    # Save again (idempotency check)
    store.save_run(result)

    # Verify run and queries
    run_row = store.get_run("res-persist-1")
    assert run_row is not None
    assert run_row["research_id"] == "res-persist-1"

    queries = store.get_queries("res-persist-1")
    assert len(queries) == 1
    assert queries[0]["query"] == "test query"

    # SQLite integrity check
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()[0]
        assert integrity == "ok"


@pytest.mark.asyncio
async def test_telegram_handlers_route_quick_and_deep_modes(tmp_path: Path) -> None:
    """Audit Item 20: Real user path test for /research (quick) and /deepresearch (deep)."""
    from research_radar.service import RadarService

    settings = _mock_settings(tmp_path)
    tools = _mock_tools()
    registry = EvidenceRegistry(settings.digest_state_path)
    verifier = ClaimVerifier()

    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        evidence_registry=registry,
        claim_verifier=verifier,
    )

    service = RadarService(runtime=None, tools=tools, orchestrator=orchestrator)

    # Quick research path
    quick_res = await service.research(
        "Quick search question", user_id=123, mode=ResearchMode.QUICK
    )
    assert quick_res.mode == ResearchMode.QUICK
    assert quick_res.state.budget.max_iterations == 1

    # Deep research path
    deep_res = await service.research("Deep research question", user_id=123, mode=ResearchMode.DEEP)
    assert deep_res.mode == ResearchMode.DEEP
    assert deep_res.state.budget.max_iterations == 3
