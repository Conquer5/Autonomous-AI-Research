"""Behavioral coverage for focus, Hermes planning and honest recent-evidence filtering."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from test_research_orchestrator import _mock_settings, _mock_tools

from research_radar.agent.hermes_runtime import FakeHermesRuntime
from research_radar.evidence.registry import EvidenceRegistry
from research_radar.research.models import (
    QueryExecutionRecord,
    ResearchBudget,
    ResearchMode,
    ResearchPlan,
    SearchStep,
)
from research_radar.research.orchestrator import ResearchOrchestrator
from research_radar.research.planner import ResearchPlanner
from research_radar.research.verifier import ClaimVerifier
from research_radar.research_focus import resolve_focus, skill_text
from research_radar.schemas import RuntimeResponse, SearchBatch, SourceType, WebResult
from research_radar.telegram.formatter import format_research_result


@pytest.mark.parametrize(
    ("question", "days"),
    [
        ("AI terbaru", 14),
        ("rilis hari ini", 1),
        ("model minggu ini", 7),
        ("coding agent 30 hari terakhir", 30),
        ("AI last 2 weeks", 14),
        ("AI 9999 hari terakhir", 365),
    ],
)
def test_recent_window_has_explicit_inclusive_bounds(question, days):
    today = date(2026, 9, 11)
    focus = resolve_focus(question, today=today)
    assert focus.since == today - timedelta(days=days - 1)
    assert focus.as_of == today


def test_general_and_historical_queries_keep_older_evidence():
    assert resolve_focus("Jelaskan transformer dari paper 2017").since is None
    assert resolve_focus("Bagaimana menghemat token coding agent?").skill == "agent-efficiency"
    assert resolve_focus("Gemini API gratis terbaru").skill == "model-access"


async def test_hermes_plan_uses_selected_skill_isolated_sessions_and_validates_tools():
    runtime = MagicMock()
    runtime.run = AsyncMock(
        return_value=RuntimeResponse(
            text=json.dumps(
                {
                    "objective": "Efficiency",
                    "search_steps": [
                        {"tool": "web", "query": "coding agent tokens"},
                        {"tool": "terminal", "query": "install something"},
                    ],
                }
            ),
            request_id="test",
            latency_ms=0,
        )
    )
    router = MagicMock()
    router.generate_structured = AsyncMock()
    planner = ResearchPlanner(
        llm_router=router, hermes_runtime=runtime, available_sources=frozenset({"github", "arxiv"})
    )
    a = await planner.plan("Codex hemat token terbaru", ResearchMode.DEEP)
    await planner.plan("Codex hemat token terbaru", ResearchMode.DEEP)
    assert a.planner_backend == "hermes"
    assert a.skill_id == "agent-efficiency"
    assert a.skill_version == "1.0.0"
    assert all(step.tool in {"github", "arxiv"} for step in a.search_steps)
    assert a.warnings
    router.generate_structured.assert_not_awaited()
    calls = runtime.run.await_args_list
    assert calls[0].kwargs["session_key"] != calls[1].kwargs["session_key"]
    assert "cost per successful task" in calls[0].kwargs["system_instruction"]
    assert "Return JSON only" in calls[0].args[0]


async def test_failed_hermes_does_not_spend_another_gemini_turn():
    router = MagicMock()
    router.generate_structured = AsyncMock()
    planner = ResearchPlanner(llm_router=router, hermes_runtime=FakeHermesRuntime("bad JSON"))
    plan = await planner.plan("Codex coding", ResearchMode.DEEP)
    assert plan.planner_backend == "deterministic"
    assert any("hermes" in warning for warning in plan.warnings)
    router.generate_structured.assert_not_awaited()


async def test_quick_uses_fast_gemini_with_source_sanitization():
    router = MagicMock()
    router.generate_structured = AsyncMock(
        return_value=SimpleNamespace(
            data=ResearchPlan(
                objective="test",
                search_steps=[SearchStep(tool="web", query="Gemini pricing")],
            )
        )
    )
    runtime = FakeHermesRuntime()
    planner = ResearchPlanner(
        llm_router=router, hermes_runtime=runtime, available_sources=frozenset({"news"})
    )
    plan = await planner.plan("Gemini gratis", ResearchMode.QUICK)
    assert plan.planner_backend == "gemini"
    assert plan.search_steps[0].tool == "news"
    assert router.generate_structured.await_args.args[2].value == "fast"
    assert not runtime.calls


async def test_recent_search_excludes_stale_future_and_undated_sources(tmp_path):
    settings = _mock_settings(tmp_path)
    tools = _mock_tools()
    today = datetime.now(UTC)
    tools.web_search.search.return_value = SearchBatch(
        query="model",
        source=SourceType.WEB,
        items=[
            WebResult(
                title=str(i),
                url=f"https://example.com/{i}",
                description="Model release",
                published_at=when,
            )
            for i, when in enumerate(
                [today, today - timedelta(days=30), today + timedelta(days=2), None]
            )
        ],
    )
    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        evidence_registry=EvidenceRegistry(settings.digest_state_path),
        claim_verifier=ClaimVerifier(),
    )
    focus = resolve_focus("model terbaru")
    record = QueryExecutionRecord(tool="web", query="model", iteration=1)
    evidence = {}
    new, total, status = await orchestrator._execute_search_step(
        "web", "model", evidence, record=record, focus=focus
    )
    assert (new, total, status) == (1, 1, "partial")
    assert len(record.failures) == 3
    tools.web_search.search.assert_awaited_once_with(
        "model", limit=5, start_date=focus.since, end_date=focus.as_of
    )
    assert next(iter(evidence.values()))["freshness"] == "in_window"


async def test_github_freshness_is_activity_not_repository_creation(tmp_path):
    settings = _mock_settings(tmp_path)
    tools = _mock_tools()
    repo = tools.github_search.search.return_value.items[0]
    repo.created_at = datetime(2020, 1, 1, tzinfo=UTC)
    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        evidence_registry=EvidenceRegistry(settings.digest_state_path),
        claim_verifier=ClaimVerifier(),
    )
    focus = resolve_focus("repo coding terbaru")
    evidence = {}
    await orchestrator._execute_search_step("github", "coding", evidence, focus=focus)
    assert len(evidence) == 1
    assert next(iter(evidence.values()))["date_basis"] == "repository_activity"
    tools.github_search.search.assert_awaited_once_with(
        "coding", limit=5, updated_after=focus.since
    )


async def test_zero_model_budget_keeps_focus_and_persists_provenance(tmp_path):
    settings = _mock_settings(tmp_path)
    tools = _mock_tools()
    hermes = FakeHermesRuntime()
    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        hermes_runtime=hermes,
        evidence_registry=EvidenceRegistry(settings.digest_state_path),
        claim_verifier=ClaimVerifier(),
    )
    result = await orchestrator.conduct_research(
        "Codex coding terbaru", budget=ResearchBudget(max_llm_calls=0)
    )
    assert not hermes.calls
    assert result.state.llm_calls == 0
    assert result.state.focus.since is not None
    stored = orchestrator.store.get_run(result.research_id)
    diagnostics = json.loads(stored["diagnostics"])
    assert diagnostics["skill_id"] == "agent-efficiency"
    assert diagnostics["planner_backend"] == "deterministic"
    assert "Jendela sumber:" in format_research_result(result)


def test_skill_loader_rejects_paths_outside_catalog():
    with pytest.raises(ValueError):
        skill_text("../../.env")
