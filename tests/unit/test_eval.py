"""Comprehensive test suite for P1C Research Quality Evaluation & Calibration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from research_radar.consensus.models import (
    ClaimStance,
    ConsensusLevel,
    ConsensusReport,
    ContradictionRecord,
    ContradictionSeverity,
    ContradictionType,
    EvidenceClaim,
    NormalizedProposition,
)
from research_radar.eval.adversarial import AdversarialEvaluator
from research_radar.eval.dataset import (
    DEFAULT_EVALUATION_CASES,
    load_dataset_from_file,
    load_default_dataset,
    save_dataset_to_file,
    validate_dataset,
)
from research_radar.eval.llm_judge import LLMJudge
from research_radar.eval.metrics import MetricCalculator
from research_radar.eval.models import (
    CaseExecutionResult,
    DifficultyLevel,
    EvaluationCase,
    EvaluationCategory,
    EvaluationDataset,
    EvaluationMetrics,
    ExecutionType,
    FailureCategory,
    HumanRubricScore,
    LLMJudgeScore,
    ResearchCharacteristics,
)
from research_radar.eval.report import ReportGenerator
from research_radar.eval.runner import EvaluationRunner
from research_radar.eval.storage import EvaluationStorage
from research_radar.evidence.models import (
    ClaimType,
    StructuredClaim,
    VerificationStatus,
)
from research_radar.research.models import (
    ConfidenceLevel,
    ResearchBudget,
    ResearchMode,
    ResearchState,
    ResearchStatus,
    ResearchSynthesisResult,
    StopReason,
)
from research_radar.research.verifier import VerificationReport, VerificationResult

# ---------------------------------------------------------------------------
# 1. Dataset Schema and Validation Tests
# ---------------------------------------------------------------------------


def test_default_dataset_schema_and_counts() -> None:
    dataset = load_default_dataset()
    assert dataset.schema_version == "1.1.0"
    assert len(dataset.cases) >= 25
    assert len(dataset.cases) == 32

    # Check categories coverage
    categories = {c.category for c in dataset.cases}
    assert len(categories) == 7
    assert EvaluationCategory.AGENTS in categories
    assert EvaluationCategory.LOCAL_INFERENCE in categories
    assert EvaluationCategory.RAG_RETRIEVAL in categories
    assert EvaluationCategory.AI_ENGINEERING in categories
    assert EvaluationCategory.MODELS_REASONING in categories
    assert EvaluationCategory.AGENT_PROTOCOLS in categories
    assert EvaluationCategory.EVALUATION_BENCHMARKS in categories

    # Check difficulties coverage
    difficulties = {c.difficulty for c in dataset.cases}
    assert difficulties == {
        DifficultyLevel.EASY,
        DifficultyLevel.MODERATE,
        DifficultyLevel.HARD,
        DifficultyLevel.CONTROVERSIAL,
    }


def test_dataset_validation_detects_errors() -> None:
    # Valid dataset
    valid_dataset = load_default_dataset()
    assert validate_dataset(valid_dataset) == []

    # Invalid dataset: duplicate IDs
    case1 = DEFAULT_EVALUATION_CASES[0]
    dup_case = EvaluationCase(
        id=case1.id,
        question="Another question with same id",
        category=EvaluationCategory.AGENTS,
        difficulty=DifficultyLevel.EASY,
        characteristics=ResearchCharacteristics(),
    )
    invalid_dataset = EvaluationDataset(
        created_at="2026-08-28T00:00:00Z",
        cases=[case1, dup_case],
    )
    errors = validate_dataset(invalid_dataset)
    assert any("Duplicate case id" in e for e in errors)


def test_dataset_file_io_roundtrip(tmp_path: Path) -> None:
    dataset = load_default_dataset()
    target_path = tmp_path / "cases" / "test_dataset.json"

    save_dataset_to_file(dataset, target_path)
    assert target_path.exists()

    loaded = load_dataset_from_file(target_path)
    assert loaded.schema_version == dataset.schema_version
    assert len(loaded.cases) == len(dataset.cases)
    assert loaded.cases[0].id == dataset.cases[0].id


# ---------------------------------------------------------------------------
# 2. Metric Calculation Tests
# ---------------------------------------------------------------------------


def test_metric_calculator_with_verification_report() -> None:
    state = ResearchState(
        research_id="res-1",
        run_id="run-1",
        question="Test Question",
        mode=ResearchMode.DEEP,
        status=ResearchStatus.COMPLETED,
        iterations=2,
        tool_calls=3,
        llm_calls=2,
        budget=ResearchBudget.for_mode(ResearchMode.DEEP),
        started_at=datetime(2026, 8, 28, 10, 0, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 28, 10, 0, 4, tzinfo=UTC),
    )

    claim1 = StructuredClaim(
        text="Supported Fact 1",
        claim_type=ClaimType.FACT,
        evidence_ids=["ev-1"],
        verification_status=VerificationStatus.SUPPORTED,
    )
    claim2 = StructuredClaim(
        text="Partially Supported Fact 2",
        claim_type=ClaimType.FACT,
        evidence_ids=["ev-2"],
        verification_status=VerificationStatus.PARTIALLY_SUPPORTED,
    )
    claim3 = StructuredClaim(
        text="Unsupported Fact 3",
        claim_type=ClaimType.FACT,
        evidence_ids=[],
        verification_status=VerificationStatus.UNSUPPORTED,
    )

    v_report = VerificationReport(
        results=[
            VerificationResult(claim=claim1, status=VerificationStatus.SUPPORTED),
            VerificationResult(claim=claim2, status=VerificationStatus.PARTIALLY_SUPPORTED),
            VerificationResult(claim=claim3, status=VerificationStatus.UNSUPPORTED),
        ],
        total_claims=3,
        supported_claims=1,
        partially_supported_claims=1,
        unsupported_claims=1,
        conflicting_claims=0,
        citation_coverage=2 / 3,
    )

    prop = NormalizedProposition(subject="Framework", predicate="readiness")
    claim_a = EvidenceClaim(
        evidence_id="ev-1",
        text="Claim A",
        proposition=prop,
        stance=ClaimStance.SUPPORTS,
    )
    claim_b = EvidenceClaim(
        evidence_id="ev-2",
        text="Claim B",
        proposition=prop,
        stance=ClaimStance.OPPOSES,
    )

    consensus_report = ConsensusReport(
        overall_consensus=ConsensusLevel.STRONG,
        contradictions=[
            ContradictionRecord(
                proposition=prop,
                claim_a=claim_a,
                claim_b=claim_b,
                severity=ContradictionSeverity.HIGH,
                contradiction_type=ContradictionType.DIRECT_NEGATION,
                explanation="Direct negation between claims",
            )
        ],
        dissenting_findings=["Dissent item"],
    )

    result = ResearchSynthesisResult(
        research_id="res-1",
        run_id="run-1",
        question="Test Question",
        mode=ResearchMode.DEEP,
        answer="Synthesis answer",
        key_findings=["F1", "F2", "F3"],
        evidence_sources=[
            {
                "url": "https://github.com/org/repo1",
                "source_type": "github",
                "authority": "primary",
            },
            {
                "url": "https://arxiv.org/abs/2608.1234",
                "source_type": "arxiv",
                "authority": "academic",
            },
            {"url": "https://tech-blog.com/post", "source_type": "web", "authority": "secondary"},
        ],
        confidence=ConfidenceLevel.HIGH,
        stop_reason=StopReason.ENOUGH_EVIDENCE,
        consensus=ConsensusLevel.STRONG,
        consensus_report=consensus_report,
        state=state,
    )

    metrics = MetricCalculator.calculate_metrics(result, verification_report=v_report)

    assert metrics.evidence_count == 3
    assert metrics.independent_source_count == 3
    assert metrics.source_type_count == 3
    assert metrics.authoritative_source_count == 2
    assert metrics.supported_fact_count == 1
    assert metrics.partially_supported_count == 1
    assert metrics.unsupported_fact_count == 1
    assert metrics.grounding_rate == round((1 + 0.5 * 1) / 3, 4)
    assert metrics.citation_coverage == round(2 / 3, 4)
    assert metrics.contradiction_count == 1
    assert metrics.high_severity_contradiction_count == 1
    assert metrics.dissent_count == 1
    assert metrics.wall_time == 4.0


def test_metric_calculator_zero_denominator_safe() -> None:
    state = ResearchState(
        research_id="res-0",
        run_id="run-0",
        question="Zero Claim Question",
        mode=ResearchMode.QUICK,
        status=ResearchStatus.COMPLETED,
        iterations=1,
        tool_calls=1,
        budget=ResearchBudget.for_mode(ResearchMode.QUICK),
    )
    result = ResearchSynthesisResult(
        research_id="res-0",
        run_id="run-0",
        question="Zero Claim Question",
        mode=ResearchMode.QUICK,
        answer="Direct answer without separate claims",
        key_findings=[],
        evidence_sources=[],
        confidence=ConfidenceLevel.LOW,
        stop_reason=StopReason.NO_NEW_EVIDENCE,
        state=state,
    )

    metrics = MetricCalculator.calculate_metrics(result)
    assert metrics.grounding_rate is None
    assert metrics.citation_coverage is None
    assert metrics.grounding_measured is False
    assert metrics.citations_measured is False
    assert metrics.evidence_count == 0

    # With empty verification report
    empty_report = VerificationReport(
        results=[],
        total_claims=0,
        supported_claims=0,
        partially_supported_claims=0,
        unsupported_claims=0,
        conflicting_claims=0,
        citation_coverage=0.0,
    )
    metrics_verif = MetricCalculator.calculate_metrics(result, verification_report=empty_report)
    assert metrics_verif.grounding_rate == 0.0
    assert metrics_verif.citation_coverage == 0.0
    assert metrics_verif.grounding_measured is True


def test_failure_classification() -> None:
    case = EvaluationCase(
        id="test_fail",
        question="Fail test question",
        category=EvaluationCategory.AGENTS,
        difficulty=DifficultyLevel.HARD,
        characteristics=ResearchCharacteristics(
            requires_primary_sources=True,
            likely_disagreement=True,
        ),
    )

    # Retrieval failure
    metrics_retrieval_fail = EvaluationMetrics(evidence_count=0)
    failures1 = MetricCalculator.classify_failures(metrics_retrieval_fail, case)
    assert FailureCategory.RETRIEVAL_FAILURE in failures1

    # Source quality & contradiction failure
    metrics_quality_fail = EvaluationMetrics(
        evidence_count=3,
        authoritative_source_count=0,
        contradiction_count=0,
    )
    failures2 = MetricCalculator.classify_failures(metrics_quality_fail, case)
    assert FailureCategory.SOURCE_QUALITY_FAILURE in failures2
    assert FailureCategory.CONTRADICTION_FAILURE in failures2

    # Verification failure
    metrics_verif_fail = EvaluationMetrics(
        evidence_count=3,
        unsupported_fact_count=2,
    )
    failures3 = MetricCalculator.classify_failures(metrics_verif_fail, case)
    assert FailureCategory.VERIFICATION_FAILURE in failures3


# ---------------------------------------------------------------------------
# 3. Adversarial Fixture Tests
# ---------------------------------------------------------------------------


def test_adversarial_evaluator_all_fixtures() -> None:
    evaluator = AdversarialEvaluator()
    results = evaluator.evaluate_all()

    assert len(results) == 5
    assert all(r.passed for r in results)

    # 1. Official vs Independent Benchmark
    res1 = next(r for r in results if r.fixture_name == "official_vs_independent_benchmark")
    assert res1.contradiction_detected is True
    assert res1.actual_contradiction_type == "direct_negation"
    assert res1.dissent_preserved is True
    assert res1.confidence_capped is True

    # 2. Multi-Paper Reproducibility
    res2 = next(r for r in results if r.fixture_name == "multi_paper_reproducibility")
    assert res2.contradiction_detected is True
    assert res2.dissent_preserved is True
    assert res2.details["consensus_level"] == "mixed"

    # 3. Syndicated Copycat Resilience
    res3 = next(r for r in results if r.fixture_name == "syndicated_copycat_resilience")
    assert res3.details["not_inflated_to_strong"] is True
    assert res3.dissent_preserved is True

    # 4. Valid Version Evolution
    res4 = next(r for r in results if r.fixture_name == "valid_version_evolution")
    assert res4.details["resolved_by_version"] is True
    assert res4.details["consensus_level"] in ("strong", "moderate")

    # 5. Temporal Benchmark Discrepancy (NOT auto-resolved)
    res5 = next(r for r in results if r.fixture_name == "temporal_benchmark_discrepancy")
    assert res5.details["resolved_by_version"] is False
    assert res5.details["consensus_level"] != "strong"


# ---------------------------------------------------------------------------
# 4. Evaluation Runner & Comparison Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluation_runner_executes_dataset() -> None:
    dataset = load_default_dataset()
    # Test on a compact subset of 4 cases
    subset_dataset = EvaluationDataset(
        schema_version="1.0.0",
        created_at="2026-08-28T00:00:00Z",
        cases=dataset.cases[:4],
    )

    runner = EvaluationRunner()
    report = await runner.run_evaluation(subset_dataset, modes=["QUICK", "DEEP"])

    assert report.total_cases == 4
    assert len(report.quick_results) == 4
    assert len(report.deep_results) == 4
    assert len(report.adversarial_results) == 5

    comp = report.comparison
    assert comp.total_cases == 4
    assert comp.quick_median_evidence <= comp.deep_median_evidence
    assert comp.quick_median_tool_calls <= comp.deep_median_tool_calls
    assert len(comp.routing_recommendations) >= 2


# ---------------------------------------------------------------------------
# 5. Report Generation Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_generator_markdown_and_json() -> None:
    dataset = load_default_dataset()
    subset_dataset = EvaluationDataset(
        schema_version="1.0.0",
        created_at="2026-08-28T00:00:00Z",
        cases=dataset.cases[:2],
    )
    runner = EvaluationRunner()
    report = await runner.run_evaluation(subset_dataset, modes=["QUICK", "DEEP"])

    md_text = ReportGenerator.generate_markdown(report)
    assert "# Research Radar — Evaluation & Calibration Report" in md_text
    assert "## 1. Research Mode Comparison (QUICK vs DEEP)" in md_text
    assert "## 2. Per-Category Breakdown" in md_text
    assert "## 4. Adversarial Fixture Results" in md_text
    assert "## 6. Routing & Calibration Recommendations" in md_text
    assert "## 7. Human Evaluation Rubric Guidelines" in md_text

    json_payload = ReportGenerator.generate_json(report)
    assert '"schema_version": "1.1.0"' in json_payload
    assert '"total_cases": 2' in json_payload


# ---------------------------------------------------------------------------
# 6. Evaluation Storage & Isolation Tests
# ---------------------------------------------------------------------------


def test_evaluation_storage_isolation(tmp_path: Path) -> None:
    storage = EvaluationStorage(tmp_path / "isolated_eval")
    dataset = load_default_dataset()

    # 1. Dataset
    dataset_file = storage.save_dataset(dataset, "custom_dataset.json")
    assert dataset_file.exists()
    loaded_ds = storage.load_dataset("custom_dataset.json")
    assert len(loaded_ds.cases) == len(dataset.cases)

    # 2. Human Rubric Score
    rubric_score = HumanRubricScore(
        case_id="agents_001",
        mode="DEEP",
        relevance=2,
        completeness=2,
        evidence_quality=2,
        conservativeness=2,
        usefulness=2,
        dissent_handling=2,
        evaluator_notes="Excellent grounded response.",
        evaluated_at=datetime.now(UTC).isoformat(),
    )
    assert rubric_score.total_score == 12
    assert rubric_score.normalized_score == 1.0

    rubric_file = storage.save_human_rubric([rubric_score], "human_scores.json")
    assert rubric_file.exists()
    loaded_rubric = storage.load_human_rubric("human_scores.json")
    assert len(loaded_rubric) == 1
    assert loaded_rubric[0].relevance == 2


# ---------------------------------------------------------------------------
# 7. LLM Judge Heuristic & Labeling Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_judge_label_and_heuristic() -> None:
    judge = LLMJudge(llm_router=None, allow_heuristic_fallback=True)
    case = DEFAULT_EVALUATION_CASES[0]

    score = await judge.evaluate_output(
        case=case,
        mode="DEEP",
        answer="Dukungan MCP terbukti luas, namun terdapat perbedaan benchmark independen.",
        evidence_sources=[{"title": "Repo", "url": "https://github.com/org/repo"}],
        safe_conclusion="Gunakan dengan evaluasi batasan.",
    )

    assert score is not None
    assert score.label == "LLM_JUDGE"
    assert 0.0 <= score.relevance_score <= 10.0
    assert 0.0 <= score.grounding_score <= 10.0
    assert 0.0 <= score.conservativeness_score <= 10.0
    assert 0.0 <= score.dissent_score <= 10.0


@pytest.mark.asyncio
async def test_llm_judge_not_run_when_no_router() -> None:
    judge = LLMJudge(llm_router=None, allow_heuristic_fallback=False)
    case = DEFAULT_EVALUATION_CASES[0]
    score = await judge.evaluate_output(
        case=case, mode="DEEP", answer="Answer", evidence_sources=[]
    )
    assert score is not None
    assert score.judge_status == "NOT_RUN"


@pytest.mark.asyncio
async def test_llm_judge_with_router_success_and_exception() -> None:
    case = DEFAULT_EVALUATION_CASES[0]

    # Success path
    mock_router = MagicMock()
    mock_response = MagicMock()
    mock_response.data = MagicMock(
        relevance_score=9.5,
        grounding_score=8.5,
        conservativeness_score=9.0,
        dissent_score=8.0,
        reasoning="Well supported answer",
    )
    mock_router.generate_structured = AsyncMock(return_value=mock_response)

    judge = LLMJudge(llm_router=mock_router)
    score = await judge.evaluate_output(
        case=case,
        mode="DEEP",
        answer="MCP answer",
        evidence_sources=[
            {"title": "Repo", "url": "https://github.com/org/repo", "authority": "primary"}
        ],
        safe_conclusion="Safe conclusion",
    )
    assert score is not None
    assert score.relevance_score == 9.5
    assert score.reasoning == "Well supported answer"

    # Exception path fallback
    mock_router.generate_structured = AsyncMock(side_effect=RuntimeError("LLM error"))
    judge_with_fallback = LLMJudge(llm_router=mock_router, allow_heuristic_fallback=True)
    score_fallback = await judge_with_fallback.evaluate_output(
        case=case,
        mode="DEEP",
        answer="Short",
        evidence_sources=[],
    )
    assert score_fallback is not None
    assert score_fallback.label == "LLM_JUDGE"
    assert score_fallback.relevance_score == 4.0


def test_storage_extended_methods(tmp_path: Path) -> None:
    storage = EvaluationStorage(tmp_path / "storage_test")

    # Results save / load
    res = CaseExecutionResult(
        case_id="c1",
        mode="QUICK",
        metrics=EvaluationMetrics(),
        executed_at=datetime.now(UTC).isoformat(),
    )
    storage.save_results([res], "res.json")
    loaded_res = storage.load_results("res.json")
    assert len(loaded_res) == 1
    assert loaded_res[0].case_id == "c1"

    # LLM judge save
    judge_score = LLMJudgeScore(
        case_id="c1",
        mode="QUICK",
        relevance_score=8.0,
        grounding_score=8.0,
        conservativeness_score=8.0,
        dissent_score=8.0,
        evaluated_at=datetime.now(UTC).isoformat(),
    )
    storage.save_llm_judge([judge_score], "judge.json")
    assert (storage.results_dir / "judge.json").exists()


@pytest.mark.asyncio
async def test_evaluation_runner_with_mock_orchestrator() -> None:
    mock_orchestrator = MagicMock()
    mock_synth = ResearchSynthesisResult(
        research_id="res-mock",
        run_id="run-mock",
        question="Mock Question",
        mode=ResearchMode.QUICK,
        answer="Mock answer",
        key_findings=["KF1"],
        evidence_sources=[{"url": "https://github.com/test/repo", "authority": "primary"}],
        confidence=ConfidenceLevel.HIGH,
        stop_reason=StopReason.ENOUGH_EVIDENCE,
        state=ResearchState(
            research_id="res-mock",
            run_id="run-mock",
            question="Mock Question",
            mode=ResearchMode.QUICK,
            status=ResearchStatus.COMPLETED,
            budget=ResearchBudget.for_mode(ResearchMode.QUICK),
        ),
    )
    mock_orchestrator.conduct_research = AsyncMock(return_value=mock_synth)

    runner = EvaluationRunner(orchestrator=mock_orchestrator, execution_type=ExecutionType.LIVE)
    case = DEFAULT_EVALUATION_CASES[0]
    result = await runner.run_case(case, mode="QUICK")

    assert result.case_id == case.id
    assert result.mode == "QUICK"
    assert result.metrics.evidence_count == 1


@pytest.mark.asyncio
async def test_runner_live_fails_closed_without_orchestrator() -> None:
    runner = EvaluationRunner(orchestrator=None, execution_type=ExecutionType.LIVE)
    case = DEFAULT_EVALUATION_CASES[0]
    with pytest.raises(
        ValueError, match="Live evaluation requires a configured ResearchOrchestrator"
    ):
        await runner.run_case(case, mode="QUICK")


def test_metric_calculator_unmeasured_grounding_without_verification() -> None:
    synth_res = ResearchSynthesisResult(
        research_id="res-unverif",
        run_id="run-unverif",
        question="Unverified Question",
        mode=ResearchMode.QUICK,
        answer="Direct unverified answer",
        key_findings=["Finding 1"],
        evidence_sources=[{"url": "https://github.com/org/repo", "authority": "primary"}],
        confidence=ConfidenceLevel.LOW,
        stop_reason=StopReason.ENOUGH_EVIDENCE,
        state=ResearchState(
            research_id="res-unverif",
            run_id="run-unverif",
            question="Unverified Question",
            mode=ResearchMode.QUICK,
            status=ResearchStatus.COMPLETED,
            budget=ResearchBudget.for_mode(ResearchMode.QUICK),
        ),
    )
    metrics = MetricCalculator.calculate_metrics(
        synth_res,
        verification_report=None,
        execution_type=ExecutionType.LIVE,
    )
    assert metrics.grounding_rate is None
    assert metrics.grounding_measured is False
    assert metrics.citation_coverage is None
    assert metrics.citations_measured is False


@pytest.mark.asyncio
async def test_human_review_package_generation_and_storage(tmp_path: Path) -> None:
    dataset = load_default_dataset()
    subset_dataset = EvaluationDataset(
        schema_version="1.1.0",
        created_at="2026-08-28T00:00:00Z",
        cases=dataset.cases[:2],
    )
    runner = EvaluationRunner()
    report = await runner.run_evaluation(subset_dataset, modes=["QUICK", "DEEP"])

    package = ReportGenerator.generate_human_review_package(report, subset_dataset)
    assert package["total_cases_for_review"] == 2
    for item in package["cases"]:
        assert item["mode_A"]["rubric_scores"]["relevance"] is None
        assert item["mode_B"]["rubric_scores"]["completeness"] is None

    storage = EvaluationStorage(tmp_path / "review_storage")
    saved_path = storage.save_human_review_package(package, "review_test.json")
    assert saved_path.exists()
