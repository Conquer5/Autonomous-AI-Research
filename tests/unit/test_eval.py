"""Comprehensive test suite for P1C Research Quality Evaluation & Calibration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

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
    DifficultyLevel,
    EvaluationCase,
    EvaluationCategory,
    EvaluationDataset,
    EvaluationMetrics,
    FailureCategory,
    HumanRubricScore,
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
    assert dataset.schema_version == "1.0.0"
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
    assert metrics.grounding_rate == 0.0
    assert metrics.citation_coverage == 0.0
    assert metrics.evidence_count == 0


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
    assert '"schema_version": "1.0.0"' in json_payload
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
    judge = LLMJudge(llm_router=None)
    case = DEFAULT_EVALUATION_CASES[0]

    score = await judge.evaluate_output(
        case=case,
        mode="DEEP",
        answer="Dukungan MCP terbukti luas, namun terdapat perbedaan benchmark independen.",
        evidence_sources=[{"title": "Repo", "url": "https://github.com/org/repo"}],
        safe_conclusion="Gunakan dengan evaluasi batasan.",
    )

    assert score.label == "LLM_JUDGE"
    assert 0.0 <= score.relevance_score <= 10.0
    assert 0.0 <= score.grounding_score <= 10.0
    assert 0.0 <= score.conservativeness_score <= 10.0
    assert 0.0 <= score.dissent_score <= 10.0
