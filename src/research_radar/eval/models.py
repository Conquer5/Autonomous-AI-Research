"""Pydantic domain models for P1C Research Quality Evaluation & Calibration."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExecutionType(StrEnum):
    """Execution environment mode for evaluation runs."""

    SIMULATED = "SIMULATED"
    LIVE = "LIVE"


class DifficultyLevel(StrEnum):
    """Deterministic difficulty levels for evaluation cases."""

    EASY = "EASY"
    MODERATE = "MODERATE"
    HARD = "HARD"
    CONTROVERSIAL = "CONTROVERSIAL"


class EvaluationCategory(StrEnum):
    """Core evaluation domain categories."""

    AGENTS = "agents"
    LOCAL_INFERENCE = "local_inference"
    RAG_RETRIEVAL = "rag_retrieval"
    AI_ENGINEERING = "ai_engineering"
    MODELS_REASONING = "models_reasoning"
    AGENT_PROTOCOLS = "agent_protocols"
    EVALUATION_BENCHMARKS = "evaluation_benchmarks"


class FailureCategory(StrEnum):
    """Standardized failure taxonomy for research quality diagnostic."""

    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    SOURCE_QUALITY_FAILURE = "SOURCE_QUALITY_FAILURE"
    GAP_DETECTION_FAILURE = "GAP_DETECTION_FAILURE"
    CLAIM_EXTRACTION_FAILURE = "CLAIM_EXTRACTION_FAILURE"
    CLUSTERING_FAILURE = "CLUSTERING_FAILURE"
    CONTRADICTION_FAILURE = "CONTRADICTION_FAILURE"
    CONSENSUS_FAILURE = "CONSENSUS_FAILURE"
    SYNTHESIS_FAILURE = "SYNTHESIS_FAILURE"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    RATE_LIMIT = "RATE_LIMIT"
    TIME_BUDGET = "TIME_BUDGET"


class ExpectedDisagreement(BaseModel):
    """Curated gold expectation for contradiction detection benchmark."""

    model_config = ConfigDict(extra="ignore")

    expected: bool = False
    critical_topics: list[str] = Field(default_factory=list)
    notes: str = ""


class ResearchCharacteristics(BaseModel):
    """Expected research traits encoded for an evaluation case."""

    model_config = ConfigDict(extra="ignore")

    requires_multiple_sources: bool = False
    requires_recent_evidence: bool = False
    likely_disagreement: bool = False
    requires_primary_sources: bool = False
    comparative: bool = False
    benchmark_oriented: bool = False


class EvaluationCase(BaseModel):
    """Single research evaluation benchmark item."""

    model_config = ConfigDict(extra="ignore")

    id: str
    question: str
    category: EvaluationCategory
    difficulty: DifficultyLevel
    characteristics: ResearchCharacteristics
    expected_disagreement: ExpectedDisagreement | None = None
    evaluation_tags: list[str] = Field(default_factory=list)
    description: str = ""


class EvaluationDataset(BaseModel):
    """Versioned collection of research evaluation benchmark cases."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = "1.1.0"
    created_at: str
    dataset_name: str = "autonomous_ai_research_radar_eval_v1"
    cases: list[EvaluationCase] = Field(default_factory=list)


class EvaluationMetrics(BaseModel):
    """Comprehensive automatic research quality and process metrics."""

    model_config = ConfigDict(extra="ignore")

    execution_type: ExecutionType = ExecutionType.SIMULATED

    # Grounding (P0 verification based)
    supported_fact_count: int = 0
    unsupported_fact_count: int = 0
    partially_supported_count: int = 0
    grounding_rate: float | None = None
    grounding_measured: bool = False

    # Citation Coverage
    factual_claims_with_evidence: int = 0
    total_factual_claims: int = 0
    citation_coverage: float | None = None
    citations_measured: bool = False

    # Evidence
    evidence_count: int = 0
    independent_source_count: int = 0
    source_type_count: int = 0
    authoritative_source_count: int = 0

    # Consensus & Contradiction
    claim_cluster_count: int = 0
    contradiction_count: int = 0
    high_severity_contradiction_count: int = 0
    dissent_count: int = 0
    consensus_level: str = "insufficient"

    # Research Process
    iterations: int = 0
    tool_calls: int = 0
    queries: int = 0
    llm_calls: int = 0
    stop_reason: str = "unknown"
    wall_time: float = 0.0

    # Token Measurement (optional/if available from provider)
    tokens_measured: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    # Reliability
    research_status: str = "completed"
    fallback_used: bool = False
    collector_failures: int = 0

    # Failure Taxonomy
    failure_tags: list[FailureCategory] = Field(default_factory=list)


class CaseExecutionResult(BaseModel):
    """Execution and evaluation result for a single case under a specific mode."""

    model_config = ConfigDict(extra="ignore")

    execution_type: ExecutionType = ExecutionType.SIMULATED
    case_id: str
    mode: str
    metrics: EvaluationMetrics
    answer_snippet: str = ""
    safe_conclusion: str = ""
    key_findings: list[str] = Field(default_factory=list)
    dissenting_findings: list[str] = Field(default_factory=list)
    confidence: str = "low"
    evidence_sources: list[dict[str, Any]] = Field(default_factory=list)
    executed_at: str


class HumanRubricScore(BaseModel):
    """Structured human evaluation score for a research output (0=poor, 1=partial, 2=good)."""

    model_config = ConfigDict(extra="ignore")

    case_id: str
    mode: str
    relevance: int = Field(ge=0, le=2, description="Does the answer address the research question?")
    completeness: int = Field(ge=0, le=2, description="Does it address important dimensions?")
    evidence_quality: int = Field(ge=0, le=2, description="Are sources appropriate and meaningful?")
    conservativeness: int = Field(
        ge=0, le=2, description="Does it avoid claims stronger than evidence?"
    )
    usefulness: int = Field(ge=0, le=2, description="Would the answer help a technical decision?")
    dissent_handling: int = Field(ge=0, le=2, description="Are important disagreements preserved?")
    evaluator_notes: str = ""
    evaluated_at: str

    @property
    def total_score(self) -> int:
        return (
            self.relevance
            + self.completeness
            + self.evidence_quality
            + self.conservativeness
            + self.usefulness
            + self.dissent_handling
        )

    @property
    def normalized_score(self) -> float:
        return self.total_score / 12.0


class LLMJudgeScore(BaseModel):
    """Secondary LLM-as-judge score with explicit non-ground-truth labeling."""

    model_config = ConfigDict(extra="ignore")

    label: str = "LLM_JUDGE"
    judge_status: str = "COMPLETED"
    case_id: str
    mode: str
    relevance_score: float = Field(default=0.0, ge=0.0, le=10.0)
    grounding_score: float = Field(default=0.0, ge=0.0, le=10.0)
    conservativeness_score: float = Field(default=0.0, ge=0.0, le=10.0)
    dissent_score: float = Field(default=0.0, ge=0.0, le=10.0)
    reasoning: str = ""
    evaluated_at: str


class AdversarialFixtureResult(BaseModel):
    """Results from deterministic adversarial evaluation fixtures."""

    model_config = ConfigDict(extra="ignore")

    fixture_name: str
    description: str
    contradiction_detected: bool
    expected_contradiction_type: str
    actual_contradiction_type: str | None = None
    dissent_preserved: bool
    confidence_capped: bool
    safe_conclusion_avoids_universal: bool
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class AggregateComparison(BaseModel):
    """Statistical comparison between research modes across dataset."""

    model_config = ConfigDict(extra="ignore")

    execution_type: ExecutionType = ExecutionType.SIMULATED
    total_cases: int = 0
    quick_median_grounding: float | None = None
    deep_median_grounding: float | None = None
    quick_median_citations: float | None = None
    deep_median_citations: float | None = None
    quick_median_evidence: float = 0.0
    deep_median_evidence: float = 0.0
    quick_median_sources: float = 0.0
    deep_median_sources: float = 0.0
    quick_median_contradictions: float = 0.0
    deep_median_contradictions: float = 0.0
    quick_median_dissent: float = 0.0
    deep_median_dissent: float = 0.0
    quick_median_wall_time: float = 0.0
    deep_median_wall_time: float = 0.0
    quick_median_tool_calls: float = 0.0
    deep_median_tool_calls: float = 0.0
    quick_median_llm_calls: float = 0.0
    deep_median_llm_calls: float = 0.0

    # Token measurements
    tokens_measured: bool = False
    quick_median_tokens: int | None = None
    deep_median_tokens: int | None = None

    # Gold contradiction recall (only on curated gold cases)
    gold_contradiction_recall_quick: float | None = None
    gold_contradiction_recall_deep: float | None = None

    # Overall contradiction detection rate (descriptive)
    contradictions_detected_quick: int = 0
    contradictions_detected_deep: int = 0

    per_category: dict[str, dict[str, Any]] = Field(default_factory=dict)
    per_difficulty: dict[str, dict[str, Any]] = Field(default_factory=dict)
    failure_distribution: dict[str, int] = Field(default_factory=dict)
    routing_recommendations: list[str] = Field(default_factory=list)


class EvaluationRunReport(BaseModel):
    """Complete evaluation run report encompassing all metrics and artifacts."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = "1.1.0"
    execution_type: ExecutionType = ExecutionType.SIMULATED
    evaluation_id: str
    generated_at: str
    dataset_version: str
    total_cases: int
    total_runs: int = 0
    categories: list[str]
    quick_results: list[CaseExecutionResult] = Field(default_factory=list)
    deep_results: list[CaseExecutionResult] = Field(default_factory=list)
    adversarial_results: list[AdversarialFixtureResult] = Field(default_factory=list)
    comparison: AggregateComparison
    human_rubric_template: list[dict[str, Any]] = Field(default_factory=list)
