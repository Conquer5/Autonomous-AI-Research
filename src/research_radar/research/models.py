"""Domain models for P1A Autonomous Research Orchestrator."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from research_radar.consensus.models import ConsensusLevel, ConsensusReport
from research_radar.evidence.models import StructuredClaim
from research_radar.retrieval import RetrievalFailure, RetrievalOperation


class ResearchMode(StrEnum):
    QUICK = "quick"
    DEEP = "deep"


class ResearchStatus(StrEnum):
    PLANNING = "planning"
    SEARCHING = "searching"
    EVALUATING = "evaluating"
    FOLLOW_UP = "follow_up"
    CONSENSUS = "consensus"
    SYNTHESIZING = "synthesizing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class StopReason(StrEnum):
    ENOUGH_EVIDENCE = "enough_evidence"
    MAX_ITERATIONS = "max_iterations"
    MAX_TOOL_CALLS = "max_tool_calls"
    MAX_QUERIES = "max_queries"
    MAX_EVIDENCE = "max_evidence"
    MAX_LLM_CALLS = "max_llm_calls"
    TIME_BUDGET_EXHAUSTED = "time_budget_exhausted"
    NO_NEW_EVIDENCE = "no_new_evidence"
    LOW_INFORMATION_GAIN = "low_information_gain"
    FAILED = "failed"


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResearchBudget(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_iterations: int = 1
    max_tool_calls: int = 5
    max_queries: int = 5
    max_evidence: int = 15
    max_wall_time_seconds: float = 60.0
    max_llm_calls: int = 3

    @classmethod
    def for_mode(cls, mode: ResearchMode) -> ResearchBudget:
        if mode == ResearchMode.DEEP:
            return cls(
                max_iterations=3,
                max_tool_calls=12,
                max_queries=10,
                max_evidence=30,
                max_wall_time_seconds=180.0,
                max_llm_calls=8,
            )
        return cls(
            max_iterations=1,
            max_tool_calls=5,
            max_queries=5,
            max_evidence=15,
            max_wall_time_seconds=60.0,
            max_llm_calls=3,
        )


class SearchStep(BaseModel):
    tool: str  # "github", "arxiv", "news", "web"
    query: str
    reason: str = ""


class ResearchPlan(BaseModel):
    objective: str
    sub_questions: list[str] = Field(default_factory=list)
    search_steps: list[SearchStep] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class KnowledgeGap(BaseModel):
    description: str
    importance: str = "medium"  # "high", "medium", "low"
    related_sub_question: str = ""
    suggested_query: str
    suggested_tool: str = "web"


class EvidenceSufficiency(BaseModel):
    sufficient: bool
    covered_sub_questions: list[str] = Field(default_factory=list)
    total_sub_questions: int = 0
    gaps: list[KnowledgeGap] = Field(default_factory=list)
    reason: str = ""


class QueryExecutionRecord(BaseModel):
    call_id: str = Field(default_factory=lambda: uuid4().hex)
    tool: str
    query: str
    iteration: int
    result_count: int = 0
    new_evidence_count: int = 0
    latency_ms: float = 0
    failures: list[RetrievalFailure] = Field(default_factory=list)
    operations: list[RetrievalOperation] = Field(default_factory=list)
    fallback: bool = False
    status: str = "success"  # "success", "partial", "failed", "rejected"


class ResearchState(BaseModel):
    research_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    question: str
    mode: ResearchMode = ResearchMode.QUICK
    objective: str = ""
    status: ResearchStatus = ResearchStatus.PLANNING
    plan: ResearchPlan | None = None
    iterations: int = 0
    budget: ResearchBudget = Field(default_factory=ResearchBudget)
    queries_executed: list[QueryExecutionRecord] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    accepted_evidence_ids: list[str] = Field(default_factory=list)
    rejected_evidence_ids: list[str] = Field(default_factory=list)
    knowledge_gaps: list[KnowledgeGap] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    tool_calls: int = 0
    llm_calls: int = 0
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    stop_reason: StopReason | None = None
    consensus_report: ConsensusReport | None = None
    coverage: dict[str, int] = Field(default_factory=dict)
    provider_failures: list[RetrievalFailure] = Field(default_factory=list)


class StructuredSynthesisItem(BaseModel):
    finding: str
    claim_type: str = "fact"  # "fact" or "interpretation"
    evidence_ids: list[str] = Field(default_factory=list)


class ResearchSynthesisResult(BaseModel):
    research_id: str
    run_id: str
    question: str
    mode: ResearchMode
    answer: str
    key_findings: list[str] = Field(default_factory=list)
    evidence_sources: list[dict[str, object]] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    remaining_gaps: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    stop_reason: StopReason = StopReason.ENOUGH_EVIDENCE
    claims: list[StructuredClaim] = Field(default_factory=list)
    consensus: ConsensusLevel = ConsensusLevel.INSUFFICIENT
    agreements: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    dissenting_findings: list[str] = Field(default_factory=list)
    safe_conclusion: str = ""
    consensus_report: ConsensusReport | None = None
    state: ResearchState
