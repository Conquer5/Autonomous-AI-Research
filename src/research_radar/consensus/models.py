"""Domain models for P1B Consensus & Contradiction Intelligence."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from research_radar.evidence.models import SourceAuthority


class ClaimStance(StrEnum):
    SUPPORTS = "supports"
    OPPOSES = "opposes"
    QUALIFIES = "qualifies"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class ConsensusLevel(StrEnum):
    STRONG = "strong"
    MODERATE = "moderate"
    MIXED = "mixed"
    WEAK = "weak"
    INSUFFICIENT = "insufficient"


class ContradictionType(StrEnum):
    DIRECT_NEGATION = "direct_negation"
    NUMERIC_CONFLICT = "numeric_conflict"
    VERSION_CONFLICT = "version_conflict"
    SCOPE_CONFLICT = "scope_conflict"
    TEMPORAL_CONFLICT = "temporal_conflict"
    QUALIFICATION = "qualification"
    UNKNOWN = "unknown"


class ContradictionSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NormalizedProposition(BaseModel):
    subject: str
    predicate: str
    object_val: str | None = None
    qualifier: str | None = None
    canonical_key: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.canonical_key:
            subj = " ".join(self.subject.lower().split())
            pred = " ".join(self.predicate.lower().split())
            val = f":{' '.join(self.object_val.lower().split())}" if self.object_val else ""
            self.canonical_key = f"{subj}::{pred}{val}"


class EvidenceClaim(BaseModel):
    claim_id: str = Field(default_factory=lambda: uuid4().hex)
    evidence_id: str
    text: str
    proposition: NormalizedProposition
    stance: ClaimStance = ClaimStance.SUPPORTS
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source_authority: SourceAuthority = SourceAuthority.UNKNOWN
    source_url: str = ""
    source_type: str = ""
    published_at: datetime | None = None
    version_tag: str | None = None


class ClaimCluster(BaseModel):
    cluster_id: str = Field(default_factory=lambda: uuid4().hex)
    canonical_topic: str
    proposition: NormalizedProposition
    claims: list[EvidenceClaim] = Field(default_factory=list)


class ContradictionRecord(BaseModel):
    contradiction_id: str = Field(default_factory=lambda: uuid4().hex)
    proposition: NormalizedProposition
    claim_a: EvidenceClaim
    claim_b: EvidenceClaim
    severity: ContradictionSeverity = ContradictionSeverity.MEDIUM
    contradiction_type: ContradictionType = ContradictionType.DIRECT_NEGATION
    explanation: str
    resolved_by_version: bool = False
    resolution_notes: str = ""


class ConsensusAssessment(BaseModel):
    cluster_id: str
    proposition: NormalizedProposition
    consensus_level: ConsensusLevel
    agreement_ratio: float = 0.0
    weighted_support: float = 0.0
    weighted_opposition: float = 0.0
    weighted_qualification: float = 0.0
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    opposing_evidence_ids: list[str] = Field(default_factory=list)
    qualifying_evidence_ids: list[str] = Field(default_factory=list)
    strongest_support: str = ""
    strongest_opposition: str = ""
    unresolved_uncertainty: str = ""
    contradictions: list[ContradictionRecord] = Field(default_factory=list)


class ConsensusReport(BaseModel):
    assessments: list[ConsensusAssessment] = Field(default_factory=list)
    contradictions: list[ContradictionRecord] = Field(default_factory=list)
    overall_consensus: ConsensusLevel = ConsensusLevel.INSUFFICIENT
    agreements: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    dissenting_findings: list[str] = Field(default_factory=list)
    safe_conclusions: list[str] = Field(default_factory=list)
