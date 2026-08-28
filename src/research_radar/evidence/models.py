from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class SourceAuthority(StrEnum):
    PRIMARY = "primary"
    OFFICIAL = "official"
    ACADEMIC = "academic"
    SECONDARY = "secondary"
    COMMUNITY = "community"
    UNKNOWN = "unknown"


class EvidenceStatus(StrEnum):
    NEW = "new"
    UPDATED = "updated"
    DUPLICATE = "duplicate"


class ClaimType(StrEnum):
    FACT = "fact"
    INTERPRETATION = "interpretation"


class VerificationStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONFLICTING = "conflicting"


class EvidenceRecord(BaseModel):
    evidence_id: str = Field(default_factory=lambda: uuid4().hex)
    canonical_url: str
    source_type: str
    title: str = ""
    first_seen_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_seen_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content_hash: str = ""
    published_at: datetime | None = None
    updated_at: datetime | None = None
    source_authority: SourceAuthority = SourceAuthority.UNKNOWN
    authority_score: float = Field(default=0.5, ge=0, le=1)
    metadata: dict[str, object] = Field(default_factory=dict)


class EvidenceVersion(BaseModel):
    version_id: str = Field(default_factory=lambda: uuid4().hex)
    evidence_id: str
    content_hash: str
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata_snapshot: dict[str, object] = Field(default_factory=dict)


class DeliveryRecord(BaseModel):
    delivery_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    evidence_id: str
    evidence_version: str = ""
    delivered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    delivery_status: str = "sent"


class DigestRunRecord(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    mode: str = "scheduled"
    force: bool = False
    dry_run: bool = False
    sources_attempted: int = 0
    sources_succeeded: int = 0
    sources_failed: int = 0
    items_collected: int = 0
    items_new: int = 0
    items_updated: int = 0
    items_duplicate: int = 0
    items_ranked: int = 0
    items_published: int = 0
    status: str = "running"


class StructuredClaim(BaseModel):
    text: str
    claim_type: ClaimType = ClaimType.FACT
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
    verification_status: VerificationStatus = VerificationStatus.UNSUPPORTED


class VerifiedDigestSynthesis(BaseModel):
    """Extended synthesis output with structured claims and verification."""

    model_config = ConfigDict(extra="ignore")
    overview: str = ""
    items: list
    claims: list[StructuredClaim] = Field(default_factory=list)
    verification_summary: str = ""
