"""Shared, source-independent data contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Generic, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TokenUsage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)


class LLMRequest(StrictModel):
    prompt: str = Field(min_length=1)
    system_instruction: str | None = None
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_output_tokens: int = Field(default=2048, ge=1, le=65536)
    request_id: str = Field(default_factory=lambda: uuid4().hex)


class LLMResponse(StrictModel):
    text: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(ge=0)
    request_id: str
    provider_response_id: str | None = None


DataT = TypeVar("DataT", bound=BaseModel)


class StructuredLLMResponse(StrictModel, Generic[DataT]):
    data: DataT
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(ge=0)
    request_id: str
    provider_response_id: str | None = None


class RuntimeResponse(StrictModel):
    text: str
    model: str = "hermes-agent"
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(ge=0)
    request_id: str


class SourceType(StrEnum):
    GITHUB = "github"
    ARXIV = "arxiv"
    NEWS = "news"
    WEB = "web"


class DigestItemInsight(StrictModel):
    """Evidence-bounded explanation produced by the digest synthesis model."""

    what_it_is: str = ""
    purpose: str = ""
    how_it_works: str = ""
    tech_stack: list[str] = Field(default_factory=list)
    key_takeaway: str = ""
    why_it_matters: str = ""
    caveat: str = ""
    evidence_quality: Literal["high", "medium", "low"] = "medium"
    claim_type: Literal["fact", "interpretation"] = "fact"


class DigestItemSynthesis(DigestItemInsight):
    # Gemini response_schema rejects complex field constraints and additionalProperties
    model_config = ConfigDict(extra="ignore")

    evidence_id: str = ""


class DigestSynthesis(StrictModel):
    model_config = ConfigDict(extra="ignore")

    overview: str = ""
    items: list[DigestItemSynthesis] = Field(default_factory=list)


class RepositorySignals(StrictModel):
    recency_score: float = Field(ge=0, le=1)
    activity_score: float = Field(ge=0, le=1)
    popularity_score: float = Field(ge=0, le=1)
    growth_score: float = Field(ge=0, le=1)
    relevance_score: float = Field(ge=0, le=1)
    technical_depth_score: float = Field(ge=0, le=1)


class RepositoryResult(StrictModel):
    source: SourceType = SourceType.GITHUB
    full_name: str
    description: str | None = None
    url: HttpUrl
    stars: int = Field(ge=0)
    forks: int = Field(ge=0)
    open_issues: int = Field(default=0, ge=0)
    language: str | None = None
    topics: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    pushed_at: datetime | None = None
    default_branch: str = "main"
    archived: bool = False
    license_name: str | None = None
    size_kb: int = Field(default=0, ge=0)
    signals: RepositorySignals | None = None
    insight: DigestItemInsight | None = None


class ReleaseInfo(StrictModel):
    tag_name: str
    name: str | None = None
    url: HttpUrl
    published_at: datetime | None = None
    prerelease: bool = False


class CommitInfo(StrictModel):
    sha: str
    message: str
    url: HttpUrl
    author_name: str | None = None
    committed_at: datetime | None = None
    verified: bool = False


class RepositoryAnalysis(StrictModel):
    repository: RepositoryResult
    readme: str | None = None
    readme_truncated: bool = False
    latest_release: ReleaseInfo | None = None
    recent_commits: list[CommitInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PaperResult(StrictModel):
    source: SourceType = SourceType.ARXIV
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    primary_category: str | None = None
    published_at: datetime
    updated_at: datetime
    url: HttpUrl
    pdf_url: HttpUrl | None = None
    insight: DigestItemInsight | None = None


class WebResult(StrictModel):
    source: SourceType = SourceType.WEB
    title: str
    url: HttpUrl
    description: str
    published_at: datetime | None = None
    age: str | None = None
    source_name: str | None = None
    source_quality_score: float = Field(default=0.5, ge=0, le=1)
    extra_snippets: list[str] = Field(default_factory=list)


class NewsResult(StrictModel):
    source: SourceType = SourceType.NEWS
    title: str
    url: HttpUrl
    description: str = ""
    source_name: str | None = None
    published_at: datetime | None = None
    relevance_score: float = Field(default=0.5, ge=0, le=1)
    insight: DigestItemInsight | None = None


ItemT = TypeVar("ItemT", bound=BaseModel)


class SearchBatch(StrictModel, Generic[ItemT]):
    query: str
    source: SourceType
    items: list[ItemT]
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_available: int | None = Field(default=None, ge=0)
    partial: bool = False
    warnings: list[str] = Field(default_factory=list)


class ToolHealth(StrictModel):
    name: str
    configured: bool
    healthy: bool | None = None
    detail: str | None = None


class ResearchDigest(StrictModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    topics: list[str]
    news: list[NewsResult] = Field(default_factory=list)
    repositories: list[RepositoryResult] = Field(default_factory=list)
    papers: list[PaperResult] = Field(default_factory=list)
    overview: str | None = None
    synthesis_model: str | None = None
    warnings: list[str] = Field(default_factory=list)

    def evidence_urls(self) -> list[str]:
        return [
            *(str(item.url) for item in self.news),
            *(str(item.url) for item in self.repositories),
            *(str(item.url) for item in self.papers),
        ]
