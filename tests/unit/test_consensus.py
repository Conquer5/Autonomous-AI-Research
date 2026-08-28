"""Comprehensive unit and adversarial test suite for P1B Consensus & Contradiction Intelligence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from research_radar.consensus.clustering import ClaimClusterer
from research_radar.consensus.contradiction import ContradictionDetector
from research_radar.consensus.engine import ConsensusEngine
from research_radar.consensus.extractor import ClaimExtractor
from research_radar.consensus.models import (
    ClaimCluster,
    ClaimStance,
    ConsensusLevel,
    ContradictionSeverity,
    ContradictionType,
    EvidenceClaim,
    NormalizedProposition,
)
from research_radar.consensus.normalizer import PropositionNormalizer
from research_radar.consensus.synthesis import ConsensusSynthesizer
from research_radar.evidence.models import SourceAuthority, VerificationStatus
from research_radar.evidence.registry import EvidenceRegistry
from research_radar.research.models import (
    ResearchMode,
)
from research_radar.research.orchestrator import ResearchOrchestrator
from research_radar.research.persistence import ResearchStore
from research_radar.research.verifier import ClaimVerifier
from research_radar.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# 1. Proposition Normalization & Stance Detection Tests
# ---------------------------------------------------------------------------


def test_proposition_normalizer_detects_supports_stance() -> None:
    normalizer = PropositionNormalizer()
    prop, stance = normalizer.normalize("Framework X is fully production ready.")
    assert stance == ClaimStance.SUPPORTS
    assert prop.predicate == "production_readiness"


def test_proposition_normalizer_detects_opposes_stance() -> None:
    normalizer = PropositionNormalizer()
    prop, stance = normalizer.normalize("Framework X is not production ready and fails under load.")
    assert stance == ClaimStance.OPPOSES
    assert prop.predicate == "production_readiness"


def test_proposition_normalizer_detects_qualifies_stance() -> None:
    normalizer = PropositionNormalizer()
    prop, stance = normalizer.normalize("Framework X is suitable only for small workloads.")
    assert stance == ClaimStance.QUALIFIES
    assert prop.predicate == "production_readiness"
    assert prop.qualifier is not None


def test_proposition_normalizer_extracts_numeric_values() -> None:
    normalizer = PropositionNormalizer()
    prop, _ = normalizer.normalize("Model requires at least 16 GB VRAM.")
    assert prop.predicate == "vram_requirement"
    assert prop.object_val == "16 gb"


# ---------------------------------------------------------------------------
# 2. Atomic Claim Extraction Tests
# ---------------------------------------------------------------------------


def test_claim_extractor_deterministic_splits_sentences() -> None:
    extractor = ClaimExtractor(llm_router=None)
    evidence_items = [
        {
            "evidence_id": "ev-1",
            "title": "Agent Framework",
            "description": (
                "Agent Framework supports Windows and Linux. "
                "It requires 8 GB VRAM for local execution."
            ),
            "source_type": "github",
            "url": "https://github.com/test/framework",
            "authority": "primary",
        }
    ]

    claims = extractor._extract_deterministic(evidence_items)
    assert len(claims) >= 2
    assert any("windows" in c.text.lower() for c in claims)
    assert any("8 gb" in c.text.lower() for c in claims)
    assert all(c.evidence_id == "ev-1" for c in claims)


def test_claim_extractor_empty_evidence_returns_empty() -> None:
    extractor = ClaimExtractor(llm_router=None)
    claims = extractor._extract_deterministic([])
    assert claims == []


# ---------------------------------------------------------------------------
# 3. Claim Clustering Tests
# ---------------------------------------------------------------------------


def test_claim_clusterer_groups_similar_propositions() -> None:
    claims = [
        EvidenceClaim(
            evidence_id="ev-1",
            text="Framework X is production ready.",
            proposition=NormalizedProposition(
                subject="Framework X", predicate="production_readiness"
            ),
            stance=ClaimStance.SUPPORTS,
            source_authority=SourceAuthority.OFFICIAL,
        ),
        EvidenceClaim(
            evidence_id="ev-2",
            text="Framework X fails in enterprise production environments.",
            proposition=NormalizedProposition(
                subject="Framework X", predicate="production_readiness"
            ),
            stance=ClaimStance.OPPOSES,
            source_authority=SourceAuthority.ACADEMIC,
        ),
        EvidenceClaim(
            evidence_id="ev-3",
            text="Framework X requires 16 GB GPU memory.",
            proposition=NormalizedProposition(
                subject="Framework X", predicate="vram_requirement", object_val="16 gb"
            ),
            stance=ClaimStance.SUPPORTS,
            source_authority=SourceAuthority.COMMUNITY,
        ),
    ]

    clusters = ClaimClusterer.cluster_claims(claims)
    assert len(clusters) == 2
    prod_cluster = next(c for c in clusters if "production_readiness" in c.canonical_topic)
    assert len(prod_cluster.claims) == 2


# ---------------------------------------------------------------------------
# 4. Contradiction Detection Tests
# ---------------------------------------------------------------------------


def test_contradiction_detector_direct_negation() -> None:
    detector = ContradictionDetector()
    cluster = ClaimCluster(
        canonical_topic="platform_support",
        proposition=NormalizedProposition(
            subject="Framework X", predicate="platform_support", object_val="windows"
        ),
        claims=[
            EvidenceClaim(
                evidence_id="ev-1",
                text="Supports Windows natively.",
                proposition=NormalizedProposition(
                    subject="Framework X", predicate="platform_support", object_val="windows"
                ),
                stance=ClaimStance.SUPPORTS,
                source_authority=SourceAuthority.PRIMARY,
            ),
            EvidenceClaim(
                evidence_id="ev-2",
                text="Windows is not supported.",
                proposition=NormalizedProposition(
                    subject="Framework X", predicate="platform_support", object_val="windows"
                ),
                stance=ClaimStance.OPPOSES,
                source_authority=SourceAuthority.SECONDARY,
            ),
        ],
    )

    contradictions = detector.detect_contradictions([cluster])
    assert len(contradictions) == 1
    c = contradictions[0]
    assert c.contradiction_type == ContradictionType.DIRECT_NEGATION
    assert c.severity == ContradictionSeverity.HIGH
    assert not c.resolved_by_version


def test_contradiction_detector_numeric_conflict() -> None:
    detector = ContradictionDetector()
    cluster = ClaimCluster(
        canonical_topic="vram_requirement",
        proposition=NormalizedProposition(subject="Model X", predicate="vram_requirement"),
        claims=[
            EvidenceClaim(
                evidence_id="ev-1",
                text="Requires 8 GB VRAM.",
                proposition=NormalizedProposition(
                    subject="Model X", predicate="vram_requirement", object_val="8 gb"
                ),
                stance=ClaimStance.SUPPORTS,
            ),
            EvidenceClaim(
                evidence_id="ev-2",
                text="Requires 16 GB VRAM.",
                proposition=NormalizedProposition(
                    subject="Model X", predicate="vram_requirement", object_val="16 gb"
                ),
                stance=ClaimStance.SUPPORTS,
            ),
        ],
    )

    contradictions = detector.detect_contradictions([cluster])
    assert len(contradictions) == 1
    assert contradictions[0].contradiction_type == ContradictionType.NUMERIC_CONFLICT
    assert not contradictions[0].resolved_by_version


def test_contradiction_detector_version_evolution_resolution() -> None:
    detector = ContradictionDetector()
    cluster = ClaimCluster(
        canonical_topic="platform_support",
        proposition=NormalizedProposition(subject="Framework X", predicate="platform_support"),
        claims=[
            EvidenceClaim(
                evidence_id="ev-1",
                text="Version 1.0 does not support Windows.",
                proposition=NormalizedProposition(
                    subject="Framework X", predicate="platform_support"
                ),
                stance=ClaimStance.OPPOSES,
                version_tag="v1.0",
                published_at=datetime(2025, 1, 1, tzinfo=UTC),
            ),
            EvidenceClaim(
                evidence_id="ev-2",
                text="Version 2.0 adds full Windows support.",
                proposition=NormalizedProposition(
                    subject="Framework X", predicate="platform_support"
                ),
                stance=ClaimStance.SUPPORTS,
                version_tag="v2.0",
                published_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        ],
    )

    contradictions = detector.detect_contradictions([cluster])
    assert len(contradictions) == 1
    c = contradictions[0]
    assert c.contradiction_type == ContradictionType.VERSION_CONFLICT
    assert c.resolved_by_version is True
    assert c.severity == ContradictionSeverity.LOW


# ---------------------------------------------------------------------------
# 5. Source Independence & Consensus Weighting Tests
# ---------------------------------------------------------------------------


def test_source_independence_penalizes_duplicated_secondary_sources() -> None:
    """3 duplicated secondary blog posts must NOT outweigh 1 official primary source."""
    engine = ConsensusEngine()
    cluster = ClaimCluster(
        canonical_topic="performance_benchmark",
        proposition=NormalizedProposition(subject="Framework X", predicate="performance_benchmark"),
        claims=[
            # 1 Official benchmark (Primary, weight 1.0)
            EvidenceClaim(
                evidence_id="ev-official",
                text="Official benchmark shows 3% difference.",
                proposition=NormalizedProposition(
                    subject="Framework X", predicate="performance_benchmark"
                ),
                stance=ClaimStance.OPPOSES,
                source_authority=SourceAuthority.PRIMARY,
                source_url="https://github.com/google/agent-benchmark",
            ),
            # 3 Copied blog posts on the same blog network (Secondary, 0.5 base)
            EvidenceClaim(
                evidence_id="ev-blog-1",
                text="Blog reports 20% speedup.",
                proposition=NormalizedProposition(
                    subject="Framework X", predicate="performance_benchmark"
                ),
                stance=ClaimStance.SUPPORTS,
                source_authority=SourceAuthority.SECONDARY,
                source_url="https://techblog.example.com/post1",
            ),
            EvidenceClaim(
                evidence_id="ev-blog-2",
                text="Blog repeats 20% speedup.",
                proposition=NormalizedProposition(
                    subject="Framework X", predicate="performance_benchmark"
                ),
                stance=ClaimStance.SUPPORTS,
                source_authority=SourceAuthority.SECONDARY,
                source_url="https://techblog.example.com/post2",
            ),
            EvidenceClaim(
                evidence_id="ev-blog-3",
                text="Blog syndicates 20% speedup.",
                proposition=NormalizedProposition(
                    subject="Framework X", predicate="performance_benchmark"
                ),
                stance=ClaimStance.SUPPORTS,
                source_authority=SourceAuthority.SECONDARY,
                source_url="https://techblog.example.com/post3",
            ),
        ],
    )

    assessment = engine._assess_single_cluster(cluster, [])
    # Diminishing weights for secondary domain repeats balance against primary source
    # Primary opposition holds substantial weight, resulting in MIXED rather than STRONG support
    assert assessment.consensus_level == ConsensusLevel.MIXED


# ---------------------------------------------------------------------------
# 6. Adversarial Evaluation Cases A – E
# ---------------------------------------------------------------------------


def test_adversarial_case_a_official_vs_independent_benchmark() -> None:
    """CASE A: Official says production ready, Independent benchmark reports failures."""
    engine = ConsensusEngine()
    detector = ContradictionDetector()

    cluster = ClaimCluster(
        canonical_topic="production_readiness",
        proposition=NormalizedProposition(subject="Framework X", predicate="production_readiness"),
        claims=[
            EvidenceClaim(
                evidence_id="ev-official",
                text="Framework X is production ready.",
                proposition=NormalizedProposition(
                    subject="Framework X", predicate="production_readiness"
                ),
                stance=ClaimStance.SUPPORTS,
                source_authority=SourceAuthority.OFFICIAL,
                source_url="https://docs.frameworkx.org",
            ),
            EvidenceClaim(
                evidence_id="ev-academic",
                text="Framework X experiences failure under sustained concurrency.",
                proposition=NormalizedProposition(
                    subject="Framework X", predicate="production_readiness"
                ),
                stance=ClaimStance.OPPOSES,
                source_authority=SourceAuthority.ACADEMIC,
                source_url="https://arxiv.org/abs/2608.9999",
            ),
        ],
    )

    contradictions = detector.detect_contradictions([cluster])
    report = engine.assess_consensus([cluster], contradictions)

    # Must NOT be STRONG consensus; dissent must be preserved
    assert report.overall_consensus != ConsensusLevel.STRONG
    assert report.overall_consensus == ConsensusLevel.MIXED
    assert len(report.dissenting_findings) >= 1
    assert "failure under sustained concurrency" in report.dissenting_findings[0].lower()


def test_adversarial_case_b_old_vs_new_version() -> None:
    """CASE B: 2025 docs say no Windows, 2026 release adds Windows."""
    engine = ConsensusEngine()
    detector = ContradictionDetector()

    cluster = ClaimCluster(
        canonical_topic="platform_support",
        proposition=NormalizedProposition(subject="Framework X", predicate="platform_support"),
        claims=[
            EvidenceClaim(
                evidence_id="ev-2025",
                text="2025 documentation: No Windows support.",
                proposition=NormalizedProposition(
                    subject="Framework X", predicate="platform_support"
                ),
                stance=ClaimStance.OPPOSES,
                source_authority=SourceAuthority.OFFICIAL,
                published_at=datetime(2025, 5, 1, tzinfo=UTC),
            ),
            EvidenceClaim(
                evidence_id="ev-2026",
                text="2026 release: Windows support added.",
                proposition=NormalizedProposition(
                    subject="Framework X", predicate="platform_support"
                ),
                stance=ClaimStance.SUPPORTS,
                source_authority=SourceAuthority.OFFICIAL,
                published_at=datetime(2026, 2, 1, tzinfo=UTC),
            ),
        ],
    )

    contradictions = detector.detect_contradictions([cluster])
    assert len(contradictions) == 1
    assert contradictions[0].resolved_by_version is True

    report = engine.assess_consensus([cluster], contradictions)
    # Does not get marked as unresolved MIXED conflict
    assert report.overall_consensus in (ConsensusLevel.STRONG, ConsensusLevel.MODERATE)


def test_adversarial_case_c_copied_news_vs_original_benchmark() -> None:
    """CASE C: 3 copied news articles (+20%) vs 1 original controlled benchmark (+3%)."""
    engine = ConsensusEngine()
    cluster = ClaimCluster(
        canonical_topic="performance_benchmark",
        proposition=NormalizedProposition(subject="Model X", predicate="performance_benchmark"),
        claims=[
            EvidenceClaim(
                evidence_id="ev-news-1",
                text="Model X beats Model Y by 20%.",
                proposition=NormalizedProposition(
                    subject="Model X", predicate="performance_benchmark", object_val="20%"
                ),
                stance=ClaimStance.SUPPORTS,
                source_authority=SourceAuthority.SECONDARY,
                source_url="https://news-aggregator.com/story1",
            ),
            EvidenceClaim(
                evidence_id="ev-news-2",
                text="Model X beats Model Y by 20%.",
                proposition=NormalizedProposition(
                    subject="Model X", predicate="performance_benchmark", object_val="20%"
                ),
                stance=ClaimStance.SUPPORTS,
                source_authority=SourceAuthority.SECONDARY,
                source_url="https://news-aggregator.com/story2",
            ),
            EvidenceClaim(
                evidence_id="ev-bench",
                text="Difference is 3% under controlled testing.",
                proposition=NormalizedProposition(
                    subject="Model X", predicate="performance_benchmark", object_val="3%"
                ),
                stance=ClaimStance.QUALIFIES,
                source_authority=SourceAuthority.ACADEMIC,
                source_url="https://arxiv.org/abs/2608.1111",
            ),
        ],
    )

    assessment = engine._assess_single_cluster(cluster, [])
    # News aggregator repeats do not overwhelm controlled benchmark
    assert assessment.consensus_level == ConsensusLevel.MIXED


def test_adversarial_case_d_academic_disagreement() -> None:
    """CASE D: Paper A (+15%), Paper B (non-reproducible), Paper C (+4%)."""
    engine = ConsensusEngine()
    detector = ContradictionDetector()

    cluster = ClaimCluster(
        canonical_topic="performance_benchmark",
        proposition=NormalizedProposition(subject="Algorithm Z", predicate="performance_benchmark"),
        claims=[
            EvidenceClaim(
                evidence_id="ev-a",
                text="Algorithm Z achieves +15% performance improvement.",
                proposition=NormalizedProposition(
                    subject="Algorithm Z", predicate="performance_benchmark", object_val="+15%"
                ),
                stance=ClaimStance.SUPPORTS,
                source_authority=SourceAuthority.ACADEMIC,
                source_url="https://arxiv.org/abs/2608.0001",
            ),
            EvidenceClaim(
                evidence_id="ev-b",
                text="Unable to reproduce +15% improvement across datasets.",
                proposition=NormalizedProposition(
                    subject="Algorithm Z", predicate="performance_benchmark"
                ),
                stance=ClaimStance.OPPOSES,
                source_authority=SourceAuthority.ACADEMIC,
                source_url="https://arxiv.org/abs/2608.0002",
            ),
            EvidenceClaim(
                evidence_id="ev-c",
                text="Algorithm Z achieves +4% improvement in limited tests.",
                proposition=NormalizedProposition(
                    subject="Algorithm Z", predicate="performance_benchmark", object_val="+4%"
                ),
                stance=ClaimStance.QUALIFIES,
                source_authority=SourceAuthority.ACADEMIC,
                source_url="https://arxiv.org/abs/2608.0003",
            ),
        ],
    )

    contradictions = detector.detect_contradictions([cluster])
    report = engine.assess_consensus([cluster], contradictions)

    assert report.overall_consensus == ConsensusLevel.MIXED
    assert len(report.dissenting_findings) >= 1


def test_adversarial_case_e_no_comparable_claims() -> None:
    """CASE E: Evidence covers unrelated topics."""
    engine = ConsensusEngine()
    cluster = ClaimCluster(
        canonical_topic="unrelated_claim",
        proposition=NormalizedProposition(subject="Solo Topic", predicate="general_claim"),
        claims=[
            EvidenceClaim(
                evidence_id="ev-solo",
                text="Solo documentation file.",
                proposition=NormalizedProposition(subject="Solo Topic", predicate="general_claim"),
                stance=ClaimStance.SUPPORTS,
                source_authority=SourceAuthority.COMMUNITY,
            )
        ],
    )

    report = engine.assess_consensus([cluster], [])
    assert report.overall_consensus == ConsensusLevel.INSUFFICIENT


# ---------------------------------------------------------------------------
# 7. Consensus Synthesis & P0 Verification Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consensus_synthesizer_produces_grounded_claims() -> None:
    synthesizer = ConsensusSynthesizer(llm_router=None)
    evidence = [
        {
            "evidence_id": "ev-10",
            "title": "Agent Benchmark",
            "description": "Agent Benchmark measures latency and token throughput.",
            "source_type": "github",
            "authority": "primary",
        }
    ]

    report = ConsensusEngine().assess_consensus([], [])
    synth_output, claims = await synthesizer.synthesize("Benchmark stats", evidence, report)

    assert len(claims) >= 1
    assert all(c.evidence_ids == ["ev-10"] for c in claims)

    # Pass through P0 ClaimVerifier
    verifier = ClaimVerifier()
    v_report = verifier.verify(
        claims,
        {"ev-10"},
        {"ev-10": "Agent Benchmark measures latency and token throughput."},
    )
    assert v_report.supported_claims >= 1
    assert v_report.unsupported_claims == 0


# ---------------------------------------------------------------------------
# 8. End-to-End Orchestration with P1B Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_runs_consensus_pipeline(tmp_path: Path) -> None:
    from research_radar.config import AppSettings
    from research_radar.schemas import PaperResult, RepositoryResult, SearchBatch, SourceType

    settings = AppSettings(
        _env_file=None,
        telegram_bot_token="123456:ABC-DEF1234567890123456789012345678",
        telegram_allowed_user_ids="12345678",
        digest_state_path=tmp_path / "radar_p1b.sqlite3",
    )

    github = MagicMock()
    github.search = AsyncMock(
        return_value=SearchBatch(
            query="test",
            source=SourceType.GITHUB,
            items=[
                RepositoryResult(
                    full_name="google/agent-runtime",
                    url="https://github.com/google/agent-runtime",
                    description="Autonomous agent runtime for high-throughput workloads.",
                    stars=1200,
                    forks=150,
                    language="Python",
                    topics=["agent", "production"],
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
                    arxiv_id="2608.5555",
                    title="Evaluation of Agent Runtime",
                    url="https://arxiv.org/abs/2608.5555",
                    abstract="Comprehensive evaluation showing agent runtime scalability.",
                    authors=["Researcher"],
                    categories=["cs.AI"],
                    published_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            ],
            partial=False,
        )
    )
    tools = ToolRegistry(
        github_search=github,
        github_analyzer=MagicMock(),
        arxiv_search=arxiv,
    )

    registry = EvidenceRegistry(settings.digest_state_path)
    verifier = ClaimVerifier()

    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        evidence_registry=registry,
        claim_verifier=verifier,
    )

    result = await orchestrator.conduct_research(
        "Is agent-runtime production ready?",
        user_id=1001,
        mode=ResearchMode.QUICK,
    )

    assert result.consensus in (
        ConsensusLevel.STRONG,
        ConsensusLevel.MODERATE,
        ConsensusLevel.MIXED,
        ConsensusLevel.INSUFFICIENT,
    )
    assert result.consensus_report is not None
    assert result.state.consensus_report is not None
    assert len(result.claims) >= 1
    assert all(c.verification_status != VerificationStatus.UNSUPPORTED for c in result.claims)

    # Check persistence in SQLite
    store = ResearchStore(settings.digest_state_path)
    saved_run = store.get_run(result.research_id)
    assert saved_run is not None
    assert saved_run["consensus_level"] is not None


# ---------------------------------------------------------------------------
# 9. LLM Extraction and Synthesis Mock Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claim_extractor_llm_mode() -> None:
    from research_radar.consensus.extractor import (
        ExtractedClaimItemSchema,
        LLMExtractedClaimBatchSchema,
    )
    from research_radar.schemas import StructuredLLMResponse

    router = MagicMock()
    mock_batch = LLMExtractedClaimBatchSchema(
        claims=[
            ExtractedClaimItemSchema(
                evidence_id="ev-llm-1",
                text="Supports Python 3.12 and 3.13 natively.",
                subject="Agent SDK",
                predicate="platform_support",
                object_val="python 3.12",
                stance="supports",
                confidence=0.95,
            )
        ]
    )
    router.generate_structured = AsyncMock(
        return_value=StructuredLLMResponse(
            request_id="req-1",
            data=mock_batch,
            model="gemini-2.5-pro",
            latency_ms=15.0,
        )
    )

    extractor = ClaimExtractor(llm_router=router)
    evidence_items = [
        {
            "evidence_id": "ev-llm-1",
            "title": "Agent SDK",
            "description": "Supports Python 3.12 and 3.13 natively.",
            "source_type": "github",
            "authority": "official",
        }
    ]

    claims = await extractor.extract_claims(evidence_items, use_llm=True)
    assert len(claims) == 1
    assert claims[0].proposition.subject == "agent_sdk"
    assert claims[0].stance == ClaimStance.SUPPORTS


@pytest.mark.asyncio
async def test_consensus_synthesizer_llm_mode() -> None:
    from research_radar.consensus.synthesis import (
        LLMConsensusSynthesisSchema,
    )
    from research_radar.schemas import StructuredLLMResponse

    router = MagicMock()
    mock_synth = LLMConsensusSynthesisSchema(
        answer="Konsensus riset menunjukkan dukungan luas untuk agen otonom.",
        key_findings=["Framework stabil pada beban tinggi."],
        agreements=["Dukungan multi-platform disepakati oleh semua rilis."],
        disagreements=["Perbedaan kecil pada konsumsi VRAM."],
        dissenting_views=["Studi independen mencatat fluktuasi latensi."],
        safe_conclusion="Aman untuk implementasi bertahap.",
        uncertainties=["Data lingkungan Windows masih terbatas."],
        claims=[
            {
                "text": "Framework stabil pada beban tinggi.",
                "claim_type": "fact",
                "evidence_ids": ["ev-synth-1"],
                "confidence": 0.85,
            }
        ],
    )
    router.generate_structured = AsyncMock(
        return_value=StructuredLLMResponse(
            request_id="req-2",
            data=mock_synth,
            model="gemini-2.5-pro",
            latency_ms=25.0,
        )
    )

    synthesizer = ConsensusSynthesizer(llm_router=router)
    report = ConsensusEngine().assess_consensus([], [])
    evidence = [
        {
            "evidence_id": "ev-synth-1",
            "title": "Doc",
            "description": "Info",
            "authority": "primary",
        }
    ]

    output, claims = await synthesizer.synthesize(
        "Evaluasi Framework", evidence, report, force_deterministic=False
    )
    assert output["answer"] == "Konsensus riset menunjukkan dukungan luas untuk agen otonom."
    assert len(output["agreements"]) == 1
    assert len(claims) == 1
    assert claims[0].evidence_ids == ["ev-synth-1"]


# ---------------------------------------------------------------------------
# 10. Telegram Formatter Consensus Integration Tests
# ---------------------------------------------------------------------------


def test_telegram_formatter_renders_consensus_sections() -> None:
    from research_radar.consensus.models import ConsensusLevel, ConsensusReport
    from research_radar.research.models import (
        ConfidenceLevel,
        ResearchBudget,
        ResearchMode,
        ResearchState,
        ResearchStatus,
        ResearchSynthesisResult,
        StopReason,
    )
    from research_radar.telegram.formatter import format_research_result

    state = ResearchState(
        research_id="res-123",
        run_id="run-123",
        question="Is Model Z enterprise ready?",
        mode=ResearchMode.DEEP,
        status=ResearchStatus.COMPLETED,
        iterations=2,
        budget=ResearchBudget.for_mode(ResearchMode.DEEP),
    )

    report = ConsensusReport(
        overall_consensus=ConsensusLevel.MIXED,
        agreements=["Dukungan Linux diverifikasi resmi."],
        disagreements=["Klaim throughput berbeda 3x lipat."],
        dissenting_findings=["Benchmark independen menunjukkan kegagalan pada 1000 RPS."],
    )

    result = ResearchSynthesisResult(
        research_id="res-123",
        run_id="run-123",
        question="Is Model Z enterprise ready?",
        mode=ResearchMode.DEEP,
        answer="Analisis menunjukkan hasil yang bervariasi.",
        key_findings=["Model Z bekerja baik pada beban moderat."],
        evidence_sources=[
            {"url": "https://github.com/test/repo", "title": "Repo", "authority": "primary"}
        ],
        uncertainties=["Beban 1000 RPS belum teruji."],
        remaining_gaps=[],
        confidence=ConfidenceLevel.MEDIUM,
        stop_reason=StopReason.ENOUGH_EVIDENCE,
        consensus=ConsensusLevel.MIXED,
        agreements=["Dukungan Linux diverifikasi resmi."],
        disagreements=["Klaim throughput berbeda 3x lipat."],
        dissenting_findings=["Benchmark independen menunjukkan kegagalan pada 1000 RPS."],
        safe_conclusion="Disarankan pengujian lokal sebelum deploy.",
        consensus_report=report,
        state=state,
    )

    text = format_research_result(result)

    assert "Status Konsensus:</b> 🟠 Bukti Beragam / Disputed" in text
    assert "<b>🤝 Kesepakatan Sumber</b>" in text
    assert "Dukungan Linux diverifikasi resmi." in text
    assert "<b>⚡ Perbedaan & Pandangan Dissent</b>" in text
    assert "Klaim throughput berbeda 3x lipat." in text
    assert "Benchmark independen menunjukkan kegagalan pada 1000 RPS." in text
    assert "<b>🛡️ Kesimpulan Konservatif</b>" in text
    assert "Disarankan pengujian lokal sebelum deploy." in text


# ---------------------------------------------------------------------------
# 11. Cluster Identity & Comparability Tests
# ---------------------------------------------------------------------------


def test_clustering_same_predicate_different_subject() -> None:
    normalizer = PropositionNormalizer()
    prop_a, stance_a = normalizer.normalize(
        "Framework A is production ready.", default_subject="Framework A"
    )
    prop_b, stance_b = normalizer.normalize(
        "Framework B is not production ready.", default_subject="Framework B"
    )

    claim_a = EvidenceClaim(
        evidence_id="ev-1",
        text="Framework A is production ready.",
        proposition=prop_a,
        stance=stance_a,
    )
    claim_b = EvidenceClaim(
        evidence_id="ev-2",
        text="Framework B is not production ready.",
        proposition=prop_b,
        stance=stance_b,
    )

    clusters = ClaimClusterer.cluster_claims([claim_a, claim_b])
    # Must produce 2 distinct clusters, not 1
    assert len(clusters) == 2
    assert {c.canonical_topic for c in clusters} == {
        "framework_a::production_readiness",
        "framework_b::production_readiness",
    }

    # Contradiction detector must not detect any contradiction across different subjects
    detector = ContradictionDetector()
    contras = detector.detect_contradictions(clusters)
    assert len(contras) == 0


def test_clustering_same_subject_different_predicate() -> None:
    normalizer = PropositionNormalizer()
    prop_a, stance_a = normalizer.normalize("vLLM is production ready.", default_subject="vLLM")
    prop_b, stance_b = normalizer.normalize("vLLM requires 16 GB VRAM.", default_subject="vLLM")

    claim_a = EvidenceClaim(
        evidence_id="ev-1", text="vLLM is production ready.", proposition=prop_a, stance=stance_a
    )
    claim_b = EvidenceClaim(
        evidence_id="ev-2", text="vLLM requires 16 GB VRAM.", proposition=prop_b, stance=stance_b
    )

    clusters = ClaimClusterer.cluster_claims([claim_a, claim_b])
    assert len(clusters) == 2
    assert {c.canonical_topic for c in clusters} == {
        "vllm::production_readiness",
        "vllm::vram_requirement",
    }


def test_clustering_same_subject_predicate_different_platform() -> None:
    normalizer = PropositionNormalizer()
    prop_win, stance_win = normalizer.normalize("vLLM supports Windows.", default_subject="vLLM")
    prop_lin, stance_lin = normalizer.normalize("vLLM supports Linux.", default_subject="vLLM")

    claim_win = EvidenceClaim(
        evidence_id="ev-1", text="vLLM supports Windows.", proposition=prop_win, stance=stance_win
    )
    claim_lin = EvidenceClaim(
        evidence_id="ev-2", text="vLLM supports Linux.", proposition=prop_lin, stance=stance_lin
    )

    clusters = ClaimClusterer.cluster_claims([claim_win, claim_lin])
    assert len(clusters) == 2
    assert {c.canonical_topic for c in clusters} == {
        "vllm::platform_support::windows",
        "vllm::platform_support::linux",
    }

    detector = ContradictionDetector()
    contras = detector.detect_contradictions(clusters)
    assert len(contras) == 0


def test_clustering_same_subject_predicate_object_opposing_stance() -> None:
    normalizer = PropositionNormalizer()
    prop_sup, stance_sup = normalizer.normalize("vLLM supports Windows.", default_subject="vLLM")
    prop_opp, stance_opp = normalizer.normalize(
        "vLLM does not support Windows.", default_subject="vLLM"
    )

    claim_sup = EvidenceClaim(
        evidence_id="ev-1", text="vLLM supports Windows.", proposition=prop_sup, stance=stance_sup
    )
    claim_opp = EvidenceClaim(
        evidence_id="ev-2",
        text="vLLM does not support Windows.",
        proposition=prop_opp,
        stance=stance_opp,
    )

    clusters = ClaimClusterer.cluster_claims([claim_sup, claim_opp])
    assert len(clusters) == 1
    assert clusters[0].canonical_topic == "vllm::platform_support::windows"

    detector = ContradictionDetector()
    contras = detector.detect_contradictions(clusters)
    assert len(contras) == 1
    assert contras[0].contradiction_type == ContradictionType.DIRECT_NEGATION
    assert contras[0].severity == ContradictionSeverity.HIGH


# ---------------------------------------------------------------------------
# 12. Subject Normalization & Aliases Tests
# ---------------------------------------------------------------------------


def test_subject_normalization_aliases() -> None:
    assert PropositionNormalizer.normalize_subject("vLLM") == "vllm"
    assert PropositionNormalizer.normalize_subject("the vLLM project") == "vllm"
    assert PropositionNormalizer.normalize_subject("vLLM runtime") == "vllm"
    assert PropositionNormalizer.normalize_subject("the vLLM framework") == "vllm"


def test_subject_normalization_distinct_entities() -> None:
    assert PropositionNormalizer.normalize_subject("Framework X") == "framework_x"
    assert PropositionNormalizer.normalize_subject("Framework Y") == "framework_y"
    assert PropositionNormalizer.normalize_subject(
        "Framework X"
    ) != PropositionNormalizer.normalize_subject("Framework Y")


# ---------------------------------------------------------------------------
# 13. Numeric Scope & Unit Equivalence Tests
# ---------------------------------------------------------------------------


def test_numeric_scope_inference_vs_training() -> None:
    normalizer = PropositionNormalizer()
    prop_a, stance_a = normalizer.normalize(
        "Requires 8 GB VRAM for inference.", default_subject="Model X"
    )
    prop_b, stance_b = normalizer.normalize(
        "Requires 16 GB VRAM for training.", default_subject="Model X"
    )

    claim_a = EvidenceClaim(
        evidence_id="ev-1",
        text="Requires 8 GB VRAM for inference.",
        proposition=prop_a,
        stance=stance_a,
    )
    claim_b = EvidenceClaim(
        evidence_id="ev-2",
        text="Requires 16 GB VRAM for training.",
        proposition=prop_b,
        stance=stance_b,
    )

    detector = ContradictionDetector()
    cluster = ClaimCluster(
        canonical_topic="model_x::vram_requirement", proposition=prop_a, claims=[claim_a, claim_b]
    )
    contras = detector.detect_contradictions([cluster])
    assert len(contras) == 0


def test_numeric_unit_conversion_mb_vs_gb() -> None:
    normalizer = PropositionNormalizer()
    prop_a, stance_a = normalizer.normalize("Memory usage is 8 GB.", default_subject="Model X")
    prop_b, stance_b = normalizer.normalize("Memory usage is 8192 MB.", default_subject="Model X")

    claim_a = EvidenceClaim(
        evidence_id="ev-1", text="Memory usage is 8 GB.", proposition=prop_a, stance=stance_a
    )
    claim_b = EvidenceClaim(
        evidence_id="ev-2", text="Memory usage is 8192 MB.", proposition=prop_b, stance=stance_b
    )

    detector = ContradictionDetector()
    cluster = ClaimCluster(
        canonical_topic="model_x::vram_requirement", proposition=prop_a, claims=[claim_a, claim_b]
    )
    contras = detector.detect_contradictions([cluster])
    assert len(contras) == 0


def test_numeric_percentage_equivalence() -> None:
    normalizer = PropositionNormalizer()
    prop_a, stance_a = normalizer.normalize(
        "Throughput improved by 20%.", default_subject="Model X"
    )
    prop_b, stance_b = normalizer.normalize(
        "Throughput improved by 0.20.", default_subject="Model X"
    )

    claim_a = EvidenceClaim(
        evidence_id="ev-1", text="Throughput improved by 20%.", proposition=prop_a, stance=stance_a
    )
    claim_b = EvidenceClaim(
        evidence_id="ev-2", text="Throughput improved by 0.20.", proposition=prop_b, stance=stance_b
    )

    detector = ContradictionDetector()
    cluster = ClaimCluster(
        canonical_topic="model_x::performance_benchmark",
        proposition=prop_a,
        claims=[claim_a, claim_b],
    )
    contras = detector.detect_contradictions([cluster])
    assert len(contras) == 0


def test_numeric_incompatible_units() -> None:
    normalizer = PropositionNormalizer()
    prop_a, stance_a = normalizer.normalize("VRAM is 8 GB.", default_subject="Model X")
    prop_b, stance_b = normalizer.normalize("Latency reduced by 20%.", default_subject="Model X")

    claim_a = EvidenceClaim(
        evidence_id="ev-1", text="VRAM is 8 GB.", proposition=prop_a, stance=stance_a
    )
    claim_b = EvidenceClaim(
        evidence_id="ev-2", text="Latency reduced by 20%.", proposition=prop_b, stance=stance_b
    )

    detector = ContradictionDetector()
    cluster = ClaimCluster(
        canonical_topic="model_x::performance", proposition=prop_a, claims=[claim_a, claim_b]
    )
    contras = detector.detect_contradictions([cluster])
    assert len(contras) == 0


# ---------------------------------------------------------------------------
# 14. Cross-Domain Copycat Syndication Resistance Test
# ---------------------------------------------------------------------------


def test_cross_domain_copycat_syndication() -> None:
    normalizer = PropositionNormalizer()
    prop_vendor, stance_vendor = normalizer.normalize(
        "Model X beats Y by 20% citing Vendor Benchmark.", default_subject="Model X"
    )
    prop_indep, stance_indep = normalizer.normalize(
        "Measured difference is only 3% under controlled testing.", default_subject="Model X"
    )

    # 3 derivative articles on 3 separate domains, all repeating the same press release narrative
    claim_a = EvidenceClaim(
        evidence_id="ev-a",
        text="Model X beats Y by 20% citing Vendor Benchmark.",
        proposition=prop_vendor,
        stance=stance_vendor,
        source_url="https://site-a.com/news/1",
        source_authority=SourceAuthority.SECONDARY,
        confidence=0.8,
    )
    claim_b = EvidenceClaim(
        evidence_id="ev-b",
        text="Model X beats Y by 20% citing Vendor Benchmark.",
        proposition=prop_vendor,
        stance=stance_vendor,
        source_url="https://site-b.com/news/2",
        source_authority=SourceAuthority.SECONDARY,
        confidence=0.8,
    )
    claim_c = EvidenceClaim(
        evidence_id="ev-c",
        text="Model X beats Y by 20% citing Vendor Benchmark.",
        proposition=prop_vendor,
        stance=stance_vendor,
        source_url="https://site-c.com/news/3",
        source_authority=SourceAuthority.SECONDARY,
        confidence=0.8,
    )

    # 1 independent academic lab finding
    claim_lab = EvidenceClaim(
        evidence_id="ev-lab",
        text="Measured difference is only 3% under controlled testing.",
        proposition=prop_indep,
        stance=stance_indep,
        source_url="https://independent-lab.org/report",
        source_authority=SourceAuthority.ACADEMIC,
        confidence=0.9,
    )

    cluster = ClaimCluster(
        canonical_topic="model_x::performance_benchmark",
        proposition=prop_vendor,
        claims=[claim_a, claim_b, claim_c, claim_lab],
    )

    detector = ContradictionDetector()
    contras = detector.detect_contradictions([cluster])
    engine = ConsensusEngine()
    report = engine.assess_consensus([cluster], contras)

    # The 3 copycat articles must not produce STRONG consensus against the independent lab
    assert report.overall_consensus != ConsensusLevel.STRONG
    assert report.overall_consensus == ConsensusLevel.MIXED
    assert len(report.dissenting_findings) >= 1


# ---------------------------------------------------------------------------
# 15. Authority ≠ Truth Test
# ---------------------------------------------------------------------------


def test_authority_does_not_determine_truth() -> None:
    normalizer = PropositionNormalizer()
    prop_sup, stance_sup = normalizer.normalize(
        "Framework is production ready.", default_subject="Framework X"
    )
    prop_opp, stance_opp = normalizer.normalize(
        "Framework fails memory leak tests and is not ready.", default_subject="Framework X"
    )

    claim_official = EvidenceClaim(
        evidence_id="ev-off",
        text="Framework is production ready.",
        proposition=prop_sup,
        stance=stance_sup,
        source_url="https://docs.framework-x.org",
        source_authority=SourceAuthority.OFFICIAL,
        confidence=0.8,
    )
    claim_acad1 = EvidenceClaim(
        evidence_id="ev-ac1",
        text="Framework fails memory leak tests and is not ready in study 1.",
        proposition=prop_opp,
        stance=stance_opp,
        source_url="https://arxiv.org/abs/2608.1111",
        source_authority=SourceAuthority.ACADEMIC,
        confidence=0.9,
    )
    claim_acad2 = EvidenceClaim(
        evidence_id="ev-ac2",
        text="Framework fails memory leak tests and is not ready in study 2.",
        proposition=prop_opp,
        stance=stance_opp,
        source_url="https://arxiv.org/abs/2608.2222",
        source_authority=SourceAuthority.ACADEMIC,
        confidence=0.9,
    )

    cluster = ClaimCluster(
        canonical_topic="framework_x::production_readiness",
        proposition=prop_sup,
        claims=[claim_official, claim_acad1, claim_acad2],
    )

    detector = ContradictionDetector()
    contras = detector.detect_contradictions([cluster])
    engine = ConsensusEngine()
    report = engine.assess_consensus([cluster], contras)

    # Official source must NOT dominate two independent academic negative studies
    assert report.overall_consensus == ConsensusLevel.MIXED
    assert len(report.dissenting_findings) >= 1


# ---------------------------------------------------------------------------
# 16. Temporal & Version Resolution Rigor Tests
# ---------------------------------------------------------------------------


def test_version_resolution_valid_release_notes() -> None:
    normalizer = PropositionNormalizer()
    prop_old, stance_old = normalizer.normalize(
        "Windows unsupported in v1.", default_subject="vLLM"
    )
    prop_new, stance_new = normalizer.normalize(
        "Windows support added in v2 release notes.", default_subject="vLLM"
    )

    claim_old = EvidenceClaim(
        evidence_id="ev-1",
        text="Windows unsupported in v1.",
        proposition=prop_old,
        stance=stance_old,
        version_tag="v1.0",
    )
    claim_new = EvidenceClaim(
        evidence_id="ev-2",
        text="Windows support added in v2 release notes.",
        proposition=prop_new,
        stance=stance_new,
        version_tag="v2.0",
    )

    detector = ContradictionDetector()
    cluster = ClaimCluster(
        canonical_topic="vllm::platform_support::windows",
        proposition=prop_old,
        claims=[claim_old, claim_new],
    )
    contras = detector.detect_contradictions([cluster])

    assert len(contras) == 1
    assert contras[0].resolved_by_version is True
    assert contras[0].severity == ContradictionSeverity.LOW


def test_version_resolution_invalid_benchmark_across_years() -> None:
    normalizer = PropositionNormalizer()
    prop_2025, stance_2025 = normalizer.normalize(
        "Measured throughput is 100 req/s in 2025.", default_subject="Framework X"
    )
    prop_2026, stance_2026 = normalizer.normalize(
        "Independent measured throughput is 55 req/s in 2026.", default_subject="Framework X"
    )

    claim_2025 = EvidenceClaim(
        evidence_id="ev-1",
        text="Measured throughput is 100 req/s in 2025.",
        proposition=prop_2025,
        stance=stance_2025,
        published_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    claim_2026 = EvidenceClaim(
        evidence_id="ev-2",
        text="Independent measured throughput is 55 req/s in 2026.",
        proposition=prop_2026,
        stance=stance_2026,
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    detector = ContradictionDetector()
    cluster = ClaimCluster(
        canonical_topic="framework_x::performance_benchmark",
        proposition=prop_2025,
        claims=[claim_2025, claim_2026],
    )
    contras = detector.detect_contradictions([cluster])

    assert len(contras) == 1
    # Raw benchmark discrepancy across years without changelog must not be auto-resolved
    assert contras[0].resolved_by_version is False
    assert contras[0].severity == ContradictionSeverity.MEDIUM


def test_version_same_version_unresolved_conflict() -> None:
    normalizer = PropositionNormalizer()
    prop_doc, stance_doc = normalizer.normalize(
        "v2 docs declare framework production ready.", default_subject="Framework X"
    )
    prop_bench, stance_bench = normalizer.normalize(
        "v2 independent benchmark fails under sustained concurrency.", default_subject="Framework X"
    )

    claim_doc = EvidenceClaim(
        evidence_id="ev-1",
        text="v2 docs declare framework production ready.",
        proposition=prop_doc,
        stance=stance_doc,
        version_tag="v2.0",
    )
    claim_bench = EvidenceClaim(
        evidence_id="ev-2",
        text="v2 independent benchmark fails under sustained concurrency.",
        proposition=prop_bench,
        stance=stance_bench,
        version_tag="v2.0",
    )

    detector = ContradictionDetector()
    cluster = ClaimCluster(
        canonical_topic="framework_x::production_readiness",
        proposition=prop_doc,
        claims=[claim_doc, claim_bench],
    )
    contras = detector.detect_contradictions([cluster])

    assert len(contras) == 1
    assert contras[0].resolved_by_version is False
    assert contras[0].severity == ContradictionSeverity.HIGH


# ---------------------------------------------------------------------------
# 17. Qualification vs Direct Negation Tests
# ---------------------------------------------------------------------------


def test_qualification_vs_direct_negation() -> None:
    detector = ContradictionDetector()
    normalizer = PropositionNormalizer()

    # Case 1: Qualification
    prop_full, s_full = normalizer.normalize(
        "Framework X is production ready.", default_subject="Framework X"
    )
    prop_qual, s_qual = normalizer.normalize(
        "Framework X is production ready for small workloads only.", default_subject="Framework X"
    )
    c_full = EvidenceClaim(
        evidence_id="ev-1",
        text="Framework X is production ready.",
        proposition=prop_full,
        stance=s_full,
    )
    c_qual = EvidenceClaim(
        evidence_id="ev-2",
        text="Framework X is production ready for small workloads only.",
        proposition=prop_qual,
        stance=s_qual,
    )
    cluster_qual = ClaimCluster(
        canonical_topic="framework_x::production_readiness",
        proposition=prop_full,
        claims=[c_full, c_qual],
    )
    contras_qual = detector.detect_contradictions([cluster_qual])
    assert len(contras_qual) == 1
    assert contras_qual[0].contradiction_type == ContradictionType.QUALIFICATION

    # Case 2: Direct Negation
    prop_neg, s_neg = normalizer.normalize(
        "Framework X is not production ready.", default_subject="Framework X"
    )
    c_neg = EvidenceClaim(
        evidence_id="ev-3",
        text="Framework X is not production ready.",
        proposition=prop_neg,
        stance=s_neg,
    )
    cluster_neg = ClaimCluster(
        canonical_topic="framework_x::production_readiness",
        proposition=prop_full,
        claims=[c_full, c_neg],
    )
    contras_neg = detector.detect_contradictions([cluster_neg])
    assert len(contras_neg) == 1
    assert contras_neg[0].contradiction_type == ContradictionType.DIRECT_NEGATION


# ---------------------------------------------------------------------------
# 18. Consensus Threshold Boundaries and Math Tests
# ---------------------------------------------------------------------------


def test_consensus_threshold_boundaries() -> None:
    engine = ConsensusEngine()
    prop = NormalizedProposition(subject="Framework X", predicate="production_readiness")

    # Ratio 0.80 -> STRONG
    claim1 = EvidenceClaim(
        evidence_id="ev-1",
        text="Ready",
        proposition=prop,
        stance=ClaimStance.SUPPORTS,
        source_url="https://a.com",
        source_authority=SourceAuthority.PRIMARY,
        confidence=1.0,
    )
    claim2 = EvidenceClaim(
        evidence_id="ev-2",
        text="Ready",
        proposition=prop,
        stance=ClaimStance.SUPPORTS,
        source_url="https://b.com",
        source_authority=SourceAuthority.PRIMARY,
        confidence=1.0,
    )
    cluster_strong = ClaimCluster(canonical_topic="t", proposition=prop, claims=[claim1, claim2])
    assessment_strong = engine._assess_single_cluster(cluster_strong, [])
    assert assessment_strong.consensus_level == ConsensusLevel.STRONG

    # Ratio < 0.80 with no conflict -> MODERATE
    claim_supp = EvidenceClaim(
        evidence_id="ev-1",
        text="Ready",
        proposition=prop,
        stance=ClaimStance.SUPPORTS,
        source_url="https://a.com",
        source_authority=SourceAuthority.PRIMARY,
        confidence=0.7,
    )
    claim_qual = EvidenceClaim(
        evidence_id="ev-2",
        text="Beta only",
        proposition=prop,
        stance=ClaimStance.QUALIFIES,
        source_url="https://b.com",
        source_authority=SourceAuthority.SECONDARY,
        confidence=0.7,
    )
    cluster_mod = ClaimCluster(
        canonical_topic="t", proposition=prop, claims=[claim_supp, claim_qual]
    )
    assessment_mod = engine._assess_single_cluster(cluster_mod, [])
    assert assessment_mod.consensus_level in (ConsensusLevel.MODERATE, ConsensusLevel.MIXED)


def test_consensus_denominator_with_neutral() -> None:
    engine = ConsensusEngine()
    prop = NormalizedProposition(subject="Framework X", predicate="production_readiness")

    claim_supp = EvidenceClaim(
        evidence_id="ev-1",
        text="Ready",
        proposition=prop,
        stance=ClaimStance.SUPPORTS,
        source_url="https://a.com",
        source_authority=SourceAuthority.PRIMARY,
        confidence=0.8,
    )
    claim_neut1 = EvidenceClaim(
        evidence_id="ev-2",
        text="Could it be ready?",
        proposition=prop,
        stance=ClaimStance.NEUTRAL,
        source_url="https://b.com",
        source_authority=SourceAuthority.COMMUNITY,
        confidence=0.8,
    )
    claim_neut2 = EvidenceClaim(
        evidence_id="ev-3",
        text="Is it ready?",
        proposition=prop,
        stance=ClaimStance.NEUTRAL,
        source_url="https://c.com",
        source_authority=SourceAuthority.COMMUNITY,
        confidence=0.8,
    )

    cluster = ClaimCluster(
        canonical_topic="t", proposition=prop, claims=[claim_supp, claim_neut1, claim_neut2]
    )
    assessment = engine._assess_single_cluster(cluster, [])
    # Neutral claims must not inflate agreement ratio to 1.0!
    assert assessment.agreement_ratio < 1.0


def test_insufficient_vs_weak_distinction() -> None:
    engine = ConsensusEngine()
    prop = NormalizedProposition(subject="Framework X", predicate="production_readiness")

    # 1 claim -> INSUFFICIENT
    claim_single = EvidenceClaim(
        evidence_id="ev-1",
        text="Ready",
        proposition=prop,
        stance=ClaimStance.SUPPORTS,
        source_url="https://a.com",
        source_authority=SourceAuthority.COMMUNITY,
        confidence=0.5,
    )
    cluster_insuff = ClaimCluster(canonical_topic="t", proposition=prop, claims=[claim_single])
    assert (
        engine._assess_single_cluster(cluster_insuff, []).consensus_level
        == ConsensusLevel.INSUFFICIENT
    )

    # 3 claims but weak agreement -> WEAK / MIXED
    c1 = EvidenceClaim(
        evidence_id="ev-1",
        text="Ready",
        proposition=prop,
        stance=ClaimStance.SUPPORTS,
        source_url="https://a.com",
        source_authority=SourceAuthority.COMMUNITY,
        confidence=0.6,
    )
    c2 = EvidenceClaim(
        evidence_id="ev-2",
        text="Not ready",
        proposition=prop,
        stance=ClaimStance.OPPOSES,
        source_url="https://b.com",
        source_authority=SourceAuthority.COMMUNITY,
        confidence=0.6,
    )
    c3 = EvidenceClaim(
        evidence_id="ev-3",
        text="Beta only",
        proposition=prop,
        stance=ClaimStance.QUALIFIES,
        source_url="https://c.com",
        source_authority=SourceAuthority.COMMUNITY,
        confidence=0.6,
    )
    cluster_weak = ClaimCluster(canonical_topic="t", proposition=prop, claims=[c1, c2, c3])
    level = engine._assess_single_cluster(cluster_weak, []).consensus_level
    assert level in (ConsensusLevel.MIXED, ConsensusLevel.WEAK)


# ---------------------------------------------------------------------------
# 19. Dissent Preservation in Synthesis Test
# ---------------------------------------------------------------------------


def test_dissent_preservation_minority_opposition() -> None:
    normalizer = PropositionNormalizer()
    prop_sup, s_sup = normalizer.normalize("vLLM is production ready.", default_subject="vLLM")
    prop_opp, s_opp = normalizer.normalize(
        "vLLM is not production ready and suffers memory fragmentation.", default_subject="vLLM"
    )

    claims_supp = [
        EvidenceClaim(
            evidence_id=f"ev-{i}",
            text=f"vLLM is production ready in setup {i}.",
            proposition=prop_sup,
            stance=s_sup,
            source_url=f"https://blog{i}.com",
            source_authority=SourceAuthority.SECONDARY,
            confidence=0.7,
        )
        for i in range(1, 4)
    ]
    claim_dissent = EvidenceClaim(
        evidence_id="ev-dissent",
        text="vLLM is not production ready and suffers memory fragmentation.",
        proposition=prop_opp,
        stance=s_opp,
        source_url="https://academic-eval.edu/paper",
        source_authority=SourceAuthority.ACADEMIC,
        confidence=0.9,
    )

    cluster = ClaimCluster(
        canonical_topic="vllm::production_readiness",
        proposition=prop_sup,
        claims=[*claims_supp, claim_dissent],
    )
    detector = ContradictionDetector()
    contras = detector.detect_contradictions([cluster])
    engine = ConsensusEngine()
    report = engine.assess_consensus([cluster], contras)

    # Dissent must be preserved
    assert len(report.dissenting_findings) >= 1
    assert any("memory fragmentation" in d for d in report.dissenting_findings)


# ---------------------------------------------------------------------------
# 20. P0 Verification on Consensus Output & Unsupported Claims
# ---------------------------------------------------------------------------


def test_p0_verifier_catches_unsupported_consensus_synthesis() -> None:
    verifier = ClaimVerifier()
    # Synthesis outputs a factual claim with non-existent evidence_id
    from research_radar.evidence.models import ClaimType, StructuredClaim

    fake_claim = StructuredClaim(
        text="Framework X is verified on 1000 GPUs.",
        claim_type=ClaimType.FACT,
        evidence_ids=["ev-non-existent"],
        confidence=0.9,
    )
    valid_claim = StructuredClaim(
        text="Framework X is an inference engine.",
        claim_type=ClaimType.FACT,
        evidence_ids=["ev-valid-1"],
        confidence=0.8,
    )

    context_ids = {"ev-valid-1"}
    report = verifier.verify(
        [fake_claim, valid_claim],
        context_ids,
        {"ev-valid-1": "Framework X is an inference engine."},
    )

    assert report.unsupported_claims == 1
    downgraded = verifier.downgrade_unsupported_claims([fake_claim, valid_claim], report)
    assert downgraded[0].claim_type == ClaimType.INTERPRETATION
    assert downgraded[0].confidence <= 0.5
    assert downgraded[1].claim_type == ClaimType.FACT


# ---------------------------------------------------------------------------
# 21. Deterministic Extraction Filters Non-Factual Language
# ---------------------------------------------------------------------------


def test_deterministic_extraction_filters_questions_and_aspirations() -> None:
    normalizer = PropositionNormalizer()
    _, stance_q = normalizer.normalize(
        "Could this framework become production ready in the future?"
    )
    assert stance_q == ClaimStance.NEUTRAL

    _, stance_asp = normalizer.normalize("We hope to support Windows soon.")
    assert stance_asp == ClaimStance.QUALIFIES


# ---------------------------------------------------------------------------
# 22. Contradiction Detector Edge Cases
# ---------------------------------------------------------------------------


def test_contradiction_missing_timestamps_and_versions() -> None:
    prop = NormalizedProposition(subject="Framework X", predicate="production_readiness")
    c1 = EvidenceClaim(
        evidence_id="ev-1",
        text="Framework X is production ready.",
        proposition=prop,
        stance=ClaimStance.SUPPORTS,
    )
    c2 = EvidenceClaim(
        evidence_id="ev-2",
        text="Framework X is not production ready.",
        proposition=prop,
        stance=ClaimStance.OPPOSES,
    )

    detector = ContradictionDetector()
    cluster = ClaimCluster(
        canonical_topic="framework_x::production_readiness", proposition=prop, claims=[c1, c2]
    )
    contras = detector.detect_contradictions([cluster])

    assert len(contras) == 1
    assert contras[0].resolved_by_version is False
    assert contras[0].severity == ContradictionSeverity.HIGH


def test_contradiction_unknown_and_neutral_stances_no_conflict() -> None:
    prop = NormalizedProposition(subject="Framework X", predicate="production_readiness")
    c1 = EvidenceClaim(
        evidence_id="ev-1",
        text="Framework X is production ready.",
        proposition=prop,
        stance=ClaimStance.SUPPORTS,
    )
    c2 = EvidenceClaim(
        evidence_id="ev-2",
        text="Could Framework X be ready?",
        proposition=prop,
        stance=ClaimStance.NEUTRAL,
    )
    c3 = EvidenceClaim(
        evidence_id="ev-3",
        text="Framework X info.",
        proposition=prop,
        stance=ClaimStance.UNKNOWN,
    )

    detector = ContradictionDetector()
    cluster = ClaimCluster(
        canonical_topic="framework_x::production_readiness", proposition=prop, claims=[c1, c2, c3]
    )
    contras = detector.detect_contradictions([cluster])
    assert len(contras) == 0


# ---------------------------------------------------------------------------
# 23. Graceful Degradation Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_graceful_degradation_on_consensus_engine_failure(
    tmp_path: Path,
) -> None:
    from research_radar.config import AppSettings
    from research_radar.research.models import ResearchMode
    from research_radar.schemas import RepositoryResult, SearchBatch, SourceType

    settings = AppSettings(
        _env_file=None,
        telegram_bot_token="123456:ABC-DEF1234567890123456789012345678",
        telegram_allowed_user_ids="12345678",
        digest_state_path=tmp_path / "degradation.sqlite3",
    )
    github = MagicMock()
    github.search = AsyncMock(
        return_value=SearchBatch(
            query="test",
            source=SourceType.GITHUB,
            items=[
                RepositoryResult(
                    full_name="org/agent-framework",
                    url="https://github.com/org/agent-framework",
                    description="Autonomous agent framework.",
                    stars=1200,
                    forks=150,
                    language="Python",
                    topics=["agent"],
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            ],
            partial=False,
        )
    )
    tools = ToolRegistry(
        github_search=github,
        github_analyzer=MagicMock(),
        arxiv_search=MagicMock(),
    )
    registry = EvidenceRegistry(settings.digest_state_path)
    verifier = ClaimVerifier()

    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        evidence_registry=registry,
        claim_verifier=verifier,
    )

    # Intentionally mock consensus engine to raise an exception
    orchestrator.consensus_engine.assess_consensus = MagicMock(
        side_effect=RuntimeError("Simulated engine crash")
    )

    result = await orchestrator.conduct_research(
        "Is agent-framework reliable?",
        user_id=1001,
        mode=ResearchMode.QUICK,
    )

    assert result is not None
    assert result.answer != ""
    assert result.state.stop_reason is not None


# ---------------------------------------------------------------------------
# 24. Persistence Migration & SQLite Integrity Test
# ---------------------------------------------------------------------------


def test_persistence_migration_integrity_and_idempotency(tmp_path: Path) -> None:
    import sqlite3

    from research_radar.consensus.models import ConsensusLevel, ConsensusReport
    from research_radar.research.models import (
        ConfidenceLevel,
        ResearchBudget,
        ResearchMode,
        ResearchState,
        ResearchStatus,
        ResearchSynthesisResult,
        StopReason,
    )

    db_path = tmp_path / "legacy_research.db"

    # 1. Create a legacy table without consensus_level and contradictions_count
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE research_runs (
                research_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                question TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                stop_reason TEXT,
                iterations INTEGER NOT NULL,
                tool_calls INTEGER NOT NULL,
                evidence_count INTEGER NOT NULL,
                confidence TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                summary TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO research_runs (
                research_id, run_id, question, mode, status,
                stop_reason, iterations, tool_calls, evidence_count,
                confidence, started_at, completed_at, summary
            ) VALUES (
                'res-old-1', 'run-old-1', 'Old Question', 'quick', 'completed',
                'enough_evidence', 1, 1, 2, 'high',
                '2026-08-01T00:00:00Z', '2026-08-01T00:01:00Z', 'Old summary'
            )
            """
        )

    # 2. Initialize ResearchStore (triggers migration)
    store = ResearchStore(db_path)

    # Check integrity
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        assert cursor.fetchone()[0] == "ok"

    # Check legacy row was preserved
    old_run = store.get_run("res-old-1")
    assert old_run is not None
    assert old_run["question"] == "Old Question"
    assert old_run["consensus_level"] is None

    # Check saving new run with consensus columns
    state = ResearchState(
        research_id="res-new-1",
        run_id="run-new-1",
        question="New Question",
        mode=ResearchMode.DEEP,
        status=ResearchStatus.COMPLETED,
        stop_reason=StopReason.ENOUGH_EVIDENCE,
        iterations=2,
        tool_calls=3,
        budget=ResearchBudget.for_mode(ResearchMode.DEEP),
        completed_at=datetime.now(UTC),
    )
    result = ResearchSynthesisResult(
        research_id="res-new-1",
        run_id="run-new-1",
        question="New Question",
        mode=ResearchMode.DEEP,
        answer="Synthesis answer",
        key_findings=["Finding 1"],
        evidence_sources=[{"url": "https://a.com", "title": "A"}],
        confidence=ConfidenceLevel.HIGH,
        stop_reason=StopReason.ENOUGH_EVIDENCE,
        consensus=ConsensusLevel.STRONG,
        consensus_report=ConsensusReport(overall_consensus=ConsensusLevel.STRONG),
        state=state,
    )
    store.save_run(result)

    new_run = store.get_run("res-new-1")
    assert new_run is not None
    assert new_run["consensus_level"] == "strong"
    assert new_run["contradictions_count"] == 0

    # 3. Re-initialize ResearchStore (idempotency check)
    store2 = ResearchStore(db_path)
    new_run2 = store2.get_run("res-new-1")
    assert new_run2 is not None
    assert new_run2["consensus_level"] == "strong"
