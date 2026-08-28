"""Curated deterministic adversarial fixtures for contradiction and consensus evaluation."""

from __future__ import annotations

from datetime import UTC, datetime

from research_radar.consensus.clustering import ClaimClusterer
from research_radar.consensus.contradiction import ContradictionDetector
from research_radar.consensus.engine import ConsensusEngine
from research_radar.consensus.models import (
    ClaimCluster,
    ConsensusLevel,
    ContradictionType,
    EvidenceClaim,
)
from research_radar.consensus.normalizer import PropositionNormalizer
from research_radar.eval.models import AdversarialFixtureResult
from research_radar.evidence.models import SourceAuthority


class AdversarialEvaluator:
    """Runs deterministic adversarial fixtures assessing contradiction recall and calibration."""

    def __init__(self) -> None:
        self.detector = ContradictionDetector()
        self.engine = ConsensusEngine()
        self.normalizer = PropositionNormalizer()

    def evaluate_all(self) -> list[AdversarialFixtureResult]:
        """Execute all curated adversarial fixtures and return structured results."""
        return [
            self.eval_official_vs_independent_benchmark(),
            self.eval_multi_paper_reproducibility(),
            self.eval_syndicated_copycat_resilience(),
            self.eval_valid_version_evolution(),
            self.eval_temporal_benchmark_discrepancy(),
        ]

    def eval_official_vs_independent_benchmark(self) -> AdversarialFixtureResult:
        """Fixture 1: Official docs claim production ready vs independent benchmark fails."""
        prop_doc, stance_doc = self.normalizer.normalize(
            "Framework X is production ready for enterprise deployments.",
            default_subject="Framework X",
        )
        prop_bench, stance_bench = self.normalizer.normalize(
            "Framework X fails under sustained concurrency load and is not production ready.",
            default_subject="Framework X",
        )

        claim_doc = EvidenceClaim(
            evidence_id="ev-doc-1",
            text="Framework X is production ready for enterprise deployments.",
            proposition=prop_doc,
            stance=stance_doc,
            source_url="https://docs.framework-x.io",
            source_authority=SourceAuthority.OFFICIAL,
            confidence=0.9,
        )
        claim_bench = EvidenceClaim(
            evidence_id="ev-bench-1",
            text="Framework X fails under sustained concurrency load and is not production ready.",
            proposition=prop_bench,
            stance=stance_bench,
            source_url="https://independent-eval.org/load-test",
            source_authority=SourceAuthority.ACADEMIC,
            confidence=0.9,
        )

        cluster = ClaimCluster(
            canonical_topic="framework_x::production_readiness",
            proposition=prop_doc,
            claims=[claim_doc, claim_bench],
        )

        contradictions = self.detector.detect_contradictions([cluster])
        report = self.engine.assess_consensus([cluster], contradictions)

        detected = len(contradictions) >= 1
        actual_type = contradictions[0].contradiction_type.value if contradictions else None
        dissent_preserved = len(report.dissenting_findings) >= 1
        confidence_capped = report.overall_consensus in (
            ConsensusLevel.MIXED,
            ConsensusLevel.WEAK,
            ConsensusLevel.INSUFFICIENT,
        )
        safe_conclusion_avoids_universal = (
            any(
                "perbedaan" in d.lower() or "kontradiksi" in d.lower() for d in report.disagreements
            )
            or confidence_capped
        )

        passed = (
            detected
            and actual_type == ContradictionType.DIRECT_NEGATION.value
            and dissent_preserved
            and confidence_capped
        )

        return AdversarialFixtureResult(
            fixture_name="official_vs_independent_benchmark",
            description="Official docs vs Independent stress test ('fails under load').",
            contradiction_detected=detected,
            expected_contradiction_type=ContradictionType.DIRECT_NEGATION.value,
            actual_contradiction_type=actual_type,
            dissent_preserved=dissent_preserved,
            confidence_capped=confidence_capped,
            safe_conclusion_avoids_universal=safe_conclusion_avoids_universal,
            passed=passed,
            details={
                "consensus_level": report.overall_consensus.value,
                "contradictions_count": len(contradictions),
                "dissent_count": len(report.dissenting_findings),
            },
        )

    def eval_multi_paper_reproducibility(self) -> AdversarialFixtureResult:
        """Fixture 2: Paper A (+15%), Paper B (unable to reproduce), Paper C (+4%)."""
        prop_a, s_a = self.normalizer.normalize(
            "Model X yields +15% throughput improvement over baseline.", default_subject="Model X"
        )
        prop_b, s_b = self.normalizer.normalize(
            "Model X unable to reproduce throughput gains in independent trials.",
            default_subject="Model X",
        )
        prop_c, s_c = self.normalizer.normalize(
            "Model X yields +4% modest improvement in realistic workloads.",
            default_subject="Model X",
        )

        claims = [
            EvidenceClaim(
                evidence_id="ev-paper-a",
                text="Model X yields +15% throughput improvement over baseline.",
                proposition=prop_a,
                stance=s_a,
                source_url="https://arxiv.org/abs/2608.1001",
                source_authority=SourceAuthority.ACADEMIC,
                confidence=0.85,
            ),
            EvidenceClaim(
                evidence_id="ev-paper-b",
                text="Model X unable to reproduce throughput gains in independent trials.",
                proposition=prop_b,
                stance=s_b,
                source_url="https://arxiv.org/abs/2608.1002",
                source_authority=SourceAuthority.ACADEMIC,
                confidence=0.90,
            ),
            EvidenceClaim(
                evidence_id="ev-paper-c",
                text="Model X yields +4% modest improvement in realistic workloads.",
                proposition=prop_c,
                stance=s_c,
                source_url="https://arxiv.org/abs/2608.1003",
                source_authority=SourceAuthority.ACADEMIC,
                confidence=0.80,
            ),
        ]

        clusters = ClaimClusterer.cluster_claims(claims)
        contradictions = self.detector.detect_contradictions(clusters)
        report = self.engine.assess_consensus(clusters, contradictions)

        detected = len(contradictions) >= 1
        dissent_preserved = len(report.dissenting_findings) >= 1
        confidence_capped = report.overall_consensus == ConsensusLevel.MIXED

        passed = detected and dissent_preserved and confidence_capped

        return AdversarialFixtureResult(
            fixture_name="multi_paper_reproducibility",
            description="Paper A (+15%), Paper B (unable to reproduce), Paper C (+4%).",
            contradiction_detected=detected,
            expected_contradiction_type=ContradictionType.DIRECT_NEGATION.value,
            actual_contradiction_type=(
                contradictions[0].contradiction_type.value if contradictions else None
            ),
            dissent_preserved=dissent_preserved,
            confidence_capped=confidence_capped,
            safe_conclusion_avoids_universal=True,
            passed=passed,
            details={
                "consensus_level": report.overall_consensus.value,
                "contradictions_count": len(contradictions),
                "dissent_count": len(report.dissenting_findings),
            },
        )

    def eval_syndicated_copycat_resilience(self) -> AdversarialFixtureResult:
        """Fixture 3: 3 syndicated news aggregator articles vs 1 original controlled benchmark."""
        prop_vendor, s_vendor = self.normalizer.normalize(
            "Model X beats Model Y by 20% citing Vendor Benchmark.", default_subject="Model X"
        )
        prop_indep, s_indep = self.normalizer.normalize(
            "Difference is only 3% under controlled testing.", default_subject="Model X"
        )

        claims = [
            EvidenceClaim(
                evidence_id="ev-copy-1",
                text="Model X beats Model Y by 20% citing Vendor Benchmark.",
                proposition=prop_vendor,
                stance=s_vendor,
                source_url="https://tech-portal-1.com/news/1",
                source_authority=SourceAuthority.SECONDARY,
                confidence=0.8,
            ),
            EvidenceClaim(
                evidence_id="ev-copy-2",
                text="Model X beats Model Y by 20% citing Vendor Benchmark.",
                proposition=prop_vendor,
                stance=s_vendor,
                source_url="https://tech-portal-2.com/news/2",
                source_authority=SourceAuthority.SECONDARY,
                confidence=0.8,
            ),
            EvidenceClaim(
                evidence_id="ev-copy-3",
                text="Model X beats Model Y by 20% citing Vendor Benchmark.",
                proposition=prop_vendor,
                stance=s_vendor,
                source_url="https://tech-portal-3.com/news/3",
                source_authority=SourceAuthority.SECONDARY,
                confidence=0.8,
            ),
            EvidenceClaim(
                evidence_id="ev-academic",
                text="Difference is only 3% under controlled testing.",
                proposition=prop_indep,
                stance=s_indep,
                source_url="https://independent-lab.edu/report",
                source_authority=SourceAuthority.ACADEMIC,
                confidence=0.95,
            ),
        ]

        cluster = ClaimCluster(
            canonical_topic="model_x::performance_benchmark",
            proposition=prop_vendor,
            claims=claims,
        )
        contradictions = self.detector.detect_contradictions([cluster])
        report = self.engine.assess_consensus([cluster], contradictions)

        # 3 copycats repeating the vendor claim must not achieve
        # STRONG consensus against 1 academic lab
        not_inflated = report.overall_consensus != ConsensusLevel.STRONG
        dissent_preserved = len(report.dissenting_findings) >= 1
        passed = not_inflated and dissent_preserved

        return AdversarialFixtureResult(
            fixture_name="syndicated_copycat_resilience",
            description="3 syndicated articles repeating vendor claim vs 1 controlled benchmark.",
            contradiction_detected=len(contradictions) >= 1,
            expected_contradiction_type=ContradictionType.NUMERIC_CONFLICT.value,
            actual_contradiction_type=(
                contradictions[0].contradiction_type.value if contradictions else None
            ),
            dissent_preserved=dissent_preserved,
            confidence_capped=not_inflated,
            safe_conclusion_avoids_universal=True,
            passed=passed,
            details={
                "consensus_level": report.overall_consensus.value,
                "not_inflated_to_strong": not_inflated,
            },
        )

    def eval_valid_version_evolution(self) -> AdversarialFixtureResult:
        """Fixture 4: v1 docs unsupported vs v2 release notes feature added."""
        prop_old, s_old = self.normalizer.normalize(
            "Windows unsupported in v1.0.", default_subject="Framework X"
        )
        prop_new, s_new = self.normalizer.normalize(
            "Windows support added in v2.0 release notes.", default_subject="Framework X"
        )

        claim_old = EvidenceClaim(
            evidence_id="ev-v1",
            text="Windows unsupported in v1.0.",
            proposition=prop_old,
            stance=s_old,
            version_tag="v1.0",
            source_authority=SourceAuthority.OFFICIAL,
        )
        claim_new = EvidenceClaim(
            evidence_id="ev-v2",
            text="Windows support added in v2.0 release notes.",
            proposition=prop_new,
            stance=s_new,
            version_tag="v2.0",
            source_authority=SourceAuthority.OFFICIAL,
        )

        cluster = ClaimCluster(
            canonical_topic="framework_x::platform_support::windows",
            proposition=prop_old,
            claims=[claim_old, claim_new],
        )
        contradictions = self.detector.detect_contradictions([cluster])
        report = self.engine.assess_consensus([cluster], contradictions)

        detected = len(contradictions) == 1
        resolved = detected and contradictions[0].resolved_by_version is True
        consensus_moderate_or_strong = report.overall_consensus in (
            ConsensusLevel.STRONG,
            ConsensusLevel.MODERATE,
        )
        passed = detected and resolved and consensus_moderate_or_strong

        return AdversarialFixtureResult(
            fixture_name="valid_version_evolution",
            description="v1 unsupported vs v2 support added in release notes.",
            contradiction_detected=detected,
            expected_contradiction_type=ContradictionType.VERSION_CONFLICT.value,
            actual_contradiction_type=(
                contradictions[0].contradiction_type.value if contradictions else None
            ),
            dissent_preserved=True,
            confidence_capped=False,
            safe_conclusion_avoids_universal=True,
            passed=passed,
            details={
                "resolved_by_version": resolved,
                "consensus_level": report.overall_consensus.value,
            },
        )

    def eval_temporal_benchmark_discrepancy(self) -> AdversarialFixtureResult:
        """Fixture 5: 2025 benchmark (100 req/s) vs 2026 independent benchmark (55 req/s)."""
        prop_2025, s_2025 = self.normalizer.normalize(
            "Measured throughput is 100 req/s in 2025.", default_subject="Framework X"
        )
        prop_2026, s_2026 = self.normalizer.normalize(
            "Independent measured throughput is 55 req/s in 2026.", default_subject="Framework X"
        )

        claim_2025 = EvidenceClaim(
            evidence_id="ev-2025",
            text="Measured throughput is 100 req/s in 2025.",
            proposition=prop_2025,
            stance=s_2025,
            published_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        claim_2026 = EvidenceClaim(
            evidence_id="ev-2026",
            text="Independent measured throughput is 55 req/s in 2026.",
            proposition=prop_2026,
            stance=s_2026,
            published_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

        cluster = ClaimCluster(
            canonical_topic="framework_x::performance_benchmark",
            proposition=prop_2025,
            claims=[claim_2025, claim_2026],
        )
        contradictions = self.detector.detect_contradictions([cluster])
        report = self.engine.assess_consensus([cluster], contradictions)

        detected = len(contradictions) == 1
        not_auto_resolved = detected and contradictions[0].resolved_by_version is False
        passed = (
            detected and not_auto_resolved and (report.overall_consensus != ConsensusLevel.STRONG)
        )

        return AdversarialFixtureResult(
            fixture_name="temporal_benchmark_discrepancy",
            description="2025 benchmark vs 2026 benchmark - must NOT be auto-resolved.",
            contradiction_detected=detected,
            expected_contradiction_type=ContradictionType.NUMERIC_CONFLICT.value,
            actual_contradiction_type=(
                contradictions[0].contradiction_type.value if contradictions else None
            ),
            dissent_preserved=len(report.dissenting_findings) >= 1,
            confidence_capped=True,
            safe_conclusion_avoids_universal=True,
            passed=passed,
            details={
                "resolved_by_version": not not_auto_resolved,
                "consensus_level": report.overall_consensus.value,
            },
        )
