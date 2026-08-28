"""Source-weighted, copy-resistant consensus engine with dissent preservation."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from urllib.parse import urlparse

from research_radar.consensus.models import (
    ClaimCluster,
    ClaimStance,
    ConsensusAssessment,
    ConsensusLevel,
    ConsensusReport,
    ContradictionRecord,
    ContradictionSeverity,
    EvidenceClaim,
)
from research_radar.evidence.models import SourceAuthority

logger = logging.getLogger(__name__)

_AUTHORITY_WEIGHTS: dict[SourceAuthority, float] = {
    SourceAuthority.PRIMARY: 1.0,
    SourceAuthority.ACADEMIC: 0.95,
    SourceAuthority.OFFICIAL: 0.90,
    SourceAuthority.SECONDARY: 0.50,
    SourceAuthority.COMMUNITY: 0.40,
    SourceAuthority.UNKNOWN: 0.30,
}


def _extract_domain_owner(url: str) -> str:
    """Extract domain or repository owner for source independence evaluation."""
    if not url:
        return "unknown"
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if "github.com" in netloc:
            parts = parsed.path.strip("/").split("/")
            if parts and parts[0]:
                return f"github:{parts[0]}"
        return netloc or "unknown"
    except Exception:
        return "unknown"


def _extract_information_origin(claim: EvidenceClaim) -> str:
    """Extract information origin identifying domain, cited attribution, or narrative."""
    domain = _extract_domain_owner(claim.source_url)

    # 1. Check for explicit upstream attribution citation in claim text
    text_lower = claim.text.lower()
    attr_match = re.search(
        r"(?:citing|according to|via|source:|dari)\s+([a-z0-9_\-\.]+)", text_lower
    )
    if attr_match:
        upstream = attr_match.group(1).strip(".")
        return f"syndicated:{upstream}"

    # 2. Check for secondary/community repost of specific numeric/proposition claims
    if claim.source_authority in (
        SourceAuthority.SECONDARY,
        SourceAuthority.COMMUNITY,
        SourceAuthority.UNKNOWN,
    ):
        if claim.proposition.object_val and len(claim.proposition.object_val) > 1:
            pred = claim.proposition.predicate
            val = claim.proposition.object_val.lower()
            return f"syndicated_narrative:{pred}:{val}"

    return domain


class ConsensusEngine:
    """Evaluates consensus across claim clusters with source weighting and dissent preservation."""

    def assess_consensus(
        self,
        clusters: list[ClaimCluster],
        contradictions: list[ContradictionRecord],
    ) -> ConsensusReport:
        """Evaluate consensus across all claim clusters."""
        assessments: list[ConsensusAssessment] = []
        dissenting_findings: list[str] = []
        agreements: list[str] = []
        disagreements: list[str] = []

        # Map contradictions by cluster/proposition canonical_key
        contra_by_key: dict[str, list[ContradictionRecord]] = defaultdict(list)
        for c in contradictions:
            contra_by_key[c.proposition.canonical_key].append(c)

        for cluster in clusters:
            assessment = self._assess_single_cluster(
                cluster, contra_by_key.get(cluster.proposition.canonical_key, [])
            )
            assessments.append(assessment)

            # Summarize agreements
            if assessment.consensus_level in (ConsensusLevel.STRONG, ConsensusLevel.MODERATE):
                supp_text = assessment.strongest_support or cluster.proposition.predicate
                agreements.append(f"{cluster.proposition.subject}: {supp_text}")
            elif assessment.consensus_level == ConsensusLevel.MIXED:
                disagreements.append(
                    f"{cluster.proposition.subject} ({cluster.proposition.predicate}): "
                    "Terdapat perbedaan bukti antara sumber pendukung dan penentang."
                )

            # Dissent preservation for active, unresolved opposition or qualification
            is_resolved_version = any(
                c.resolved_by_version
                for c in contradictions
                if c.proposition == cluster.proposition
            )
            if (
                assessment.opposing_evidence_ids or assessment.qualifying_evidence_ids
            ) and not is_resolved_version:
                if assessment.strongest_opposition:
                    subj = cluster.proposition.subject
                    pred = cluster.proposition.predicate
                    opp = assessment.strongest_opposition
                    dissent_text = f"Dissent ({subj} - {pred}): {opp}"
                    if dissent_text not in dissenting_findings:
                        dissenting_findings.append(dissent_text)

        # Derive overall consensus
        overall_consensus = self._derive_overall_consensus(assessments, contradictions)

        logger.info(
            "Consensus assessment completed",
            extra={
                "event": "consensus_assessed",
                "overall_consensus": overall_consensus.value,
                "assessments_count": len(assessments),
                "dissent_count": len(dissenting_findings),
            },
        )

        return ConsensusReport(
            assessments=assessments,
            contradictions=contradictions,
            overall_consensus=overall_consensus,
            agreements=agreements,
            disagreements=disagreements,
            dissenting_findings=dissenting_findings,
        )

    def _assess_single_cluster(
        self,
        cluster: ClaimCluster,
        cluster_contradictions: list[ContradictionRecord],
    ) -> ConsensusAssessment:
        """Assess a single claim cluster."""
        claims = cluster.claims
        if not claims:
            return ConsensusAssessment(
                cluster_id=cluster.cluster_id,
                proposition=cluster.proposition,
                consensus_level=ConsensusLevel.INSUFFICIENT,
            )

        # 1. Compute Source Independence and Weighted Scores
        origin_counts: dict[str, int] = defaultdict(int)
        weighted_support = 0.0
        weighted_opposition = 0.0
        weighted_qualification = 0.0
        weighted_neutral = 0.0

        support_ids: list[str] = []
        oppose_ids: list[str] = []
        qualify_ids: list[str] = []

        strongest_sup = ""
        strongest_opp = ""
        best_sup_score = -1.0
        best_opp_score = -1.0

        for claim in claims:
            origin = _extract_information_origin(claim)
            origin_counts[origin] += 1
            repeat_idx = origin_counts[origin]

            # Diminishing independence factor: 1.0, 0.5, 0.25...
            independence_factor = 1.0 / (2.0 ** (repeat_idx - 1))
            base_weight = _AUTHORITY_WEIGHTS.get(claim.source_authority, 0.30)
            effective_weight = base_weight * independence_factor * claim.confidence

            if claim.stance == ClaimStance.SUPPORTS:
                weighted_support += effective_weight
                support_ids.append(claim.evidence_id)
                if effective_weight > best_sup_score:
                    best_sup_score = effective_weight
                    strongest_sup = claim.text
            elif claim.stance == ClaimStance.OPPOSES:
                weighted_opposition += effective_weight
                oppose_ids.append(claim.evidence_id)
                if effective_weight > best_opp_score:
                    best_opp_score = effective_weight
                    strongest_opp = claim.text
            elif claim.stance == ClaimStance.QUALIFIES:
                weighted_qualification += effective_weight
                qualify_ids.append(claim.evidence_id)
                if effective_weight > best_opp_score:
                    best_opp_score = effective_weight
                    strongest_opp = claim.text
            else:
                weighted_neutral += effective_weight

        total_weight = (
            weighted_support + weighted_opposition + weighted_qualification + weighted_neutral
        )
        assertive_weight = weighted_support + weighted_opposition + weighted_qualification
        agreement_ratio = (weighted_support / total_weight) if total_weight > 0 else 0.0
        distinct_origins = len(origin_counts)

        # 2. Check for unresolved high/medium contradictions and version resolution
        unresolved_high = any(
            c.severity == ContradictionSeverity.HIGH and not c.resolved_by_version
            for c in cluster_contradictions
        )
        unresolved_medium = any(
            c.severity == ContradictionSeverity.MEDIUM and not c.resolved_by_version
            for c in cluster_contradictions
        )
        all_resolved_by_version = bool(cluster_contradictions) and all(
            c.resolved_by_version for c in cluster_contradictions
        )

        # 3. Determine Consensus Level
        if len(claims) < 2 or assertive_weight < 0.60:
            level = ConsensusLevel.INSUFFICIENT
        elif unresolved_high:
            level = ConsensusLevel.MIXED
        elif (
            (weighted_opposition + weighted_qualification) / max(assertive_weight, 0.001) >= 0.25
            and (weighted_opposition >= 0.20 or weighted_qualification >= 0.35)
            and not all_resolved_by_version
        ):
            level = ConsensusLevel.MIXED
        elif (
            (agreement_ratio >= 0.80 or all_resolved_by_version)
            and distinct_origins >= 2
            and not unresolved_medium
            and (weighted_opposition < 0.15 or all_resolved_by_version)
        ):
            level = ConsensusLevel.STRONG
        elif (
            (agreement_ratio >= 0.60 or all_resolved_by_version)
            and not unresolved_medium
            and (weighted_opposition < 0.35 or all_resolved_by_version)
        ):
            level = ConsensusLevel.MODERATE
        elif agreement_ratio < 0.50:
            level = ConsensusLevel.WEAK
        else:
            level = ConsensusLevel.MODERATE

        uncertainty_note = ""
        if unresolved_high:
            uncertainty_note = (
                "Terdapat pertentangan tingkat tinggi yang belum terselesaikan antar sumber."
            )
        elif weighted_qualification > 0:
            uncertainty_note = "Beberapa sumber memberikan kualifikasi atau batasan cakupan."

        return ConsensusAssessment(
            cluster_id=cluster.cluster_id,
            proposition=cluster.proposition,
            consensus_level=level,
            agreement_ratio=agreement_ratio,
            weighted_support=weighted_support,
            weighted_opposition=weighted_opposition,
            weighted_qualification=weighted_qualification,
            supporting_evidence_ids=support_ids,
            opposing_evidence_ids=oppose_ids,
            qualifying_evidence_ids=qualify_ids,
            strongest_support=strongest_sup,
            strongest_opposition=strongest_opp,
            unresolved_uncertainty=uncertainty_note,
            contradictions=cluster_contradictions,
        )

    def _derive_overall_consensus(
        self,
        assessments: list[ConsensusAssessment],
        contradictions: list[ContradictionRecord],
    ) -> ConsensusLevel:
        """Derive global consensus across all topic clusters."""
        if not assessments:
            return ConsensusLevel.INSUFFICIENT

        has_unresolved_high = any(
            c.severity == ContradictionSeverity.HIGH and not c.resolved_by_version
            for c in contradictions
        )
        if has_unresolved_high:
            return ConsensusLevel.MIXED

        levels = [a.consensus_level for a in assessments]
        if all(lvl == ConsensusLevel.STRONG for lvl in levels):
            return ConsensusLevel.STRONG
        if any(lvl == ConsensusLevel.MIXED for lvl in levels):
            return ConsensusLevel.MIXED
        if any(lvl == ConsensusLevel.MODERATE for lvl in levels):
            return ConsensusLevel.MODERATE
        if all(lvl == ConsensusLevel.INSUFFICIENT for lvl in levels):
            return ConsensusLevel.INSUFFICIENT
        return ConsensusLevel.WEAK
