"""Deterministic post-synthesis claim verification.

Verifies that factual claims in digest synthesis output are properly grounded
in registered evidence IDs and evidence content.

Checks:
1. Evidence ID existence and belonging to digest context
2. Factual claim grounding (token overlap against evidence content)
3. Direct support vs partial support vs unsupported content
4. Detection of explicit contradictions (CONFLICTING status)
5. Downgrading unsupported factual claims to interpretations
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from research_radar.evidence.models import (
    ClaimType,
    StructuredClaim,
    VerificationStatus,
)

logger = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"[a-z0-9][a-z0-9+.-]*", re.IGNORECASE)

_STOPWORDS = {
    "a",
    "about",
    "adalah",
    "after",
    "akan",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "bisa",
    "but",
    "by",
    "can",
    "dapat",
    "dan",
    "dari",
    "dengan",
    "di",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "in",
    "ini",
    "into",
    "is",
    "it",
    "its",
    "itu",
    "juga",
    "ke",
    "lebih",
    "more",
    "new",
    "no",
    "not",
    "of",
    "oleh",
    "on",
    "or",
    "our",
    "pada",
    "sebagai",
    "sebuah",
    "she",
    "some",
    "sudah",
    "telah",
    "than",
    "that",
    "the",
    "their",
    "them",
    "these",
    "they",
    "this",
    "those",
    "to",
    "untuk",
    "up",
    "uses",
    "using",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "will",
    "with",
    "yang",
    "you",
    "your",
    # Domain generic terms
    "repository",
    "framework",
    "tool",
    "library",
    "system",
    "model",
    "paper",
    "project",
    "supports",
    "support",
}

_CONTRADICTION_TRIGGERS = (
    "does not support",
    "doesn't support",
    "not supported",
    "cannot use",
    "can not use",
    "deprecated",
    "abandoned",
    "no longer supports",
    "tidak mendukung",
    "tidak kompatibel",
    "bukan bagian dari",
)


def _extract_keywords(text: str) -> list[str]:
    return [
        token.lower()
        for token in _WORD_PATTERN.findall(text)
        if len(token) > 2 and token.lower() not in _STOPWORDS
    ]


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Result of verifying a single claim."""

    claim: StructuredClaim
    status: VerificationStatus
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Aggregate verification report for a digest synthesis."""

    results: list[VerificationResult]
    total_claims: int
    supported_claims: int
    partially_supported_claims: int
    unsupported_claims: int
    conflicting_claims: int
    citation_coverage: float  # 0-1, fraction of claims with at least one evidence

    @property
    def all_supported(self) -> bool:
        return self.unsupported_claims == 0 and self.conflicting_claims == 0


class ClaimVerifier:
    """Deterministic claim-evidence verifier."""

    def __init__(self, *, min_evidence_for_fact: int = 1) -> None:
        self.min_evidence_for_fact = min_evidence_for_fact

    def verify(
        self,
        claims: list[StructuredClaim],
        context_evidence_ids: set[str],
        context_evidence_content: dict[str, str] | None = None,
    ) -> VerificationReport:
        """Verify claims against available evidence context.

        Args:
            claims: Claims from synthesis output
            context_evidence_ids: Set of evidence IDs in the current digest
            context_evidence_content: Optional mapping of evidence_id -> text content

        Returns:
            VerificationReport with per-claim results and aggregate metrics
        """
        results: list[VerificationResult] = []

        for claim in claims:
            result = self._verify_single(claim, context_evidence_ids, context_evidence_content)
            results.append(result)

        supported = sum(1 for r in results if r.status == VerificationStatus.SUPPORTED)
        partial = sum(1 for r in results if r.status == VerificationStatus.PARTIALLY_SUPPORTED)
        unsupported = sum(1 for r in results if r.status == VerificationStatus.UNSUPPORTED)
        conflicting = sum(1 for r in results if r.status == VerificationStatus.CONFLICTING)

        with_evidence = sum(1 for c in claims if c.evidence_ids)
        coverage = with_evidence / len(claims) if claims else 1.0

        report = VerificationReport(
            results=results,
            total_claims=len(claims),
            supported_claims=supported,
            partially_supported_claims=partial,
            unsupported_claims=unsupported,
            conflicting_claims=conflicting,
            citation_coverage=coverage,
        )

        if report.unsupported_claims > 0 or report.conflicting_claims > 0:
            logger.warning(
                "Unsupported or conflicting claims detected",
                extra={
                    "event": "verification_issues_detected",
                    "unsupported": report.unsupported_claims,
                    "conflicting": report.conflicting_claims,
                    "total": report.total_claims,
                },
            )

        return report

    def _verify_single(
        self,
        claim: StructuredClaim,
        context_evidence_ids: set[str],
        context_evidence_content: dict[str, str] | None = None,
    ) -> VerificationResult:
        """Verify a single claim."""
        reasons: list[str] = []

        # Interpretations are always labeled as inference
        if claim.claim_type == ClaimType.INTERPRETATION:
            if not claim.evidence_ids:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.PARTIALLY_SUPPORTED,
                    reasons=["Interpretation without explicit evidence reference"],
                )
            valid_refs = [eid for eid in claim.evidence_ids if eid in context_evidence_ids]
            if valid_refs:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.SUPPORTED,
                    reasons=[f"Interpretation grounded in {len(valid_refs)} evidence item(s)"],
                )
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.PARTIALLY_SUPPORTED,
                reasons=["Interpretation references evidence not in current context"],
            )

        # Factual claims require evidence
        if not claim.evidence_ids:
            reasons.append("Factual claim without evidence references")
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNSUPPORTED,
                reasons=reasons,
            )

        # Check evidence IDs exist in context
        valid_ids = [eid for eid in claim.evidence_ids if eid in context_evidence_ids]
        missing_ids = [eid for eid in claim.evidence_ids if eid not in context_evidence_ids]

        if missing_ids:
            reasons.append(f"Evidence IDs not in context: {', '.join(missing_ids)}")

        if not valid_ids:
            reasons.append("No valid evidence references in current context")
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNSUPPORTED,
                reasons=reasons,
            )

        if len(valid_ids) < self.min_evidence_for_fact:
            reasons.append(
                f"Insufficient evidence: {len(valid_ids)} < {self.min_evidence_for_fact} required"
            )
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.PARTIALLY_SUPPORTED,
                reasons=reasons,
            )

        # If content map is provided, verify content overlap and contradiction
        if context_evidence_content is not None:
            combined_evidence_text = " ".join(
                context_evidence_content.get(eid, "") for eid in valid_ids
            ).lower()

            # 1. Contradiction check
            for trigger in _CONTRADICTION_TRIGGERS:
                if trigger in combined_evidence_text:
                    reasons.append(
                        f"Evidence explicitly contains contradiction/negation: '{trigger}'"
                    )
                    return VerificationResult(
                        claim=claim,
                        status=VerificationStatus.CONFLICTING,
                        reasons=reasons,
                    )

            # 2. Content token overlap check
            claim_keywords = _extract_keywords(claim.text)
            if claim_keywords:
                matched_keywords = [kw for kw in claim_keywords if kw in combined_evidence_text]
                match_ratio = len(matched_keywords) / len(claim_keywords)

                if match_ratio < 0.25:
                    reasons.append(
                        "Evidence content does not support key claim assertions "
                        f"({len(matched_keywords)}/{len(claim_keywords)} matched)"
                    )
                    return VerificationResult(
                        claim=claim,
                        status=VerificationStatus.UNSUPPORTED,
                        reasons=reasons,
                    )
                if match_ratio < 0.70:
                    reasons.append(
                        "Only partial keyword match "
                        f"({len(matched_keywords)}/{len(claim_keywords)} terms matched)"
                    )
                    return VerificationResult(
                        claim=claim,
                        status=VerificationStatus.PARTIALLY_SUPPORTED,
                        reasons=reasons,
                    )

        if missing_ids:
            reasons.append(f"Supported by {len(valid_ids)} valid evidence item(s)")
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.PARTIALLY_SUPPORTED,
                reasons=reasons,
            )

        return VerificationResult(
            claim=claim,
            status=VerificationStatus.SUPPORTED,
            reasons=[f"Fully supported by {len(valid_ids)} evidence item(s)"],
        )

    def downgrade_unsupported_claims(
        self,
        claims: list[StructuredClaim],
        report: VerificationReport,
    ) -> list[StructuredClaim]:
        """Downgrade unsupported or conflicting factual claims to interpretations."""
        downgraded: list[StructuredClaim] = []
        status_by_text = {r.claim.text: r.status for r in report.results}

        for claim in claims:
            status = status_by_text.get(claim.text)
            if claim.claim_type == ClaimType.FACT and status in (
                VerificationStatus.UNSUPPORTED,
                VerificationStatus.CONFLICTING,
            ):
                downgraded.append(
                    StructuredClaim(
                        text=claim.text,
                        claim_type=ClaimType.INTERPRETATION,
                        evidence_ids=claim.evidence_ids,
                        confidence=min(claim.confidence, 0.4),
                        verification_status=status or VerificationStatus.UNSUPPORTED,
                    )
                )
                logger.info(
                    "Downgraded unsupported/conflicting fact to interpretation",
                    extra={"event": "claim_downgraded", "claim_text": claim.text[:100]},
                )
            else:
                updated_status = status or claim.verification_status
                downgraded.append(
                    StructuredClaim(
                        text=claim.text,
                        claim_type=claim.claim_type,
                        evidence_ids=claim.evidence_ids,
                        confidence=claim.confidence,
                        verification_status=updated_status,
                    )
                )

        return downgraded
