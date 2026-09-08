"""Atomic claim extraction from evidence items."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from research_radar.consensus.models import (
    ClaimStance,
    EvidenceClaim,
    NormalizedProposition,
)
from research_radar.consensus.normalizer import PropositionNormalizer
from research_radar.evidence.models import SourceAuthority
from research_radar.llm.base import LLMRequest
from research_radar.llm.router import LLMRouter, Workload
from research_radar.security.untrusted_content import (
    EVIDENCE_BOUNDARY_INSTRUCTION,
    wrap_evidence_for_llm,
)

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


class ExtractedClaimItemSchema(BaseModel):
    evidence_id: str
    text: str
    subject: str = "general"
    predicate: str = "general_claim"
    object_val: str | None = None
    qualifier: str | None = None
    stance: str = "supports"  # "supports", "opposes", "qualifies", "neutral", "unknown"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class LLMExtractedClaimBatchSchema(BaseModel):
    claims: list[ExtractedClaimItemSchema] = Field(default_factory=list)


class ClaimExtractor:
    """Extracts atomic proposition claims from collected evidence items."""

    def __init__(self, llm_router: LLMRouter | None = None) -> None:
        self.llm_router = llm_router
        self.normalizer = PropositionNormalizer()

    async def extract_claims(
        self,
        evidence_items: list[dict[str, Any]],
        *,
        use_llm: bool = True,
        max_claims_per_item: int = 4,
    ) -> list[EvidenceClaim]:
        """Extract atomic evidence claims using deterministic and bounded LLM extraction."""
        if not evidence_items:
            return []

        # If LLM available and requested, try bounded structured extraction
        if self.llm_router is not None and use_llm:
            try:
                llm_claims = await self._extract_with_llm(evidence_items)
                if llm_claims:
                    logger.info(
                        "Extracted atomic claims with LLM",
                        extra={
                            "event": "claim_extraction_llm_success",
                            "claim_count": len(llm_claims),
                        },
                    )
                    return llm_claims
            except Exception as exc:
                logger.warning(
                    "LLM claim extraction failed; falling back to deterministic extraction",
                    extra={"event": "claim_extraction_fallback", "error_type": type(exc).__name__},
                )

        # Deterministic extraction fallback
        return self._extract_deterministic(evidence_items, max_claims_per_item=max_claims_per_item)

    def _extract_deterministic(
        self,
        evidence_items: list[dict[str, Any]],
        *,
        max_claims_per_item: int = 4,
    ) -> list[EvidenceClaim]:
        """Deterministic extraction splitting text into coherent atomic sentences."""
        extracted: list[EvidenceClaim] = []

        for item in evidence_items:
            ev_id = str(item.get("evidence_id", ""))
            if not ev_id:
                continue

            title = str(item.get("title", ""))
            desc = str(item.get("description", "") or "")
            abstract = str(item.get("abstract", "") or "")
            readme = str(item.get("readme_excerpt", "") or "")
            source_type = str(item.get("source_type", "web"))
            source_url = str(item.get("url", "") or item.get("canonical_url", ""))
            authority_str = str(item.get("authority", item.get("source_authority", "unknown")))
            try:
                authority = SourceAuthority(authority_str.lower())
            except ValueError:
                authority = SourceAuthority.UNKNOWN

            published_at = item.get("published_at")
            if isinstance(published_at, str):
                try:
                    published_at = datetime.fromisoformat(published_at)
                except ValueError:
                    published_at = None

            version_tag = (
                str(item.get("latest_release_tag", "") or item.get("version_tag", "")) or None
            )

            # Combine descriptive text
            full_text = f"{title}. {desc} {abstract} {readme}".strip()
            sentences = [
                s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(full_text) if len(s.strip()) > 15
            ]

            added_for_item = 0
            for sent in sentences:
                if added_for_item >= max_claims_per_item:
                    break

                prop, stance = self.normalizer.normalize(sent, default_subject=title or "framework")
                extracted.append(
                    EvidenceClaim(
                        evidence_id=ev_id,
                        text=sent[:200],
                        proposition=prop,
                        stance=stance,
                        confidence=0.8,
                        source_authority=authority,
                        source_url=source_url,
                        source_type=source_type,
                        published_at=published_at,
                        version_tag=version_tag,
                    )
                )
                added_for_item += 1

        logger.info(
            "Extracted atomic claims deterministically",
            extra={
                "event": "claim_extraction_deterministic_success",
                "claim_count": len(extracted),
            },
        )
        return extracted

    async def _extract_with_llm(
        self,
        evidence_items: list[dict[str, Any]],
    ) -> list[EvidenceClaim]:
        """Bounded structured LLM claim extraction."""
        import json

        assert self.llm_router is not None

        # Build compact evidence summary
        compact_items = []
        for it in evidence_items[:6]:
            compact_items.append(
                {
                    "evidence_id": it.get("evidence_id"),
                    "title": it.get("title"),
                    "description": (it.get("description") or it.get("abstract") or "")[:300],
                    "authority": str(it.get("authority", "unknown")),
                }
            )

        evidence_json = json.dumps(compact_items, default=str, ensure_ascii=False)
        wrapped = wrap_evidence_for_llm(evidence_json, max_length=16000)

        prompt = (
            "Tugas: Ekstraksi klaim-klaim atomik (atomic claims) dari bukti eksternal berikut.\n"
            "Klaim atomik harus berupa satu proposisi spesifik yang dapat diverifikasi.\n\n"
            f"{wrapped}\n\n"
            "Instruksi:\n"
            "1. Ekstraksi 1-3 klaim per evidence_id.\n"
            "2. Identifikasi subject, predicate (misal: 'production_readiness', "
            "'vram_requirement', 'platform_support'), dan object_val jika ada.\n"
            "3. Tentukan stance ('supports', 'opposes', 'qualifies', 'neutral', 'unknown').\n"
            "4. JANGAN membuat klaim yang tidak ada di bukti."
        )

        response = await self.llm_router.generate_structured(
            LLMRequest(
                prompt=prompt,
                system_instruction=(
                    f"{EVIDENCE_BOUNDARY_INSTRUCTION} "
                    "Extract structured atomic propositions from untrusted evidence data."
                ),
                max_output_tokens=2048,
            ),
            LLMExtractedClaimBatchSchema,
            Workload.REASONING,
        )

        data = response.data
        results: list[EvidenceClaim] = []
        ev_map = {str(it.get("evidence_id")): it for it in evidence_items}

        for c in data.claims:
            ev_id = str(c.evidence_id)
            if ev_id not in ev_map or not c.text:
                continue

            orig = ev_map[ev_id]
            auth_str = str(orig.get("authority", orig.get("source_authority", "unknown")))
            try:
                auth = SourceAuthority(auth_str.lower())
            except ValueError:
                auth = SourceAuthority.UNKNOWN

            try:
                stance_val = ClaimStance(c.stance.lower())
            except ValueError:
                stance_val = ClaimStance.SUPPORTS

            norm_subj = PropositionNormalizer.normalize_subject(c.subject)
            prop = NormalizedProposition(
                subject=norm_subj,
                predicate=c.predicate,
                object_val=c.object_val,
                qualifier=c.qualifier,
            )

            results.append(
                EvidenceClaim(
                    evidence_id=ev_id,
                    text=c.text[:200],
                    proposition=prop,
                    stance=stance_val,
                    confidence=float(c.confidence),
                    source_authority=auth,
                    source_url=str(orig.get("url", "") or orig.get("canonical_url", "")),
                    source_type=str(orig.get("source_type", "web")),
                    version_tag=str(
                        orig.get("latest_release_tag", "") or orig.get("version_tag", "")
                    )
                    or None,
                )
            )

        return results
