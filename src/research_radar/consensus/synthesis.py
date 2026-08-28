"""Conservative, consensus-aware research synthesis."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from research_radar.consensus.models import ConsensusReport
from research_radar.evidence.models import ClaimType, StructuredClaim
from research_radar.llm.base import LLMRequest
from research_radar.llm.router import LLMRouter, Workload
from research_radar.security.untrusted_content import (
    EVIDENCE_BOUNDARY_INSTRUCTION,
    wrap_evidence_for_llm,
)

logger = logging.getLogger(__name__)


class LLMConsensusSynthesisSchema(BaseModel):
    answer: str = Field(description="Comprehensive conservative research answer")
    key_findings: list[str] = Field(default_factory=list)
    agreements: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    dissenting_views: list[str] = Field(default_factory=list)
    safe_conclusion: str = Field(default="")
    uncertainties: list[str] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)


class ConsensusSynthesizer:
    """Produces conservative, consensus-aware synthesis grounded in cross-evidence analysis."""

    def __init__(self, llm_router: LLMRouter | None = None) -> None:
        self.llm_router = llm_router

    async def synthesize(
        self,
        question: str,
        evidence_items: list[dict[str, Any]],
        consensus_report: ConsensusReport,
        *,
        force_deterministic: bool = False,
    ) -> tuple[dict[str, Any], list[StructuredClaim]]:
        """Synthesize research results incorporating consensus, contradiction, and dissent."""
        if not evidence_items:
            return {
                "answer": "Tidak ditemukan bukti memadai dari sumber terdaftar.",
                "key_findings": [
                    "Tidak ada repositori, paper, atau sumber web relevan yang ditemukan."
                ],
                "agreements": [],
                "disagreements": [],
                "dissenting_views": [],
                "safe_conclusion": "Ketiadaan data empiris.",
                "uncertainties": ["Ketiadaan bukti empiris."],
            }, []

        if self.llm_router is None or force_deterministic:
            return self._synthesize_deterministic(question, evidence_items, consensus_report)

        return await self._synthesize_llm(question, evidence_items, consensus_report)

    def _synthesize_deterministic(
        self,
        question: str,
        evidence_items: list[dict[str, Any]],
        consensus_report: ConsensusReport,
    ) -> tuple[dict[str, Any], list[StructuredClaim]]:
        """Deterministic consensus synthesis fallback."""
        findings: list[str] = []
        for item in evidence_items[:4]:
            t = str(item.get("title", "Sumber"))
            desc = str(item.get("description", "") or item.get("abstract", ""))
            findings.append(f"{t}: {desc}"[:150])

        claims: list[StructuredClaim] = []
        for idx, f in enumerate(findings):
            ev_id = str(evidence_items[idx].get("evidence_id"))
            claims.append(
                StructuredClaim(
                    text=f[:120],
                    claim_type=ClaimType.FACT,
                    evidence_ids=[ev_id],
                    confidence=0.8,
                )
            )

        # Build conservative answer text
        status_val = consensus_report.overall_consensus.value
        lines = [
            f"Analisis konsensus terhadap {len(evidence_items)} sumber (Status: {status_val}):",
        ]
        if consensus_report.agreements:
            lines.append("• Kesepakatan Sumber: " + "; ".join(consensus_report.agreements[:2]))
        if consensus_report.disagreements:
            lines.append("• Perbedaan Temuan: " + "; ".join(consensus_report.disagreements[:2]))
        if consensus_report.dissenting_findings:
            lines.append(
                "• Pandangan Dissent: " + "; ".join(consensus_report.dissenting_findings[:2])
            )

        lines.append("• Temuan Utama:\n" + "\n".join(f"  - {f}" for f in findings))

        safe_conclusion = (
            "Bukti menunjukkan dukungan moderat dengan beberapa kualifikasi pada skala produksi."
            if consensus_report.overall_consensus.value in ("mixed", "moderate")
            else "Bukti mendukung klaim utama secara konsisten."
        )

        return {
            "answer": "\n".join(lines),
            "key_findings": findings,
            "agreements": consensus_report.agreements,
            "disagreements": consensus_report.disagreements,
            "dissenting_views": consensus_report.dissenting_findings,
            "safe_conclusion": safe_conclusion,
            "uncertainties": ["Sintesis konsensus dihasilkan dalam mode deterministik."],
        }, claims

    async def _synthesize_llm(
        self,
        question: str,
        evidence_items: list[dict[str, Any]],
        consensus_report: ConsensusReport,
    ) -> tuple[dict[str, Any], list[StructuredClaim]]:
        """LLM-assisted structured consensus synthesis."""
        assert self.llm_router is not None

        evidence_json = json.dumps(evidence_items, default=str, ensure_ascii=False)
        wrapped_evidence = wrap_evidence_for_llm(evidence_json, max_length=24000)

        # Build consensus context summary
        consensus_summary_lines = [
            f"Status Konsensus Global: {consensus_report.overall_consensus.value.upper()}",
            "Kesepakatan Terdeteksi: "
            + ("; ".join(consensus_report.agreements) or "Belum teridentifikasi"),
            "Perbedaan/Kontradiksi: "
            + ("; ".join(consensus_report.disagreements) or "Tidak ada pertentangan"),
            "Dissenting Views: " + ("; ".join(consensus_report.dissenting_findings) or "N/A"),
        ]
        consensus_context = "\n".join(consensus_summary_lines)

        prompt = (
            f"Pertanyaan Riset: '{question}'\n\n"
            f"Ringkasan Analisis Konsensus & Kontradiksi:\n{consensus_context}\n\n"
            f"{wrapped_evidence}\n\n"
            "Tugas: Buat sintesis riset berbasis konsensus yang objektif dan konservatif.\n"
            "Prinsip Wajib:\n"
            "1. Jelaskan apa yang disepakati secara kuat oleh mayoritas sumber independen.\n"
            "2. Jelaskan di mana sumber berbeda pendapat (disagreements) dan mengapa berbeda "
            "(perbedaan versi, skala pengujian, atau metodologi benchmark).\n"
            "3. JAGA bukti minoritas (dissent) dari benchmark independen agar tidak hilang.\n"
            "4. Buat kesimpulan konservatif (safe conclusion) yang membedakan fakta terbukti "
            "dari klaim yang belum matang.\n"
            "5. Berikan daftar klaim terstruktur dengan evidence_id valid untuk verifikasi P0.\n"
        )

        try:
            response = await self.llm_router.generate_structured(
                LLMRequest(
                    prompt=prompt,
                    system_instruction=(
                        f"{EVIDENCE_BOUNDARY_INSTRUCTION} "
                        "You are an evidence-first AI research scientist specialized in consensus "
                        "and contradiction analysis. Ground every claim in evidence IDs. "
                        "Never fabricate consensus or suppress dissenting findings."
                    ),
                    max_output_tokens=3000,
                ),
                LLMConsensusSynthesisSchema,
                Workload.REASONING,
            )
            data = response.data

            claims: list[StructuredClaim] = []
            for c in data.claims:
                text = c.get("text", "")
                if not text:
                    continue
                raw_ids = c.get("evidence_ids")
                if not isinstance(raw_ids, list):
                    single_id = c.get("evidence_id")
                    raw_ids = [single_id] if single_id else []
                ev_ids = [str(eid) for eid in raw_ids if eid]
                is_interp = c.get("claim_type") == "interpretation" or text.startswith("Inferensi:")
                claims.append(
                    StructuredClaim(
                        text=text,
                        claim_type=ClaimType.INTERPRETATION if is_interp else ClaimType.FACT,
                        evidence_ids=ev_ids,
                        confidence=0.85,
                    )
                )

            return {
                "answer": data.answer,
                "key_findings": data.key_findings,
                "agreements": data.agreements,
                "disagreements": data.disagreements,
                "dissenting_views": data.dissenting_views,
                "safe_conclusion": data.safe_conclusion,
                "uncertainties": data.uncertainties,
            }, claims

        except Exception as exc:
            logger.warning(
                "LLM consensus synthesis failed; falling back to deterministic synthesis",
                extra={"event": "consensus_synthesis_fallback", "error": str(exc)},
            )
            return self._synthesize_deterministic(question, evidence_items, consensus_report)
