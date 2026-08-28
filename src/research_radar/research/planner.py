"""Research planning, question decomposition, and knowledge gap detection."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from research_radar.research.models import (
    EvidenceSufficiency,
    KnowledgeGap,
    ResearchMode,
    ResearchPlan,
    SearchStep,
)

if TYPE_CHECKING:
    from research_radar.agent.base import HermesRuntime
    from research_radar.llm.router import LLMRouter

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
    "more",
    "new",
    "not",
    "of",
    "on",
    "or",
    "pada",
    "sebagai",
    "that",
    "the",
    "their",
    "these",
    "they",
    "this",
    "to",
    "untuk",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "yang",
}


def _extract_keywords(text: str) -> list[str]:
    return [
        w.lower() for w in _WORD_PATTERN.findall(text) if len(w) > 2 and w.lower() not in _STOPWORDS
    ]


class ResearchPlanner:
    """Decomposes research questions and identifies knowledge gaps."""

    def __init__(
        self,
        *,
        llm_router: LLMRouter | None = None,
        hermes_runtime: HermesRuntime | None = None,
    ) -> None:
        self.llm_router = llm_router
        self.hermes_runtime = hermes_runtime

    async def plan(self, question: str, mode: ResearchMode) -> ResearchPlan:
        """Create a structured research plan with sub-questions and initial search steps."""
        if not question.strip():
            return self._fallback_plan("AI Research", mode)

        # Attempt structured planning via LLM if available
        if self.llm_router is not None:
            try:
                return await self._plan_with_llm(question, mode)
            except Exception as exc:
                logger.warning(
                    "LLM research planning failed; using deterministic fallback",
                    extra={"event": "planner_fallback", "error": str(exc)},
                )

        return self._fallback_plan(question, mode)

    async def _plan_with_llm(self, question: str, mode: ResearchMode) -> ResearchPlan:
        from research_radar.llm.base import LLMRequest
        from research_radar.llm.router import Workload

        step_limit = 2 if mode == ResearchMode.QUICK else 4
        prompt = (
            f"Generate a concise research plan for this question: '{question}'.\n"
            f"Mode: {mode.value.upper()}.\n"
            "Return JSON matching this exact structure:\n"
            "{\n"
            '  "objective": "Clear single-sentence goal",\n'
            '  "sub_questions": ["sub-question 1", "sub-question 2"],\n'
            '  "search_steps": [\n'
            '    {"tool": "github", "query": "search query 1", "reason": "why"},\n'
            '    {"tool": "arxiv", "query": "search query 2", "reason": "why"}\n'
            "  ],\n"
            '  "success_criteria": ["criterion 1"]\n'
            "}\n"
            "Tools allowed ONLY: 'github', 'arxiv', 'web', 'news'. "
            f"Limit search_steps to max {step_limit}."
        )

        assert self.llm_router is not None
        response = await self.llm_router.generate_structured(
            LLMRequest(
                prompt=prompt,
                system_instruction=(
                    "You are a principal AI research engineer. Create structured, highly specific "
                    "technical research plans. Return valid JSON only."
                ),
                max_output_tokens=1024,
            ),
            ResearchPlan,
            Workload.REASONING,
        )
        plan = response.data

        # Sanitize and validate tool choices
        sanitized_steps: list[SearchStep] = []
        valid_tools = {"github", "arxiv", "news", "web"}
        for step in plan.search_steps[:step_limit]:
            tool = step.tool.lower().strip()
            if tool not in valid_tools:
                tool = "web"
            if step.query.strip():
                sanitized_steps.append(
                    SearchStep(tool=tool, query=step.query.strip(), reason=step.reason)
                )

        if not sanitized_steps:
            return self._fallback_plan(question, mode)

        return ResearchPlan(
            objective=plan.objective or question,
            sub_questions=plan.sub_questions or [question],
            search_steps=sanitized_steps,
            success_criteria=plan.success_criteria or ["Sufficient technical evidence collected"],
        )

    def _fallback_plan(self, question: str, mode: ResearchMode) -> ResearchPlan:
        """Deterministic plan when LLM is unavailable or unconfigured."""
        keywords = _extract_keywords(question)
        query_str = " ".join(keywords[:4]) if keywords else question

        if mode == ResearchMode.QUICK:
            steps = [
                SearchStep(
                    tool="github", query=query_str, reason="Cari implementasi kode dan framework"
                ),
                SearchStep(tool="arxiv", query=query_str, reason="Cari makalah riset terbaru"),
            ]
            sub_q = [question]
        else:
            steps = [
                SearchStep(
                    tool="github", query=query_str, reason="Cari implementasi kode dan repository"
                ),
                SearchStep(
                    tool="arxiv",
                    query=query_str,
                    reason="Cari publikasi akademis dan metodologi",
                ),
                SearchStep(
                    tool="web",
                    query=f"{query_str} benchmark production",
                    reason="Cari hasil evaluasi dan dokumentasi",
                ),
            ]
            sub_q = [
                f"Bagaimana arsitektur teknis dari {query_str}?",
                f"Bagaimana performa dan benchmark {query_str}?",
                f"Apa keterbatasan dan status kesiapan produksi {query_str}?",
            ]

        return ResearchPlan(
            objective=f"Riset mendalam mengenai: {question}",
            sub_questions=sub_q,
            search_steps=steps,
            success_criteria=[
                "Menemukan implementasi atau arsitektur yang relevan",
                "Mengumpulkan data performa atau bukti empiris",
            ],
        )

    def detect_gaps(
        self,
        plan: ResearchPlan,
        evidence_items: list[dict[str, object]],
        executed_queries: set[str],
        iteration: int = 1,
        max_iterations: int = 1,
    ) -> EvidenceSufficiency:
        """Evaluate evidence coverage against sub-questions and identify knowledge gaps.

        Note: Sufficiency is an intrinsic property of the evidence corpus.
        Budget exhaustion (e.g. reaching max_iterations) NEVER implies sufficiency.
        """
        gaps: list[KnowledgeGap] = []
        if not evidence_items:
            gaps = [
                KnowledgeGap(
                    description=f"Belum ada bukti yang ditemukan untuk: {sub_q}",
                    importance="high",
                    related_sub_question=sub_q,
                    suggested_query=_make_query(sub_q, "overview"),
                    suggested_tool="web",
                )
                for sub_q in plan.sub_questions
                if _make_query(sub_q, "overview") not in executed_queries
            ]
            return EvidenceSufficiency(
                sufficient=False,
                covered_sub_questions=[],
                total_sub_questions=len(plan.sub_questions),
                gaps=gaps,
                reason="Belum ada bukti relevan yang berhasil dikumpulkan.",
            )

        # Concatenate all evidence text to check coverage
        combined_text = " ".join(
            str(item.get("title", ""))
            + " "
            + str(item.get("description", ""))
            + " "
            + str(item.get("abstract", ""))
            + " "
            + str(item.get("readme_excerpt", ""))
            for item in evidence_items
        ).lower()

        covered: list[str] = []
        critical_terms = (
            "production",
            "prod",
            "benchmark",
            "eval",
            "keamanan",
            "security",
            "limitasi",
            "kesiapan",
            "keterbatasan",
            "performa",
        )

        for sub_q in plan.sub_questions:
            sub_keywords = _extract_keywords(sub_q)
            if not sub_keywords:
                covered.append(sub_q)
                continue

            matches = sum(1 for kw in sub_keywords if kw in combined_text)
            coverage_ratio = matches / len(sub_keywords)

            if coverage_ratio >= 0.40:
                covered.append(sub_q)
            else:
                # Sub-question is not adequately covered
                is_academic = any(
                    w in sub_q.lower() for w in ("benchmark", "eval", "teori", "metode")
                )
                suggested_tool = "arxiv" if is_academic else "web"
                suggested_q = _make_query(sub_q, "detail")
                is_critical = any(t in sub_q.lower() for t in critical_terms)
                importance = "high" if (is_critical or iteration == 1) else "medium"

                if suggested_q not in executed_queries:
                    gaps.append(
                        KnowledgeGap(
                            description=f"Bukti belum mencakup aspek: {sub_q}",
                            importance=importance,
                            related_sub_question=sub_q,
                            suggested_query=suggested_q,
                            suggested_tool=suggested_tool,
                        )
                    )

        # Determine sufficiency independently of budget
        total_sub_q = len(plan.sub_questions) or 1
        coverage_pct = len(covered) / total_sub_q
        has_high_gaps = any(g.importance == "high" for g in gaps)
        distinct_sources = len(
            {str(item.get("source_type")) for item in evidence_items if item.get("source_type")}
        )
        has_authoritative = any(
            item.get("authority") in ("primary", "official", "academic") for item in evidence_items
        )

        # Strict sufficiency evaluation
        if has_high_gaps:
            is_sufficient = False
            reason = (
                f"Terdapat {sum(1 for g in gaps if g.importance == 'high')} "
                "knowledge gap prioritas tinggi yang belum terjawab."
            )
        elif len(gaps) == 0 and len(evidence_items) >= 2:
            is_sufficient = True
            reason = (
                f"Semua {total_sub_q} sub-pertanyaan terjawab dengan {len(evidence_items)} "
                f"sumber bukti ({distinct_sources} tipe sumber)."
            )
        elif (
            coverage_pct >= 0.75
            and len(evidence_items) >= 3
            and (distinct_sources >= 2 or has_authoritative)
        ):
            is_sufficient = True
            reason = (
                f"Cakupan sub-pertanyaan mencapai {len(covered)}/{total_sub_q} "
                f"dengan {len(evidence_items)} sumber bukti berotoritas."
            )
        else:
            is_sufficient = False
            reason = (
                f"Bukti belum mencukupi: cakupan {len(covered)}/{total_sub_q} sub-pertanyaan "
                f"dan {len(gaps)} knowledge gap tersisa."
            )

        return EvidenceSufficiency(
            sufficient=is_sufficient,
            covered_sub_questions=covered,
            total_sub_questions=total_sub_q,
            gaps=gaps,
            reason=reason,
        )


def _make_query(text: str, context_type: str) -> str:
    kws = _extract_keywords(text)
    base = " ".join(kws[:4]) if kws else text
    if context_type == "overview":
        return f"{base} overview"
    return base
