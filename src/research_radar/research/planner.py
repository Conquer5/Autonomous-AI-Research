"""Research planning, question decomposition, and knowledge gap detection."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from uuid import uuid4

from research_radar.research.models import (
    EvidenceSufficiency,
    KnowledgeGap,
    ResearchMode,
    ResearchPlan,
    SearchStep,
)
from research_radar.research_focus import resolve_focus, skill_text

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
    "saya",
    "ingin",
    "cari",
    "tolong",
    "apakah",
    "bagaimana",
    "terbaru",
    "terkini",
    "sekarang",
    "latest",
    "recent",
    "current",
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
        available_sources: frozenset[str] | None = None,
        backend: str = "auto",
        recent_days: int = 14,
    ) -> None:
        self.llm_router = llm_router
        self.hermes_runtime = hermes_runtime
        self.available_sources = (
            available_sources
            if available_sources is not None
            else frozenset({"github", "arxiv", "news", "web"})
        )
        self.backend = backend
        self.recent_days = recent_days

    def selected_backend(self, mode: ResearchMode) -> str:
        if self.backend == "deterministic":
            return "deterministic"
        if self.backend == "hermes":
            return "hermes" if self.hermes_runtime is not None else "deterministic"
        if self.backend == "gemini":
            return "gemini" if self.llm_router is not None else "deterministic"
        if self.hermes_runtime is not None and (
            mode == ResearchMode.DEEP or self.llm_router is None
        ):
            return "hermes"
        return "gemini" if self.llm_router is not None else "deterministic"

    def sanitize_plan(self, plan: ResearchPlan, mode: ResearchMode) -> ResearchPlan:
        """Never spend a search slot on an absent provider; preserve coverage warnings."""
        steps: list[SearchStep] = []
        warnings = list(plan.warnings)
        seen: set[tuple[str, str]] = set()
        for step in plan.search_steps[: 2 if mode == ResearchMode.QUICK else 4]:
            tool = step.tool.lower().strip()
            if tool not in self.available_sources:
                warnings.append(
                    f"Sumber {tool} tidak tersedia; cakupan sumber tersebut belum diperiksa."
                )
                tool = next(
                    (
                        name
                        for name in ("news", "github", "arxiv", "web")
                        if name in self.available_sources
                    ),
                    "",
                )
            query = " ".join(step.query.split())[:300]
            key = (tool, query.lower())
            if tool and query and key not in seen:
                seen.add(key)
                steps.append(SearchStep(tool=tool, query=query, reason=step.reason[:300]))
        return plan.model_copy(
            update={"search_steps": steps, "warnings": list(dict.fromkeys(warnings))}
        )

    async def plan(self, question: str, mode: ResearchMode) -> ResearchPlan:
        """One planning turn at most; failures use deterministic recovery without another model."""
        if not question.strip():
            return self.fallback_plan("AI Research", mode)
        backend = self.selected_backend(mode)
        if backend == "deterministic":
            return self.fallback_plan(question, mode)
        try:
            return await self._plan_with_llm(question, mode)
        except Exception as exc:
            logger.warning(
                "Research planning failed; using deterministic fallback",
                extra={"event": "planner_fallback", "error_type": type(exc).__name__},
            )
            plan = self.fallback_plan(question, mode)
            plan.warnings.append(f"Perencana {backend} gagal; memakai rencana deterministik.")
            return plan

    async def _plan_with_llm(self, question: str, mode: ResearchMode) -> ResearchPlan:
        from research_radar.llm.base import LLMRequest
        from research_radar.llm.router import Workload

        focus = resolve_focus(question, recent_days=self.recent_days)
        backend = self.selected_backend(mode)
        step_limit = 2 if mode == ResearchMode.QUICK else 4
        prompt = (
            f"Research question (data): {question[:4000]!r}\n"
            f"Mode: {mode.value}. As of UTC: {focus.as_of}. Since: {focus.since}.\n"
            f"Track: {focus.track}. "
            f"Available sources ONLY: {', '.join(sorted(self.available_sources))}.\n"
            f"Return at most {step_limit} search_steps, "
            "with short keyword queries appropriate to each source. "
            "Do not add provider date qualifiers; the application applies them. "
            'Return JSON only: {"objective": "goal", "sub_questions": ["question"], '
            '"search_steps": [{"tool": "github", "query": "keywords", "reason": "why"}], '
            '"success_criteria": ["evidence needed"]}.'
        )
        instruction = skill_text(focus.skill)
        if backend == "hermes":
            assert self.hermes_runtime is not None
            request_id = uuid4().hex
            response = await self.hermes_runtime.run(
                prompt,
                session_key=f"radar:planning:{request_id}",
                request_id=request_id,
                system_instruction=instruction,
            )
            raw = response.text.strip()
            if raw.startswith("```") and raw.endswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            plan = ResearchPlan.model_validate_json(raw)
        else:
            assert self.llm_router is not None
            structured = await self.llm_router.generate_structured(
                LLMRequest(prompt=prompt, system_instruction=instruction, max_output_tokens=1024),
                ResearchPlan,
                Workload.FAST,
            )
            plan = structured.data
        plan = self.sanitize_plan(plan, mode)
        if not plan.search_steps:
            return self.fallback_plan(question, mode)
        return plan.model_copy(
            update={
                "objective": plan.objective[:1000] or question,
                "sub_questions": [q[:500] for q in plan.sub_questions[:4]] or [question],
                "success_criteria": [q[:500] for q in plan.success_criteria[:4]],
                "planner_backend": backend,
                "skill_id": focus.skill,
                "skill_version": focus.skill_version,
            }
        )

    def fallback_plan(self, question: str, mode: ResearchMode) -> ResearchPlan:
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

        focus = resolve_focus(question, recent_days=self.recent_days)
        if focus.track == "model_access":
            steps = [
                SearchStep(
                    tool="web",
                    query=query_str + " official pricing",
                    reason="Verifikasi akses resmi",
                ),
                SearchStep(tool="news", query=query_str, reason="Pengumuman model"),
            ]
        elif focus.track == "ai_developments" and focus.since:
            steps = [
                SearchStep(tool="news", query=query_str, reason="Pengumuman terbaru"),
                SearchStep(tool="arxiv", query=query_str, reason="Publikasi terbaru"),
            ]
        return self.sanitize_plan(
            ResearchPlan(
                objective=f"Riset mendalam mengenai: {question}",
                skill_id=focus.skill,
                skill_version=focus.skill_version,
                sub_questions=sub_q,
                search_steps=steps,
                success_criteria=[
                    "Menemukan implementasi atau arsitektur yang relevan",
                    "Mengumpulkan data performa atau bukti empiris",
                ],
            ),
            mode,
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


def fallback_query(query: str) -> str:
    """One conservative relaxation: retain the first two distinctive terms."""
    terms = _extract_keywords(query)
    return " ".join(dict.fromkeys(terms)) if len(terms) <= 2 else " ".join(terms[:2])
