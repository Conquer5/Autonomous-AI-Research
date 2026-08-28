"""Report generator producing Markdown summaries and JSON payloads for evaluation runs."""

from __future__ import annotations

import json
from typing import Any

from research_radar.eval.models import EvaluationDataset, EvaluationRunReport, ExecutionType


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.1%}"


def _fmt_delta_pct(v1: float | None, v2: float | None) -> str:
    if v1 is None or v2 is None:
        return "N/A"
    delta = v2 - v1
    return f"{delta:+.1%}"


class ReportGenerator:
    """Generates structured Markdown and JSON reports from evaluation execution runs."""

    @staticmethod
    def generate_markdown(report: EvaluationRunReport) -> str:
        """Generate human-readable GitHub Flavored Markdown report."""
        comp = report.comparison
        is_live = report.execution_type == ExecutionType.LIVE
        lines: list[str] = []

        lines.append(f"# Research Radar — Evaluation & Calibration Report ({report.evaluation_id})")
        lines.append("")

        if is_live:
            lines.append("> 🔬 **LIVE RESEARCH EXECUTION — REAL-WORLD CALIBRATION DATA**")
        else:
            lines.append("> ⚠️ **SIMULATED — NOT REAL-WORLD PERFORMANCE DATA**")

        lines.append("")
        lines.append(f"- **Execution Type:** `{report.execution_type.value}`")
        lines.append(f"- **Generated At:** `{report.generated_at}`")
        lines.append(f"- **Dataset Version:** `{report.dataset_version}`")
        lines.append(
            f"- **Evaluated:** `{report.total_cases}` cases ({report.total_runs} total runs) "
            f"across `{len(report.categories)}` categories"
        )
        lines.append("")

        # 1. Executive Summary Table
        lines.append("## 1. Research Mode Comparison (QUICK vs DEEP)")
        lines.append("")
        lines.append("| Metric | QUICK Mode | DEEP Mode | Delta / Improvement |")
        lines.append("|---|---|---|---|")

        lines.append(
            f"| **Median Grounding Rate** | `{_fmt_pct(comp.quick_median_grounding)}` | "
            f"`{_fmt_pct(comp.deep_median_grounding)}` | "
            f"`{_fmt_delta_pct(comp.quick_median_grounding, comp.deep_median_grounding)}` |"
        )

        lines.append(
            f"| **Median Citation Coverage** | `{_fmt_pct(comp.quick_median_citations)}` | "
            f"`{_fmt_pct(comp.deep_median_citations)}` | "
            f"`{_fmt_delta_pct(comp.quick_median_citations, comp.deep_median_citations)}` |"
        )

        d_ev = comp.deep_median_evidence - comp.quick_median_evidence
        lines.append(
            f"| **Median Evidence Count** | `{comp.quick_median_evidence:.1f}` | "
            f"`{comp.deep_median_evidence:.1f}` | `{d_ev:+.1f}` items |"
        )

        d_src = comp.deep_median_sources - comp.quick_median_sources
        lines.append(
            f"| **Median Independent Sources** | `{comp.quick_median_sources:.1f}` | "
            f"`{comp.deep_median_sources:.1f}` | `{d_src:+.1f}` sources |"
        )

        d_contra = comp.deep_median_contradictions - comp.quick_median_contradictions
        lines.append(
            f"| **Median Contradictions Found** | `{comp.quick_median_contradictions:.1f}` | "
            f"`{comp.deep_median_contradictions:.1f}` | `{d_contra:+.1f}` |"
        )

        g_q = _fmt_pct(comp.gold_contradiction_recall_quick)
        g_d = _fmt_pct(comp.gold_contradiction_recall_deep)
        g_delta = _fmt_delta_pct(
            comp.gold_contradiction_recall_quick, comp.gold_contradiction_recall_deep
        )
        lines.append(f"| **Gold Contradiction Recall** | `{g_q}` | `{g_d}` | `{g_delta}` |")

        d_dis = comp.deep_median_dissent - comp.quick_median_dissent
        lines.append(
            f"| **Median Dissenting Points** | `{comp.quick_median_dissent:.1f}` | "
            f"`{comp.deep_median_dissent:.1f}` | `{d_dis:+.1f}` |"
        )

        d_tool = comp.deep_median_tool_calls - comp.quick_median_tool_calls
        lines.append(
            f"| **Median Tool Calls** | `{comp.quick_median_tool_calls:.1f}` | "
            f"`{comp.deep_median_tool_calls:.1f}` | `{d_tool:+.1f}` calls |"
        )

        d_llm = comp.deep_median_llm_calls - comp.quick_median_llm_calls
        lines.append(
            f"| **Median LLM Calls** | `{comp.quick_median_llm_calls:.1f}` | "
            f"`{comp.deep_median_llm_calls:.1f}` | `{d_llm:+.1f}` calls |"
        )

        d_time = comp.deep_median_wall_time - comp.quick_median_wall_time
        lines.append(
            f"| **Median Latency (Wall Time)** | `{comp.quick_median_wall_time:.2f}s` | "
            f"`{comp.deep_median_wall_time:.2f}s` | `{d_time:+.2f}s` |"
        )

        if comp.tokens_measured:
            q_tok = comp.quick_median_tokens or 0
            d_tok = comp.deep_median_tokens or 0
            lines.append(
                f"| **Median Total Tokens** | `{q_tok}` | `{d_tok}` | `{d_tok - q_tok:+d}` tokens |"
            )
        else:
            lines.append("| **Token Usage** | `NOT_MEASURED` | `NOT_MEASURED` | `N/A` |")

        lines.append("")

        # 2. Category Breakdown
        lines.append("## 2. Per-Category Breakdown")
        lines.append("")
        lines.append(
            "| Category | QUICK Evidence | DEEP Evidence | QUICK Grounding | DEEP Grounding |"
        )
        lines.append("|---|---|---|---|---|")
        for cat, stats in sorted(comp.per_category.items()):
            q_e = stats.get("quick_median_evidence", 0.0)
            d_e = stats.get("deep_median_evidence", 0.0)
            q_g = _fmt_pct(stats.get("quick_median_grounding"))
            d_g = _fmt_pct(stats.get("deep_median_grounding"))
            lines.append(f"| `{cat}` | `{q_e:.1f}` | `{d_e:.1f}` | `{q_g}` | `{d_g}` |")
        lines.append("")

        # 3. Difficulty Breakdown
        lines.append("## 3. Per-Difficulty Breakdown")
        lines.append("")
        lines.append(
            "| Difficulty | QUICK Ev | DEEP Ev | QUICK Contradictions | DEEP Contradictions |"
        )
        lines.append("|---|---|---|---|---|")
        for diff, stats in sorted(comp.per_difficulty.items()):
            q_e = stats.get("quick_median_evidence", 0.0)
            d_e = stats.get("deep_median_evidence", 0.0)
            q_c = stats.get("quick_median_contradictions", 0.0)
            d_c = stats.get("deep_median_contradictions", 0.0)
            lines.append(f"| `{diff}` | `{q_e:.1f}` | `{d_e:.1f}` | `{q_c:.1f}` | `{d_c:.1f}` |")
        lines.append("")

        # 4. Adversarial Fixture Results
        if report.adversarial_results:
            lines.append("## 4. Adversarial Fixture Results")
            lines.append("")
            lines.append("| Fixture | Contradiction Detected | Dissent Preserved | Status |")
            lines.append("|---|---|---|---|")
            for adv in report.adversarial_results:
                status_emoji = "✅ PASS" if adv.passed else "❌ FAIL"
                c_det = "Yes" if adv.contradiction_detected else "No"
                d_pres = "Yes" if adv.dissent_preserved else "No"
                lines.append(
                    f"| **{adv.fixture_name}** | `{c_det}` | `{d_pres}` | {status_emoji} |"
                )
            lines.append("")

        # 5. Failure Taxonomy Distribution
        lines.append("## 5. Observed Failure Taxonomy")
        lines.append("")
        if comp.failure_distribution:
            lines.append("| Failure Category | Occurrence Count | Description |")
            lines.append("|---|---|---|")
            for fail_tag, count in sorted(comp.failure_distribution.items(), key=lambda x: -x[1]):
                lines.append(f"| `{fail_tag}` | `{count}` | Diagnostic error category |")
        else:
            lines.append("No active failures recorded during this benchmark run.")
        lines.append("")

        # 6. Routing Recommendations
        lines.append("## 6. Routing & Calibration Recommendations")
        lines.append("")
        for rec in comp.routing_recommendations:
            lines.append(f"- {rec}")
        lines.append("")

        # 7. Human Review Rubric Guidelines
        lines.append("## 7. Human Evaluation Rubric Guidelines")
        lines.append("")
        lines.append("Score each category from **0 (poor)** to **2 (good)**:")
        lines.append("- **RELEVANCE (0-2):** Does the answer address the core question directly?")
        lines.append("- **COMPLETENESS (0-2):** Does it address important trade-offs?")
        lines.append("- **EVIDENCE QUALITY (0-2):** Are sources authoritative?")
        lines.append("- **CONSERVATIVENESS (0-2):** Does it avoid ungrounded claims?")
        lines.append("- **USEFULNESS (0-2):** Would this enable an engineering decision?")
        lines.append("- **DISSENT HANDLING (0-2):** Are conflicting benchmarks preserved?")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def generate_human_review_package(
        report: EvaluationRunReport,
        dataset: EvaluationDataset,
    ) -> dict[str, Any]:
        """Generate human review packet with blank rubric fields for manual evaluation."""
        case_map = {c.id: c for c in dataset.cases}
        quick_map = {r.case_id: r for r in report.quick_results}
        deep_map = {r.case_id: r for r in report.deep_results}

        items: list[dict[str, Any]] = []
        for case_id in sorted(set(quick_map.keys()) | set(deep_map.keys())):
            case = case_map.get(case_id)
            q_res = quick_map.get(case_id)
            d_res = deep_map.get(case_id)

            item = {
                "case_id": case_id,
                "question": case.question if case else "Unknown",
                "category": case.category.value if case else "unknown",
                "difficulty": case.difficulty.value if case else "unknown",
                "mode_A": {
                    "mode_name": "QUICK",
                    "answer_snippet": q_res.answer_snippet if q_res else "",
                    "safe_conclusion": q_res.safe_conclusion if q_res else "",
                    "sources": [
                        {"title": s.get("title", ""), "url": s.get("url", "")}
                        for s in (q_res.evidence_sources if q_res else [])
                    ],
                    "rubric_scores": {
                        "relevance": None,
                        "completeness": None,
                        "evidence_quality": None,
                        "conservativeness": None,
                        "usefulness": None,
                        "dissent_handling": None,
                        "evaluator_notes": "",
                    },
                },
                "mode_B": {
                    "mode_name": "DEEP",
                    "answer_snippet": d_res.answer_snippet if d_res else "",
                    "safe_conclusion": d_res.safe_conclusion if d_res else "",
                    "sources": [
                        {"title": s.get("title", ""), "url": s.get("url", "")}
                        for s in (d_res.evidence_sources if d_res else [])
                    ],
                    "rubric_scores": {
                        "relevance": None,
                        "completeness": None,
                        "evidence_quality": None,
                        "conservativeness": None,
                        "usefulness": None,
                        "dissent_handling": None,
                        "evaluator_notes": "",
                    },
                },
            }
            items.append(item)

        return {
            "evaluation_id": report.evaluation_id,
            "generated_at": report.generated_at,
            "total_cases_for_review": len(items),
            "cases": items,
        }

    @staticmethod
    def generate_json(report: EvaluationRunReport) -> str:
        """Generate machine-readable JSON payload."""
        return json.dumps(report.model_dump(), indent=2, ensure_ascii=False)
