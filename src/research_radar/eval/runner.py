"""Evaluation runner executing test cases across research modes and computing metrics."""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from research_radar.consensus.models import ConsensusLevel, ConsensusReport
from research_radar.eval.adversarial import AdversarialEvaluator
from research_radar.eval.metrics import MetricCalculator
from research_radar.eval.models import (
    AggregateComparison,
    CaseExecutionResult,
    EvaluationCase,
    EvaluationDataset,
    EvaluationRunReport,
    ExecutionType,
)
from research_radar.research.models import (
    ConfidenceLevel,
    ResearchBudget,
    ResearchMode,
    ResearchState,
    ResearchStatus,
    ResearchSynthesisResult,
    StopReason,
)
from research_radar.research.orchestrator import ResearchOrchestrator


def _median(values: Sequence[float | int]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _median_nullable(values: Sequence[float | int | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(statistics.median(valid))


class EvaluationRunner:
    """Orchestrates evaluation runs across modes with Live vs Simulated separation."""

    def __init__(
        self,
        orchestrator: ResearchOrchestrator | None = None,
        adversarial_evaluator: AdversarialEvaluator | None = None,
        execution_type: ExecutionType = ExecutionType.SIMULATED,
    ) -> None:
        self.orchestrator = orchestrator
        self.adversarial_evaluator = adversarial_evaluator or AdversarialEvaluator()
        self.execution_type = execution_type

    async def run_case(
        self,
        case: EvaluationCase,
        mode: str = "QUICK",
        user_id: int = 9999,
    ) -> CaseExecutionResult:
        """Execute a single evaluation case under the specified mode."""
        now_str = datetime.now(UTC).isoformat()
        research_mode = ResearchMode.DEEP if mode.upper() == "DEEP" else ResearchMode.QUICK

        start_time = time.perf_counter()

        if self.execution_type == ExecutionType.LIVE:
            if self.orchestrator is None:
                raise ValueError(
                    "Live evaluation requires a configured ResearchOrchestrator; failing closed."
                )
            synth_res = await self.orchestrator.conduct_research(
                question=case.question,
                user_id=user_id,
                mode=research_mode,
            )
            elapsed_time = time.perf_counter() - start_time
            # Update state wall time if needed
            metrics = MetricCalculator.calculate_metrics(
                synth_res,
                case,
                execution_type=ExecutionType.LIVE,
            )
            metrics.wall_time = round(elapsed_time, 2)
        else:
            # Deterministic simulation for test execution without external dependencies
            synth_res = self._simulate_research_result(case, research_mode)
            elapsed_time = time.perf_counter() - start_time
            metrics = MetricCalculator.calculate_metrics(
                synth_res,
                case,
                execution_type=ExecutionType.SIMULATED,
            )

        return CaseExecutionResult(
            execution_type=self.execution_type,
            case_id=case.id,
            mode=mode.upper(),
            metrics=metrics,
            answer_snippet=synth_res.answer[:300],
            safe_conclusion=synth_res.safe_conclusion,
            key_findings=synth_res.key_findings,
            dissenting_findings=synth_res.dissenting_findings,
            confidence=synth_res.confidence.value,
            evidence_sources=synth_res.evidence_sources,
            executed_at=now_str,
        )

    async def run_evaluation(
        self,
        dataset: EvaluationDataset,
        modes: list[str] | None = None,
    ) -> EvaluationRunReport:
        """Run evaluation suite across all cases and modes, generating a structured report."""
        target_modes = [m.upper() for m in (modes or ["QUICK", "DEEP"])]
        quick_results: list[CaseExecutionResult] = []
        deep_results: list[CaseExecutionResult] = []

        for case in dataset.cases:
            if "QUICK" in target_modes:
                res_q = await self.run_case(case, mode="QUICK")
                quick_results.append(res_q)
            if "DEEP" in target_modes:
                res_d = await self.run_case(case, mode="DEEP")
                deep_results.append(res_d)

        adversarial_results = self.adversarial_evaluator.evaluate_all()
        comparison = self.compute_comparison(dataset, quick_results, deep_results)

        categories = sorted({c.category.value for c in dataset.cases})
        prefix = "live" if self.execution_type == ExecutionType.LIVE else "sim"
        evaluation_id = f"{prefix}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        total_runs = len(quick_results) + len(deep_results)

        human_template = [
            {
                "case_id": c.id,
                "question": c.question,
                "category": c.category.value,
                "difficulty": c.difficulty.value,
                "modes": target_modes,
                "rubric_fields": [
                    "relevance (0-2)",
                    "completeness (0-2)",
                    "evidence_quality (0-2)",
                    "conservativeness (0-2)",
                    "usefulness (0-2)",
                    "dissent_handling (0-2)",
                ],
            }
            for c in dataset.cases
        ]

        return EvaluationRunReport(
            schema_version="1.1.0",
            execution_type=self.execution_type,
            evaluation_id=evaluation_id,
            generated_at=datetime.now(UTC).isoformat(),
            dataset_version=dataset.schema_version,
            total_cases=len(dataset.cases),
            total_runs=total_runs,
            categories=categories,
            quick_results=quick_results,
            deep_results=deep_results,
            adversarial_results=adversarial_results,
            comparison=comparison,
            human_rubric_template=human_template,
        )

    def compute_comparison(
        self,
        dataset: EvaluationDataset,
        quick_results: list[CaseExecutionResult],
        deep_results: list[CaseExecutionResult],
    ) -> AggregateComparison:
        """Compute statistical comparisons, category breakdowns, and data-driven recommendations."""
        q_grd = [r.metrics.grounding_rate for r in quick_results]
        d_grd = [r.metrics.grounding_rate for r in deep_results]

        q_cit = [r.metrics.citation_coverage for r in quick_results]
        d_cit = [r.metrics.citation_coverage for r in deep_results]

        q_ev = [r.metrics.evidence_count for r in quick_results]
        d_ev = [r.metrics.evidence_count for r in deep_results]

        q_src = [r.metrics.independent_source_count for r in quick_results]
        d_src = [r.metrics.independent_source_count for r in deep_results]

        q_contra = [r.metrics.contradiction_count for r in quick_results]
        d_contra = [r.metrics.contradiction_count for r in deep_results]

        q_dis = [r.metrics.dissent_count for r in quick_results]
        d_dis = [r.metrics.dissent_count for r in deep_results]

        q_time = [r.metrics.wall_time for r in quick_results]
        d_time = [r.metrics.wall_time for r in deep_results]

        q_tool = [r.metrics.tool_calls for r in quick_results]
        d_tool = [r.metrics.tool_calls for r in deep_results]

        q_llm = [r.metrics.llm_calls for r in quick_results]
        d_llm = [r.metrics.llm_calls for r in deep_results]

        # Token measurements
        tokens_measured = any(r.metrics.tokens_measured for r in quick_results + deep_results)
        q_tok = [r.metrics.total_tokens for r in quick_results if r.metrics.total_tokens]
        d_tok = [r.metrics.total_tokens for r in deep_results if r.metrics.total_tokens]

        # Gold contradiction recall (evaluated strictly on curated gold cases with expected=True)
        gold_cases = [
            c
            for c in dataset.cases
            if c.expected_disagreement is not None and c.expected_disagreement.expected is True
        ]
        gold_case_ids = {c.id for c in gold_cases}

        q_gold_detected = sum(
            1
            for r in quick_results
            if r.case_id in gold_case_ids and r.metrics.contradiction_count > 0
        )
        d_gold_detected = sum(
            1
            for r in deep_results
            if r.case_id in gold_case_ids and r.metrics.contradiction_count > 0
        )

        gold_q_recall = round(q_gold_detected / len(gold_cases), 4) if gold_cases else None
        gold_d_recall = round(d_gold_detected / len(gold_cases), 4) if gold_cases else None

        # Overall descriptive contradiction counts
        contra_detected_q = sum(1 for r in quick_results if r.metrics.contradiction_count > 0)
        contra_detected_d = sum(1 for r in deep_results if r.metrics.contradiction_count > 0)

        # Per category metrics
        per_category: dict[str, dict[str, Any]] = {}
        for cat in {c.category.value for c in dataset.cases}:
            cat_case_ids = {c.id for c in dataset.cases if c.category.value == cat}
            c_q_grd = [r.metrics.grounding_rate for r in quick_results if r.case_id in cat_case_ids]
            c_d_grd = [r.metrics.grounding_rate for r in deep_results if r.case_id in cat_case_ids]
            c_q_ev = [r.metrics.evidence_count for r in quick_results if r.case_id in cat_case_ids]
            c_d_ev = [r.metrics.evidence_count for r in deep_results if r.case_id in cat_case_ids]

            per_category[cat] = {
                "quick_median_grounding": _median_nullable(c_q_grd),
                "deep_median_grounding": _median_nullable(c_d_grd),
                "quick_median_evidence": _median(c_q_ev),
                "deep_median_evidence": _median(c_d_ev),
            }

        # Per difficulty metrics
        per_difficulty: dict[str, dict[str, Any]] = {}
        for diff in {c.difficulty.value for c in dataset.cases}:
            diff_case_ids = {c.id for c in dataset.cases if c.difficulty.value == diff}
            d_q_ev = [r.metrics.evidence_count for r in quick_results if r.case_id in diff_case_ids]
            d_d_ev = [r.metrics.evidence_count for r in deep_results if r.case_id in diff_case_ids]
            d_q_contra = [
                r.metrics.contradiction_count for r in quick_results if r.case_id in diff_case_ids
            ]
            d_d_contra = [
                r.metrics.contradiction_count for r in deep_results if r.case_id in diff_case_ids
            ]

            per_difficulty[diff] = {
                "quick_median_evidence": _median(d_q_ev),
                "deep_median_evidence": _median(d_d_ev),
                "quick_median_contradictions": _median(d_q_contra),
                "deep_median_contradictions": _median(d_d_contra),
            }

        # Failure distribution
        failure_counts: dict[str, int] = {}
        for r in quick_results + deep_results:
            for tag in r.metrics.failure_tags:
                tag_name = tag.value if hasattr(tag, "value") else str(tag)
                failure_counts[tag_name] = failure_counts.get(tag_name, 0) + 1

        # Dynamic data-driven routing analysis based on observed empirical deltas
        routing: list[str] = []
        ev_delta = _median(d_ev) - _median(q_ev)
        src_delta = _median(d_src) - _median(q_src)
        time_delta = _median(d_time) - _median(q_time)

        if ev_delta >= 2.0 and src_delta >= 1.0:
            routing.append(
                f"EVIDENCE DEPTH: DEEP mode yielded +{ev_delta:.1f} more evidence items and "
                f"+{src_delta:.1f} independent sources."
            )
        else:
            routing.append(
                "EVIDENCE DEPTH: Evidence volume difference between QUICK and DEEP was modest."
            )

        if (gold_d_recall or 0.0) > (gold_q_recall or 0.0):
            routing.append(
                f"CONTRADICTION DISCOVERY: DEEP mode improved gold contradiction recall "
                f"from {(gold_q_recall or 0.0):.1%} (QUICK) to {(gold_d_recall or 0.0):.1%} (DEEP)."
            )
        else:
            routing.append(
                "CONTRADICTION DISCOVERY: QUICK and DEEP showed similar contradiction recall."
            )

        if time_delta > 0:
            routing.append(
                f"LATENCY OVERHEAD: DEEP mode required +{time_delta:.2f}s latency and "
                f"+{(_median(d_tool) - _median(q_tool)):.1f} additional tool calls."
            )

        routing.append(
            "CALIBRATION: QUICK mode is recommended for EASY factual queries to conserve latency. "
            "DEEP mode is recommended for CONTROVERSIAL, HARD, or comparative queries."
        )

        return AggregateComparison(
            execution_type=self.execution_type,
            total_cases=len(dataset.cases),
            quick_median_grounding=_median_nullable(q_grd),
            deep_median_grounding=_median_nullable(d_grd),
            quick_median_citations=_median_nullable(q_cit),
            deep_median_citations=_median_nullable(d_cit),
            quick_median_evidence=_median(q_ev),
            deep_median_evidence=_median(d_ev),
            quick_median_sources=_median(q_src),
            deep_median_sources=_median(d_src),
            quick_median_contradictions=_median(q_contra),
            deep_median_contradictions=_median(d_contra),
            quick_median_dissent=_median(q_dis),
            deep_median_dissent=_median(d_dis),
            quick_median_wall_time=_median(q_time),
            deep_median_wall_time=_median(d_time),
            quick_median_tool_calls=_median(q_tool),
            deep_median_tool_calls=_median(d_tool),
            quick_median_llm_calls=_median(q_llm),
            deep_median_llm_calls=_median(d_llm),
            tokens_measured=tokens_measured,
            quick_median_tokens=int(_median(q_tok)) if q_tok else None,
            deep_median_tokens=int(_median(d_tok)) if d_tok else None,
            gold_contradiction_recall_quick=gold_q_recall,
            gold_contradiction_recall_deep=gold_d_recall,
            contradictions_detected_quick=contra_detected_q,
            contradictions_detected_deep=contra_detected_d,
            per_category=per_category,
            per_difficulty=per_difficulty,
            failure_distribution=failure_counts,
            routing_recommendations=routing,
        )

    def _simulate_research_result(
        self,
        case: EvaluationCase,
        mode: ResearchMode,
    ) -> ResearchSynthesisResult:
        """Deterministic simulation of research execution for benchmark runs."""
        is_deep = mode == ResearchMode.DEEP
        evidence_count = 5 if is_deep else 2
        tool_calls = 4 if is_deep else 1
        iterations = 2 if is_deep else 1
        llm_calls = 2 if is_deep else 0

        # Build simulated evidence items
        evidence_sources: list[dict[str, Any]] = []
        for i in range(evidence_count):
            auth = "primary" if i == 0 else ("academic" if i == 1 else "secondary")
            evidence_sources.append(
                {
                    "url": f"https://source-{i}.org/{case.id}",
                    "title": f"Evidence {i} for {case.id}",
                    "source_type": "github" if i % 2 == 0 else "arxiv",
                    "authority": auth,
                }
            )

        # Build simulated consensus report
        has_disagreement = (
            case.expected_disagreement is not None
            and case.expected_disagreement.expected
            and is_deep
        )
        consensus_report = ConsensusReport(
            overall_consensus=ConsensusLevel.MIXED if has_disagreement else ConsensusLevel.STRONG,
            agreements=[f"Dukungan spesifikasi terverifikasi untuk {case.id}."],
            disagreements=[f"Terdapat perbedaan benchmark pada {case.id}."]
            if has_disagreement
            else [],
            dissenting_findings=[f"Studi independen menunjukkan batasan pada {case.id}."]
            if has_disagreement
            else [],
        )

        state = ResearchState(
            research_id=f"sim_res_{case.id}_{mode.value}",
            run_id=f"sim_run_{case.id}",
            question=case.question,
            mode=mode,
            status=ResearchStatus.COMPLETED,
            stop_reason=StopReason.ENOUGH_EVIDENCE,
            iterations=iterations,
            tool_calls=tool_calls,
            llm_calls=llm_calls,
            budget=ResearchBudget.for_mode(mode),
            started_at=datetime(2026, 8, 28, 10, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 8, 28, 10, 0, 5 if is_deep else 1, tzinfo=UTC),
        )

        return ResearchSynthesisResult(
            research_id=state.research_id,
            run_id=state.run_id,
            question=case.question,
            mode=mode,
            answer=f"Hasil sintesis riset terstruktur untuk: {case.question}",
            key_findings=[f"Temuan utama 1 untuk {case.id}", f"Temuan utama 2 untuk {case.id}"],
            evidence_sources=evidence_sources,
            uncertainties=[f"Ketidakpastian operasional pada {case.id}"]
            if has_disagreement
            else [],
            remaining_gaps=[],
            confidence=ConfidenceLevel.MEDIUM if has_disagreement else ConfidenceLevel.HIGH,
            stop_reason=StopReason.ENOUGH_EVIDENCE,
            consensus=consensus_report.overall_consensus,
            agreements=consensus_report.agreements,
            disagreements=consensus_report.disagreements,
            dissenting_findings=consensus_report.dissenting_findings,
            safe_conclusion=f"Disarankan evaluasi bertahap untuk {case.id}.",
            consensus_report=consensus_report,
            state=state,
        )
