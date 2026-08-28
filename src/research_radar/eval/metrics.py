"""Metric calculation and failure classification for research evaluations."""

from __future__ import annotations

import re
from typing import Any

from research_radar.consensus.engine import _extract_domain_owner
from research_radar.consensus.models import ConsensusReport, ContradictionSeverity
from research_radar.eval.models import (
    EvaluationCase,
    EvaluationMetrics,
    ExecutionType,
    FailureCategory,
)
from research_radar.evidence.models import ClaimType, VerificationStatus
from research_radar.research.models import ResearchSynthesisResult, StopReason


def _extract_source_origin(item: dict[str, Any]) -> str:
    """Extract independent information origin using P1B independence logic."""
    url = str(item.get("url", ""))
    domain = _extract_domain_owner(url)
    title = str(item.get("title", "")).lower()

    # Detect syndicated / wire attribution in title
    attr_match = re.search(r"(?:citing|according to|via|source:|dari)\s+([a-z0-9_\-\.]+)", title)
    if attr_match:
        upstream = attr_match.group(1).strip(".")
        return f"syndicated:{upstream}"
    return domain


class MetricCalculator:
    """Calculates objective metrics and classifies failures from research execution results."""

    @staticmethod
    def calculate_metrics(
        result: ResearchSynthesisResult,
        case: EvaluationCase | None = None,
        *,
        verification_report: Any | None = None,
        execution_type: ExecutionType = ExecutionType.SIMULATED,
    ) -> EvaluationMetrics:
        """Extract and compute all evaluation metrics from a synthesis result."""
        state = result.state
        evidence_items = result.evidence_sources or []
        consensus_report = result.consensus_report or ConsensusReport()

        # 1. Evidence & Information Independence Metrics (reusing P1B logic)
        evidence_count = len(evidence_items)
        origins = {_extract_source_origin(item) for item in evidence_items}
        origins.discard("unknown")
        independent_source_count = len(origins) if origins else (1 if evidence_count > 0 else 0)

        source_types = {
            str(item.get("source_type", "")) for item in evidence_items if item.get("source_type")
        }
        source_type_count = len(source_types)

        authoritative_source_count = sum(
            1
            for item in evidence_items
            if str(item.get("authority", "")).lower() in ("primary", "academic", "official")
        )

        # 2. Strict Verification / Grounding Metrics (P0 Verifier Grounded)
        supported_fact_count = 0
        unsupported_fact_count = 0
        partially_supported_count = 0
        factual_claims_with_evidence = 0
        total_factual_claims = 0
        grounding_rate: float | None = None
        citation_coverage: float | None = None
        grounding_measured = False
        citations_measured = False

        if verification_report is not None:
            total_factual_claims = getattr(verification_report, "total_claims", 0)
            supported_fact_count = getattr(verification_report, "supported_claims", 0)
            partially_supported_count = getattr(
                verification_report, "partially_supported_claims", 0
            )
            unsupported_fact_count = getattr(verification_report, "unsupported_claims", 0)
            factual_claims_with_evidence = sum(
                1 for r in getattr(verification_report, "results", []) if r.claim.evidence_ids
            )
            grounding_rate = (
                (supported_fact_count + 0.5 * partially_supported_count) / total_factual_claims
                if total_factual_claims > 0
                else (1.0 if evidence_count > 0 else 0.0)
            )
            citation_coverage = (
                factual_claims_with_evidence / total_factual_claims
                if total_factual_claims > 0
                else (1.0 if evidence_count > 0 else 0.0)
            )
            grounding_measured = True
            citations_measured = True
        elif hasattr(result, "claims") and result.claims:
            for c in result.claims:
                if getattr(c, "claim_type", None) == ClaimType.FACT:
                    total_factual_claims += 1
                    if getattr(c, "evidence_ids", None):
                        factual_claims_with_evidence += 1
                    status = getattr(c, "verification_status", VerificationStatus.UNSUPPORTED)
                    if status == VerificationStatus.SUPPORTED:
                        supported_fact_count += 1
                    elif status == VerificationStatus.PARTIALLY_SUPPORTED:
                        partially_supported_count += 1
                    elif status == VerificationStatus.UNSUPPORTED:
                        unsupported_fact_count += 1

            grounding_rate = (
                (supported_fact_count + 0.5 * partially_supported_count) / total_factual_claims
                if total_factual_claims > 0
                else (1.0 if evidence_count > 0 else 0.0)
            )
            citation_coverage = (
                factual_claims_with_evidence / total_factual_claims
                if total_factual_claims > 0
                else (1.0 if evidence_count > 0 else 0.0)
            )
            grounding_measured = True
            citations_measured = True
        else:
            # LIVE / strict mode: if verification was not performed, mark as unmeasured
            grounding_rate = None
            citation_coverage = None
            grounding_measured = False
            citations_measured = False

        # 3. Consensus & Contradiction Metrics
        claim_cluster_count = len(consensus_report.assessments)
        contradiction_count = len(consensus_report.contradictions)
        high_severity_contradiction_count = sum(
            1 for c in consensus_report.contradictions if c.severity == ContradictionSeverity.HIGH
        )
        dissent_count = len(consensus_report.dissenting_findings)
        consensus_level_str = (
            consensus_report.overall_consensus.value
            if hasattr(consensus_report.overall_consensus, "value")
            else str(consensus_report.overall_consensus)
        )

        # 4. Research Process Metrics
        iterations = state.iterations
        tool_calls = state.tool_calls
        queries = len(state.queries_executed)
        llm_calls = state.llm_calls
        stop_reason_str = state.stop_reason.value if state.stop_reason else "unknown"

        wall_time = 0.0
        if state.completed_at and state.started_at:
            wall_time = max(0.0, (state.completed_at - state.started_at).total_seconds())

        # 5. Token Usage (measured strictly if provider returned metadata)
        tokens_measured = False
        input_tokens: int | None = None
        output_tokens: int | None = None
        total_tokens: int | None = None
        if hasattr(state, "token_usage") and state.token_usage:
            tu = state.token_usage
            input_tokens = tu.get("input_tokens")
            output_tokens = tu.get("output_tokens")
            total_tokens = tu.get("total_tokens")
            tokens_measured = total_tokens is not None

        # 6. Reliability Metrics
        research_status_str = state.status.value
        fallback_used = "Sintesis terdegradasi" in result.answer or any(
            "kendala" in u.lower() for u in result.uncertainties
        )
        collector_failures = sum(
            1 for q in state.queries_executed if getattr(q, "status", "") == "failed"
        )

        metrics = EvaluationMetrics(
            execution_type=execution_type,
            supported_fact_count=supported_fact_count,
            unsupported_fact_count=unsupported_fact_count,
            partially_supported_count=partially_supported_count,
            grounding_rate=round(grounding_rate, 4) if grounding_rate is not None else None,
            grounding_measured=grounding_measured,
            factual_claims_with_evidence=factual_claims_with_evidence,
            total_factual_claims=total_factual_claims,
            citation_coverage=(
                round(citation_coverage, 4) if citation_coverage is not None else None
            ),
            citations_measured=citations_measured,
            evidence_count=evidence_count,
            independent_source_count=independent_source_count,
            source_type_count=source_type_count,
            authoritative_source_count=authoritative_source_count,
            claim_cluster_count=claim_cluster_count,
            contradiction_count=contradiction_count,
            high_severity_contradiction_count=high_severity_contradiction_count,
            dissent_count=dissent_count,
            consensus_level=consensus_level_str,
            iterations=iterations,
            tool_calls=tool_calls,
            queries=queries,
            llm_calls=llm_calls,
            stop_reason=stop_reason_str,
            wall_time=round(wall_time, 2),
            tokens_measured=tokens_measured,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            research_status=research_status_str,
            fallback_used=fallback_used,
            collector_failures=collector_failures,
        )

        # 7. Failure Tag Classification
        metrics.failure_tags = MetricCalculator.classify_failures(metrics, case, state)
        return metrics

    @staticmethod
    def classify_failures(
        metrics: EvaluationMetrics,
        case: EvaluationCase | None,
        state: Any | None = None,
    ) -> list[FailureCategory]:
        """Classify failure categories based on evaluation thresholds and case characteristics."""
        failures: list[FailureCategory] = []

        if metrics.evidence_count == 0:
            failures.append(FailureCategory.RETRIEVAL_FAILURE)

        if case is not None:
            if (
                case.characteristics.requires_primary_sources
                and metrics.authoritative_source_count == 0
                and metrics.evidence_count > 0
            ):
                failures.append(FailureCategory.SOURCE_QUALITY_FAILURE)

            if (
                case.characteristics.likely_disagreement
                and metrics.contradiction_count == 0
                and metrics.evidence_count >= 3
            ):
                failures.append(FailureCategory.CONTRADICTION_FAILURE)

        if metrics.unsupported_fact_count > 0:
            failures.append(FailureCategory.VERIFICATION_FAILURE)

        if metrics.fallback_used:
            failures.append(FailureCategory.SYNTHESIS_FAILURE)

        if state is not None:
            if getattr(state, "stop_reason", None) == StopReason.TIME_BUDGET_EXHAUSTED:
                failures.append(FailureCategory.TIME_BUDGET)

        return failures
