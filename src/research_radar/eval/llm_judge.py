"""Secondary LLM-as-a-Judge evaluator for research outputs with explicit labeling."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from research_radar.eval.models import EvaluationCase, LLMJudgeScore
from research_radar.llm.base import LLMRequest
from research_radar.llm.router import LLMRouter, Workload

logger = logging.getLogger(__name__)


class LLMJudgeStructuredOutput(BaseModel):
    relevance_score: float = Field(ge=0.0, le=10.0)
    grounding_score: float = Field(ge=0.0, le=10.0)
    conservativeness_score: float = Field(ge=0.0, le=10.0)
    dissent_score: float = Field(ge=0.0, le=10.0)
    reasoning: str = ""


class LLMJudge:
    """Secondary automated judge evaluating research quality against a structured rubric."""

    def __init__(
        self,
        llm_router: LLMRouter | None = None,
        *,
        allow_heuristic_fallback: bool = False,
    ) -> None:
        self.llm_router = llm_router
        self.allow_heuristic_fallback = allow_heuristic_fallback

    async def evaluate_output(
        self,
        case: EvaluationCase,
        mode: str,
        answer: str,
        evidence_sources: list[dict[str, Any]],
        safe_conclusion: str = "",
    ) -> LLMJudgeScore | None:
        """Evaluate a single research output using LLM-as-judge or return None if unrun."""
        now_str = datetime.now(UTC).isoformat()

        if self.llm_router is not None:
            try:
                sources_summary = "\n".join(
                    f"- {s.get('title', 'Source')}: {s.get('url', '')} "
                    f"(Authority: {s.get('authority', 'unknown')})"
                    for s in evidence_sources[:6]
                )
                prompt = (
                    f"Evaluate the following research answer against the question and evidence.\n\n"
                    f"Question: {case.question}\n"
                    f"Category: {case.category.value}\n"
                    f"Expected Characteristics: {case.characteristics.model_dump()}\n\n"
                    f"Evidence Sources:\n{sources_summary}\n\n"
                    f"Research Answer:\n{answer}\n\n"
                    f"Safe Conclusion:\n{safe_conclusion}\n\n"
                    "Score the output on a scale of 0.0 to 10.0 for:\n"
                    "1. relevance_score: Does it directly answer the question?\n"
                    "2. grounding_score: Are claims supported by the evidence?\n"
                    "3. conservativeness_score: Does it avoid speculative assertions?\n"
                    "4. dissent_score: Are disagreements and limitations acknowledged?\n"
                    "Provide clear reasoning."
                )

                response = await self.llm_router.generate_structured(
                    LLMRequest(
                        prompt=prompt,
                        system_instruction=(
                            "You are a scientific evaluation judge. "
                            "Evaluate the output objectively based on provided evidence."
                        ),
                        max_output_tokens=1024,
                    ),
                    LLMJudgeStructuredOutput,
                    Workload.REASONING,
                )
                data = response.data
                return LLMJudgeScore(
                    label="LLM_JUDGE",
                    judge_status="COMPLETED",
                    case_id=case.id,
                    mode=mode,
                    relevance_score=float(data.relevance_score),
                    grounding_score=float(data.grounding_score),
                    conservativeness_score=float(data.conservativeness_score),
                    dissent_score=float(data.dissent_score),
                    reasoning=data.reasoning,
                    evaluated_at=now_str,
                )
            except Exception as exc:
                logger.warning(
                    "LLM Judge evaluation failed",
                    extra={"event": "llm_judge_failure", "error": str(exc)},
                )
                if not self.allow_heuristic_fallback:
                    return LLMJudgeScore(
                        label="LLM_JUDGE",
                        judge_status="NOT_RUN",
                        case_id=case.id,
                        mode=mode,
                        reasoning=f"LLM Judge invocation failed: {exc}",
                        evaluated_at=now_str,
                    )

        if not self.allow_heuristic_fallback:
            return LLMJudgeScore(
                label="LLM_JUDGE",
                judge_status="NOT_RUN",
                case_id=case.id,
                mode=mode,
                reasoning="LLM Router not configured; judge not run.",
                evaluated_at=now_str,
            )

        # Deterministic heuristic fallback (strictly for unit tests)
        rel = 8.0 if len(answer) > 80 else 4.0
        grd = 8.5 if len(evidence_sources) >= 2 else (6.0 if evidence_sources else 2.0)
        cons = 9.0 if safe_conclusion else 7.0
        dis = 8.0 if "perbedaan" in answer.lower() or "dissent" in answer.lower() else 6.0
        return LLMJudgeScore(
            label="LLM_JUDGE",
            judge_status="HEURISTIC_FALLBACK_TEST_ONLY",
            case_id=case.id,
            mode=mode,
            relevance_score=rel,
            grounding_score=grd,
            conservativeness_score=cons,
            dissent_score=dis,
            reasoning="Heuristic fallback evaluation (TEST ONLY).",
            evaluated_at=now_str,
        )
