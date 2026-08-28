"""Evaluation and Calibration framework for Autonomous AI Research Radar."""

from __future__ import annotations

from research_radar.eval.adversarial import AdversarialEvaluator
from research_radar.eval.dataset import (
    DEFAULT_EVALUATION_CASES,
    LIVE_PILOT_CASE_IDS,
    load_dataset_from_file,
    load_default_dataset,
    load_live_pilot_dataset,
    save_dataset_to_file,
    validate_dataset,
)
from research_radar.eval.llm_judge import LLMJudge, LLMJudgeStructuredOutput
from research_radar.eval.metrics import MetricCalculator
from research_radar.eval.models import (
    AdversarialFixtureResult,
    AggregateComparison,
    CaseExecutionResult,
    DifficultyLevel,
    EvaluationCase,
    EvaluationCategory,
    EvaluationDataset,
    EvaluationMetrics,
    EvaluationRunReport,
    ExecutionType,
    ExpectedDisagreement,
    FailureCategory,
    HumanRubricScore,
    LLMJudgeScore,
    ResearchCharacteristics,
)
from research_radar.eval.report import ReportGenerator
from research_radar.eval.runner import EvaluationRunner
from research_radar.eval.storage import EvaluationStorage

__all__ = [
    "DEFAULT_EVALUATION_CASES",
    "LIVE_PILOT_CASE_IDS",
    "AdversarialEvaluator",
    "AdversarialFixtureResult",
    "AggregateComparison",
    "CaseExecutionResult",
    "DifficultyLevel",
    "EvaluationCase",
    "EvaluationCategory",
    "EvaluationDataset",
    "EvaluationMetrics",
    "EvaluationRunReport",
    "EvaluationRunner",
    "EvaluationStorage",
    "ExecutionType",
    "ExpectedDisagreement",
    "FailureCategory",
    "HumanRubricScore",
    "LLMJudge",
    "LLMJudgeScore",
    "LLMJudgeStructuredOutput",
    "MetricCalculator",
    "ReportGenerator",
    "ResearchCharacteristics",
    "load_dataset_from_file",
    "load_default_dataset",
    "load_live_pilot_dataset",
    "save_dataset_to_file",
    "validate_dataset",
]
