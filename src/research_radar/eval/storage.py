"""Isolated persistence for research quality evaluation outputs and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_radar.eval.models import (
    CaseExecutionResult,
    EvaluationDataset,
    EvaluationRunReport,
    HumanRubricScore,
    LLMJudgeScore,
)


class EvaluationStorage:
    """Manages file storage for evaluation datasets, execution outputs, rubrics, and reports."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path("eval")
        self.cases_dir = self.base_dir / "cases"
        self.results_dir = self.base_dir / "results"
        self.reports_dir = self.base_dir / "reports"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def save_dataset(self, dataset: EvaluationDataset, filename: str = "dataset_v1.json") -> Path:
        target = self.cases_dir / filename
        target.write_text(json.dumps(dataset.model_dump(), indent=2, ensure_ascii=False) + "\n")
        return target

    def load_dataset(self, filename: str = "dataset_v1.json") -> EvaluationDataset:
        source = self.cases_dir / filename
        data = json.loads(source.read_text())
        return EvaluationDataset.model_validate(data)

    def save_results(self, results: list[CaseExecutionResult], filename: str) -> Path:
        target = self.results_dir / filename
        payload = [r.model_dump() for r in results]
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return target

    def load_results(self, filename: str) -> list[CaseExecutionResult]:
        source = self.results_dir / filename
        raw = json.loads(source.read_text())
        return [CaseExecutionResult.model_validate(item) for item in raw]

    def save_human_rubric(self, scores: list[HumanRubricScore], filename: str) -> Path:
        target = self.results_dir / filename
        payload = [s.model_dump() for s in scores]
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return target

    def load_human_rubric(self, filename: str) -> list[HumanRubricScore]:
        source = self.results_dir / filename
        raw = json.loads(source.read_text())
        return [HumanRubricScore.model_validate(item) for item in raw]

    def save_llm_judge(self, scores: list[LLMJudgeScore], filename: str) -> Path:
        target = self.results_dir / filename
        payload = [s.model_dump() for s in scores]
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return target

    def save_report_json(self, report: EvaluationRunReport, filename: str = "report.json") -> Path:
        target = self.reports_dir / filename
        target.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False) + "\n")
        return target

    def save_report_markdown(self, markdown: str, filename: str = "report.md") -> Path:
        target = self.reports_dir / filename
        target.write_text(markdown.strip() + "\n")
        return target

    def save_human_review_package(
        self, package_data: dict[str, Any], filename: str = "live_pilot_human_review.json"
    ) -> Path:
        target = self.results_dir / filename
        target.write_text(json.dumps(package_data, indent=2, ensure_ascii=False) + "\n")
        return target
