"""Compare saved pilots without treating missing historical telemetry as zero."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def summarize(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    runs = report["quick_results"] + report["deep_results"]
    metrics = [run["metrics"] for run in runs]
    count = len(metrics)
    if count == 0:
        raise ValueError("report contains no runs")
    observed = all("retrieval_attempts" in metric for metric in metrics)
    attempts = sum(metric.get("retrieval_attempts", 0) for metric in metrics)
    retries = sum(metric.get("retrieval_retries", 0) for metric in metrics)
    calls = sum(metric["tool_calls"] for metric in metrics)
    failures = sum(metric["collector_failures"] for metric in metrics)
    return {
        "report": str(path),
        "generated_at": report["generated_at"],
        "runs": count,
        "runs_with_evidence": sum(metric["evidence_count"] > 0 for metric in metrics),
        "retrieval_success_percent": 100 * mean(metric["evidence_count"] > 0 for metric in metrics),
        "mean_evidence_per_run": mean(metric["evidence_count"] for metric in metrics),
        "mean_latency_seconds": mean(metric["wall_time"] for metric in metrics),
        "partial_run_percent": 100
        * mean(metric["research_status"] == "partial" for metric in metrics),
        "collector_failures_reported": failures,
        "collector_failure_percent": 100 * failures / calls if calls else None,
        "tool_calls": calls,
        "http_attempts": attempts if observed else None,
        "retries": retries if observed else None,
        "retry_percent_of_http_attempts": 100 * retries / attempts
        if observed and attempts
        else None,
        "empty_queries": sum(metric.get("empty_queries", 0) for metric in metrics)
        if observed
        else None,
        "nonempty_queries": sum(metric.get("nonempty_queries", 0) for metric in metrics)
        if observed
        else None,
        "mean_retrieval_latency_ms": mean(metric["retrieval_latency_ms"] for metric in metrics)
        if observed
        else None,
        "adversarial_passed": sum(fixture["passed"] for fixture in report["adversarial_results"]),
        "budget_violations": [
            {"case": run["case_id"], "mode": run["mode"]}
            for run in runs
            if any(
                run["metrics"][key] > limit
                for key, limit in zip(
                    [
                        "iterations",
                        "tool_calls",
                        "queries",
                        "evidence_count",
                        "wall_time",
                        "llm_calls",
                    ],
                    [1, 5, 5, 15, 60, 3] if run["mode"] == "QUICK" else [3, 12, 10, 30, 180, 8],
                    strict=True,
                )
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    print(json.dumps({"before": summarize(args.before), "after": summarize(args.after)}, indent=2))


if __name__ == "__main__":
    main()
