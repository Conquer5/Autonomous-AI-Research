"""Execute the 8-case (16 runs) Live Research Calibration Pilot in full isolation."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

from research_radar.agent.hermes_runtime import HermesHttpRuntime
from research_radar.config import AppSettings
from research_radar.eval.dataset import load_live_pilot_dataset
from research_radar.eval.models import ExecutionType
from research_radar.eval.report import ReportGenerator
from research_radar.eval.runner import EvaluationRunner
from research_radar.eval.storage import EvaluationStorage
from research_radar.evidence.registry import EvidenceRegistry
from research_radar.llm.gemini import GeminiProvider
from research_radar.llm.router import LLMRouter
from research_radar.research.orchestrator import ResearchOrchestrator
from research_radar.research.planner import ResearchPlanner
from research_radar.research.verifier import ClaimVerifier
from research_radar.tools.arxiv import ArxivSearchTool
from research_radar.tools.github import GitHubClient, GitHubRepositoryAnalyzer, GitHubSearchTool
from research_radar.tools.news import RssNewsTool
from research_radar.tools.registry import ToolRegistry
from research_radar.tools.web import BraveWebSearchTool
from research_radar.utils.retry import RetryPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("live_pilot")


def _generate_human_review_markdown(package: dict) -> str:
    """Generate human review Markdown artifact with blank scoring forms."""
    lines: list[str] = []
    lines.append("# Research Radar — Live Pilot Human Review Packet")
    lines.append("")
    lines.append(f"- **Evaluation ID:** `{package.get('evaluation_id')}`")
    lines.append(f"- **Generated At:** `{package.get('generated_at')}`")
    lines.append(f"- **Cases To Review:** `{package.get('total_cases_for_review')}`")
    lines.append("")
    lines.append("## Instructions for Human Evaluator")
    lines.append("For each case below, review the responses from **Mode A** and **Mode B**.")
    lines.append("Score each dimension on a 3-point scale:")
    lines.append("- `0` = Poor / Unacceptable / Hallucinated")
    lines.append("- `1` = Partial / Marginally Acceptable")
    lines.append("- `2` = Good / High Quality / Rigorous")
    lines.append("")
    lines.append("---")
    lines.append("")

    for item in package.get("cases", []):
        case_id = item["case_id"]
        q = item["question"]
        cat = item["category"]
        diff = item["difficulty"]

        lines.append(f"### Case: `{case_id}` — {cat.upper()} ({diff})")
        lines.append(f"**Question:** {q}")
        lines.append("")

        for mode_key in ("mode_A", "mode_B"):
            m = item[mode_key]
            m_name = m["mode_name"]
            lines.append(f"#### {mode_key.upper()} ({m_name})")
            lines.append("**Answer Snippet:**")
            lines.append(f"> {m['answer_snippet']}")
            lines.append("")
            if m.get("safe_conclusion"):
                lines.append(f"**Safe Conclusion:** `{m['safe_conclusion']}`")
                lines.append("")

            lines.append("**Evidence Sources:**")
            for s in m.get("sources", [])[:4]:
                lines.append(f"- [{s.get('title')}]({s.get('url')})")
            lines.append("")

            lines.append("**Scoring Form (Fill 0, 1, or 2):**")
            lines.append("- [ ] **Relevance (0-2):** `___`")
            lines.append("- [ ] **Completeness (0-2):** `___`")
            lines.append("- [ ] **Evidence Quality (0-2):** `___`")
            lines.append("- [ ] **Conservativeness (0-2):** `___`")
            lines.append("- [ ] **Usefulness (0-2):** `___`")
            lines.append("- [ ] **Dissent Handling (0-2):** `___`")
            lines.append("- **Notes:** `___________________________________`")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


async def main() -> int:
    settings = AppSettings()
    logger.info("Initializing isolated live pilot calibration harness...")

    # 1. Isolated Evaluation Database & Storage
    eval_dir = Path("eval")
    storage = EvaluationStorage(eval_dir)
    eval_db_path = eval_dir / "data" / "radar_eval.sqlite3"
    eval_db_path.parent.mkdir(parents=True, exist_ok=True)

    evidence_registry = EvidenceRegistry(eval_db_path)
    claim_verifier = ClaimVerifier()

    # 2. Build live tools and models
    retry = RetryPolicy(attempts=settings.max_retries)
    gemini = (
        GeminiProvider(
            api_key=settings.gemini_api_key.get_secret_value(),
            default_model=settings.gemini_model,
            fallback_models=settings.gemini_fallback_models,
            rpm_limit=settings.gemini_rpm_limit,
            timeout_seconds=settings.request_timeout_seconds,
            retry_policy=retry,
        )
        if settings.gemini_api_key
        else None
    )
    llm_router = (
        LLMRouter(
            gemini,
            default_model=settings.gemini_model,
            fast_model=settings.gemini_fast_model,
            reasoning_model=settings.gemini_reasoning_model,
        )
        if gemini is not None
        else None
    )
    hermes = (
        HermesHttpRuntime(
            base_url=str(settings.hermes_api_url),
            api_key=settings.hermes_api_key.get_secret_value() if settings.hermes_api_key else None,
            model_name=settings.hermes_model_name,
            timeout_seconds=max(settings.request_timeout_seconds, 120),
            retry_policy=retry,
        )
        if settings.hermes_api_key
        else None
    )
    github = GitHubClient(
        token=settings.github_token.get_secret_value() if settings.github_token else None,
        base_url=str(settings.github_api_url),
        api_version=settings.github_api_version,
        timeout_seconds=settings.request_timeout_seconds,
        concurrency=settings.max_concurrency,
        retry_policy=retry,
    )
    arxiv = ArxivSearchTool(
        base_url=str(settings.arxiv_api_url),
        timeout_seconds=settings.request_timeout_seconds,
        min_interval_seconds=settings.arxiv_min_interval_seconds,
        retry_policy=retry,
    )
    news = RssNewsTool(
        settings.news_feed_urls,
        timeout_seconds=settings.request_timeout_seconds,
        concurrency=settings.max_concurrency,
    )
    web = (
        BraveWebSearchTool(
            api_key=settings.brave_search_api_key.get_secret_value(),
            base_url=str(settings.brave_search_api_url),
            timeout_seconds=settings.request_timeout_seconds,
            concurrency=settings.max_concurrency,
            retry_policy=retry,
        )
        if settings.brave_search_api_key
        else None
    )

    tools = ToolRegistry(
        github_search=GitHubSearchTool(github),
        github_analyzer=GitHubRepositoryAnalyzer(github),
        arxiv_search=arxiv,
        news_search=news,
        web_search=web,
    )
    planner = ResearchPlanner(llm_router=llm_router, hermes_runtime=hermes)
    orchestrator = ResearchOrchestrator(
        settings=settings,
        tools=tools,
        evidence_registry=evidence_registry,
        claim_verifier=claim_verifier,
        llm_router=llm_router,
        hermes_runtime=hermes,
        planner=planner,
    )

    try:
        # 3. Load 8-Case Live Pilot Dataset
        dataset = load_live_pilot_dataset()
        logger.info(
            "Loaded Live Pilot dataset: %d cases (%s)",
            len(dataset.cases),
            [c.id for c in dataset.cases],
        )

        # 4. Execute Live Calibration Run (8 cases x 2 modes = 16 runs)
        runner = EvaluationRunner(
            orchestrator=orchestrator,
            execution_type=ExecutionType.LIVE,
        )

        t0 = time.perf_counter()
        logger.info("Starting live evaluation runs across QUICK and DEEP modes...")
        report = await runner.run_evaluation(dataset, modes=["QUICK", "DEEP"])
        elapsed = time.perf_counter() - t0
        logger.info("Completed %d runs in %.2fs", report.total_runs, elapsed)

        # 5. Persist Output Artifacts
        json_path = storage.save_report_json(report, "live_pilot_v1.json")
        md_text = ReportGenerator.generate_markdown(report)
        md_path = storage.save_report_markdown(md_text, "live_pilot_v1.md")

        # 6. Generate and Save Human Review Package
        review_pkg = ReportGenerator.generate_human_review_package(report, dataset)
        review_json_path = storage.save_human_review_package(
            review_pkg, "live_pilot_human_review.json"
        )
        review_md_text = _generate_human_review_markdown(review_pkg)
        review_md_path = eval_dir / "results" / "live_pilot_human_review.md"
        review_md_path.write_text(review_md_text.strip() + "\n")

        logger.info("Saved JSON report -> %s", json_path)
        logger.info("Saved Markdown report -> %s", md_path)
        logger.info("Saved Human Review JSON -> %s", review_json_path)
        logger.info("Saved Human Review MD -> %s", review_md_path)

        # Print Executive Summary to stdout
        print("\n" + "=" * 60)
        print("P1C LIVE RESEARCH CALIBRATION PILOT COMPLETED")
        print("=" * 60)
        print(f"Total Cases: {report.total_cases}")
        print(f"Total Runs: {report.total_runs}")
        print(f"Execution Type: {report.execution_type.value}")
        print(f"Elapsed Time: {elapsed:.2f}s")
        print("-" * 60)
        print(md_text)
        print("=" * 60)

        return 0
    finally:
        if gemini is not None:
            await gemini.aclose()
        if hermes is not None:
            await hermes.aclose()
        await github.aclose()
        await arxiv.aclose()
        await news.aclose()
        if web is not None:
            await web.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
