from __future__ import annotations

from unittest.mock import MagicMock

from research_radar.agent.hermes_runtime import FakeHermesRuntime
from research_radar.service import RadarService
from research_radar.tools.registry import ToolRegistry


async def test_service_proves_mocked_telegram_to_hermes_boundary() -> None:
    runtime = FakeHermesRuntime("evidence-based answer")
    registry = ToolRegistry(
        github_search=MagicMock(),
        github_analyzer=MagicMock(),
        arxiv_search=MagicMock(),
    )
    service = RadarService(runtime=runtime, tools=registry)

    result = await service.ask_agent("What changed?", user_id=42)
    status = await service.status()

    assert result.text == "evidence-based answer"
    assert runtime.calls == [("What changed?", "telegram:42")]
    assert status.metrics.successful_tasks == 1
    assert status.metrics.runtime_calls == 1


async def test_zero_token_waste_invariant_on_direct_tools() -> None:
    from unittest.mock import AsyncMock

    from research_radar.schemas import (
        SearchBatch,
        SourceType,
    )

    runtime = FakeHermesRuntime("should not be called")
    github_search = MagicMock()
    github_search.search = AsyncMock(
        return_value=SearchBatch(query="agent", source=SourceType.GITHUB, items=[])
    )
    arxiv_search = MagicMock()
    arxiv_search.search = AsyncMock(
        return_value=SearchBatch(query="agent", source=SourceType.ARXIV, items=[])
    )
    web_search = MagicMock()
    web_search.search = AsyncMock(
        return_value=SearchBatch(query="agent", source=SourceType.WEB, items=[])
    )

    registry = ToolRegistry(
        github_search=github_search,
        github_analyzer=MagicMock(),
        arxiv_search=arxiv_search,
        web_search=web_search,
    )
    service = RadarService(runtime=runtime, tools=registry)

    # 1. Search repos
    await service.search_repositories("agent", user_id=1)
    # 2. Search papers
    await service.search_papers("agent", user_id=1)
    # 3. Search web
    await service.search_web("agent", user_id=1)

    status = await service.status()

    # Zero LLM / Hermes calls made
    assert len(runtime.calls) == 0
    assert status.metrics.runtime_calls == 0
    assert status.metrics.llm_calls == 0
    assert status.metrics.tool_calls == 3
