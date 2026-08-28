from __future__ import annotations

from datetime import UTC, datetime

import httpx

from research_radar.tools.github import (
    GitHubClient,
    GitHubRepositoryAnalyzer,
    GitHubSearchTool,
    calculate_repository_signals,
    normalize_repository_reference,
)
from research_radar.utils.retry import RetryPolicy

REPOSITORY = {
    "full_name": "example/agent",
    "description": "AI agent harness with evaluation tools",
    "html_url": "https://github.com/example/agent",
    "stargazers_count": 1200,
    "forks_count": 120,
    "open_issues_count": 12,
    "language": "Python",
    "topics": ["ai-agent", "evaluation"],
    "created_at": "2026-07-01T00:00:00Z",
    "updated_at": "2026-08-25T00:00:00Z",
    "pushed_at": "2026-08-26T00:00:00Z",
    "default_branch": "main",
    "archived": False,
    "license": {"spdx_id": "MIT"},
    "size": 2048,
}


async def test_github_search_normalizes_and_scores_repositories() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search/repositories"
        assert request.headers["x-github-api-version"] == "2026-03-10"
        return httpx.Response(
            200,
            json={"total_count": 1, "incomplete_results": False, "items": [REPOSITORY]},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    github = GitHubClient(client=http, retry_policy=RetryPolicy(attempts=1))
    result = await GitHubSearchTool(github).search("AI agent", limit=5)

    assert result.total_available == 1
    assert result.items[0].full_name == "example/agent"
    assert result.items[0].signals is not None
    assert result.items[0].signals.activity_score > 0
    await http.aclose()


async def test_repository_analysis_handles_optional_partial_failures() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/example/agent":
            return httpx.Response(200, json=REPOSITORY)
        if request.url.path.endswith("/readme"):
            assert request.headers["accept"] == "application/vnd.github.raw+json"
            return httpx.Response(200, text="A" * 50)
        if request.url.path.endswith("/releases/latest"):
            return httpx.Response(404, json={"message": "Not Found"})
        if request.url.path.endswith("/commits"):
            return httpx.Response(
                200,
                json=[
                    {
                        "sha": "abcdef123456",
                        "html_url": "https://github.com/example/agent/commit/abcdef",
                        "commit": {
                            "message": "Add evaluation\n\nDetails",
                            "author": {"name": "Alice", "date": "2026-08-26T00:00:00Z"},
                            "verification": {"verified": True},
                        },
                    }
                ],
            )
        raise AssertionError(f"unexpected path: {request.url.path}")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    github = GitHubClient(client=http, retry_policy=RetryPolicy(attempts=1))
    analyzer = GitHubRepositoryAnalyzer(github, readme_character_budget=20)

    result = await analyzer.analyze("https://github.com/example/agent")

    assert result.repository.full_name == "example/agent"
    assert result.readme_truncated is True
    assert result.latest_release is None
    assert result.recent_commits[0].message == "Add evaluation"
    assert "Release terbaru tidak tersedia." in result.warnings
    await http.aclose()


def test_repository_reference_and_signal_bounds() -> None:
    assert normalize_repository_reference("https://github.com/example/agent.git") == (
        "example",
        "agent",
    )
    from research_radar.schemas import RepositoryResult

    repo = RepositoryResult(
        full_name="example/agent",
        url="https://github.com/example/agent",
        stars=1_000_000,
        forks=50_000,
        language="Python",
        topics=["agent"],
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 8, 1, tzinfo=UTC),
        pushed_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    scores = calculate_repository_signals(repo, "agent", now=datetime(2026, 8, 27, tzinfo=UTC))
    assert all(0 <= value <= 1 for value in scores.model_dump().values())
