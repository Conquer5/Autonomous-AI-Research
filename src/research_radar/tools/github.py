"""Normalized GitHub repository search and progressive repository analysis."""

from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import urlparse

import httpx

from research_radar.evidence.canonicalizer import canonicalize_url
from research_radar.exceptions import InvalidResponseError
from research_radar.retrieval import RetrievalFailure, observed_search, request_with_retry
from research_radar.schemas import (
    CommitInfo,
    ReleaseInfo,
    RepositoryAnalysis,
    RepositoryResult,
    RepositorySignals,
    SearchBatch,
    SourceType,
)
from research_radar.utils.retry import RetryPolicy

logger = logging.getLogger(__name__)
_WORD_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.+-]*", re.IGNORECASE)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def calculate_repository_signals(
    repository: RepositoryResult,
    query: str,
    *,
    now: datetime | None = None,
) -> RepositorySignals:
    """Calculate transparent heuristic signals; these are not scientific scores."""

    current = now or datetime.now(UTC)
    created = repository.created_at.astimezone(UTC)
    pushed = (repository.pushed_at or repository.updated_at).astimezone(UTC)
    age_days = max(1.0, (current - created).total_seconds() / 86400)
    inactive_days = max(0.0, (current - pushed).total_seconds() / 86400)

    query_tokens = set(_WORD_PATTERN.findall(query.lower()))
    haystack = " ".join(
        [
            repository.full_name,
            repository.description or "",
            repository.language or "",
            " ".join(repository.topics),
        ]
    ).lower()
    result_tokens = set(_WORD_PATTERN.findall(haystack))
    relevance = len(query_tokens & result_tokens) / len(query_tokens) if query_tokens else 0.5

    popularity = math.log10(repository.stars + repository.forks * 2 + 1) / 5
    star_velocity = repository.stars / age_days
    growth = math.log10(star_velocity + 1) / 3
    size_depth = math.log10(repository.size_kb + 1) / 6
    topic_depth = min(len(repository.topics), 8) / 8
    technical_depth = (
        0.25 * size_depth
        + 0.20 * topic_depth
        + 0.20 * float(repository.language is not None)
        + 0.20 * float(repository.license_name is not None)
        + 0.15 * float(not repository.archived)
    )
    return RepositorySignals(
        recency_score=_clamp(1 - age_days / 365),
        activity_score=_clamp(1 - inactive_days / 90),
        popularity_score=_clamp(popularity),
        growth_score=_clamp(growth),
        relevance_score=_clamp(relevance),
        technical_depth_score=_clamp(technical_depth),
    )


def normalize_repository_reference(reference: str) -> tuple[str, str]:
    """Accept `owner/repo` or a normal github.com repository URL."""

    cleaned = reference.strip().removesuffix(".git").rstrip("/")
    if "://" in cleaned:
        parsed = urlparse(cleaned)
        if parsed.hostname not in {"github.com", "www.github.com"}:
            raise ValueError("repository URL must use github.com")
        parts = [part for part in parsed.path.split("/") if part]
    else:
        parts = [part for part in cleaned.split("/") if part]
    if len(parts) != 2 or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts):
        raise ValueError("repository must be in owner/repo form")
    return parts[0], parts[1]


class GitHubClient:
    """Shared authenticated GitHub REST transport."""

    def __init__(
        self,
        *,
        token: str | None = None,
        base_url: str = "https://api.github.com",
        api_version: str = "2026-03-10",
        timeout_seconds: float = 30,
        concurrency: int = 5,
        retry_policy: RetryPolicy | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.retry_policy = retry_policy or RetryPolicy()
        self._semaphore = asyncio.Semaphore(concurrency)
        self.timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": api_version,
            "User-Agent": "autonomous-ai-research-radar/0.2",
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    async def request(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        accept: str | None = None,
    ) -> httpx.Response:
        headers = dict(self._headers)
        if accept:
            headers["Accept"] = accept

        async def operation() -> httpx.Response:
            async with self._semaphore:
                response = await self._client.get(
                    f"{self.base_url}{path}", params=params, headers=headers
                )
            response.raise_for_status()
            return response

        return await request_with_retry(
            "github", operation, policy=self.retry_policy, timeout_seconds=self.timeout_seconds
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _repository_from_json(data: dict[str, Any]) -> RepositoryResult:
    license_data = data.get("license") or {}
    pushed_at = _parse_datetime(data.get("pushed_at"))
    created_at = _parse_datetime(data.get("created_at"))
    updated_at = _parse_datetime(data.get("updated_at"))
    if created_at is None or updated_at is None:
        raise ValueError("repository timestamps are missing")
    return RepositoryResult(
        full_name=data["full_name"],
        description=data.get("description"),
        url=data["html_url"],
        stars=data.get("stargazers_count", 0),
        forks=data.get("forks_count", 0),
        open_issues=data.get("open_issues_count", 0),
        language=data.get("language"),
        topics=data.get("topics") or [],
        created_at=created_at,
        updated_at=updated_at,
        pushed_at=pushed_at,
        default_branch=data.get("default_branch") or "main",
        archived=data.get("archived", False),
        license_name=license_data.get("spdx_id") or license_data.get("name"),
        size_kb=data.get("size", 0),
    )


class GitHubSearchTool:
    def __init__(self, github: GitHubClient) -> None:
        self.github = github

    @observed_search("github")
    async def search(
        self,
        query: str,
        *,
        created_after: date | None = None,
        updated_after: date | None = None,
        language: str | None = None,
        limit: int = 10,
        sort: str = "stars",
    ) -> SearchBatch[RepositoryResult]:
        if not query.strip():
            raise ValueError("GitHub query must not be empty")
        if not 1 <= limit <= 100:
            raise ValueError("GitHub result limit must be between 1 and 100")
        if sort not in {"stars", "forks", "help-wanted-issues", "updated"}:
            raise ValueError("unsupported GitHub sort")
        qualifiers = [query.strip()]
        if created_after:
            qualifiers.append(f"created:>={created_after.isoformat()}")
        if updated_after:
            qualifiers.append(f"pushed:>={updated_after.isoformat()}")
        if language:
            qualifiers.append(f"language:{language}")
        normalized_query = " ".join(qualifiers)
        response = await self.github.request(
            "/search/repositories",
            params={
                "q": normalized_query,
                "sort": sort,
                "order": "desc",
                "per_page": limit,
            },
        )
        if response.status_code == 204:
            return SearchBatch[RepositoryResult](query=query, source=SourceType.GITHUB, items=[])
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            raise TypeError("repository search envelope changed")
        repositories = []
        failures = []
        for item in data["items"][:limit]:
            try:
                repository = _repository_from_json(item)
                if not repository.full_name.strip():
                    raise ValueError("missing repository name")
                repositories.append(repository)
            except (ValueError, KeyError, TypeError, AttributeError):
                failures.append(
                    RetrievalFailure(
                        provider="github",
                        failure_category="schema_drift",
                        error_code="INVALID_ENTRY",
                    )
                )
        scored = [
            repository.model_copy(
                update={"signals": calculate_repository_signals(repository, query)}
            )
            for repository in repositories
        ]
        return SearchBatch[RepositoryResult](
            query=query,
            source=SourceType.GITHUB,
            items=list({canonicalize_url(str(item.url)): item for item in scored}.values()),
            failures=failures,
            warnings=["Invalid repository entry skipped"] if failures else [],
            total_available=data.get("total_count"),
            partial=bool(data.get("incomplete_results", False) or failures),
        )


class GitHubRepositoryAnalyzer:
    def __init__(self, github: GitHubClient, *, readme_character_budget: int = 12_000) -> None:
        self.github = github
        self.readme_character_budget = readme_character_budget

    async def get_repository(self, reference: str) -> RepositoryResult:
        owner, repo = normalize_repository_reference(reference)
        response = await self.github.request(f"/repos/{owner}/{repo}")
        try:
            repository = _repository_from_json(response.json())
        except (ValueError, KeyError, TypeError) as exc:
            raise InvalidResponseError(
                "github",
                "Metadata repository GitHub tidak valid.",
                detail=type(exc).__name__,
            ) from exc
        return repository.model_copy(
            update={"signals": calculate_repository_signals(repository, repository.full_name)}
        )

    async def _readme(self, owner: str, repo: str) -> str:
        response = await self.github.request(
            f"/repos/{owner}/{repo}/readme", accept="application/vnd.github.raw+json"
        )
        return response.text

    async def readme(self, reference: str) -> str:
        """Fetch bounded README evidence without the heavier full repository analysis."""

        owner, repo = normalize_repository_reference(reference)
        readme = await self._readme(owner, repo)
        if len(readme) > self.readme_character_budget:
            return readme[: self.readme_character_budget].rstrip() + "\n\n[README truncated]"
        return readme

    async def _latest_release(self, owner: str, repo: str) -> ReleaseInfo:
        response = await self.github.request(f"/repos/{owner}/{repo}/releases/latest")
        data = response.json()
        return ReleaseInfo(
            tag_name=data["tag_name"],
            name=data.get("name"),
            url=data["html_url"],
            published_at=_parse_datetime(data.get("published_at")),
            prerelease=data.get("prerelease", False),
        )

    async def _recent_commits(self, owner: str, repo: str, *, limit: int) -> list[CommitInfo]:
        response = await self.github.request(
            f"/repos/{owner}/{repo}/commits", params={"per_page": limit}
        )
        commits: list[CommitInfo] = []
        for item in response.json():
            commit = item.get("commit") or {}
            author = commit.get("author") or {}
            verification = commit.get("verification") or {}
            commits.append(
                CommitInfo(
                    sha=item["sha"],
                    message=(commit.get("message") or "").splitlines()[0],
                    url=item["html_url"],
                    author_name=author.get("name"),
                    committed_at=_parse_datetime(author.get("date")),
                    verified=verification.get("verified", False),
                )
            )
        return commits

    async def analyze(self, reference: str, *, recent_commit_limit: int = 5) -> RepositoryAnalysis:
        owner, repo = normalize_repository_reference(reference)
        repository_task = self.get_repository(f"{owner}/{repo}")
        readme_task = self._readme(owner, repo)
        release_task = self._latest_release(owner, repo)
        commits_task = self._recent_commits(owner, repo, limit=recent_commit_limit)
        results = await asyncio.gather(
            repository_task,
            readme_task,
            release_task,
            commits_task,
            return_exceptions=True,
        )
        repository_result = results[0]
        if isinstance(repository_result, BaseException):
            raise repository_result
        if not isinstance(repository_result, RepositoryResult):
            raise InvalidResponseError("github", "Metadata repository GitHub tidak valid.")

        warnings: list[str] = []
        readme: str | None = None
        release: ReleaseInfo | None = None
        commits: list[CommitInfo] = []

        if isinstance(results[1], str):
            readme = results[1]
        else:
            warnings.append("README tidak tersedia.")
        if isinstance(results[2], ReleaseInfo):
            release = results[2]
        else:
            warnings.append("Release terbaru tidak tersedia.")
        if isinstance(results[3], list):
            commits = results[3]
        else:
            warnings.append("Commit terbaru tidak tersedia.")

        truncated = bool(readme and len(readme) > self.readme_character_budget)
        if truncated and readme is not None:
            readme = readme[: self.readme_character_budget].rstrip() + "\n\n[README truncated]"
        return RepositoryAnalysis(
            repository=repository_result,
            readme=readme,
            readme_truncated=truncated,
            latest_release=release,
            recent_commits=commits,
            warnings=warnings,
        )
