"""Explicit tool registry; tools are selected rather than called indiscriminately."""

from __future__ import annotations

from dataclasses import dataclass

from research_radar.schemas import ToolHealth
from research_radar.tools.arxiv import ArxivSearchTool
from research_radar.tools.github import GitHubRepositoryAnalyzer, GitHubSearchTool
from research_radar.tools.news import RssNewsTool
from research_radar.tools.web import BraveWebSearchTool


@dataclass(slots=True)
class ToolRegistry:
    github_search: GitHubSearchTool
    github_analyzer: GitHubRepositoryAnalyzer
    arxiv_search: ArxivSearchTool
    news_search: RssNewsTool | None = None
    web_search: BraveWebSearchTool | None = None

    def available_sources(self) -> frozenset[str]:
        return frozenset(
            name
            for name, tool in (
                ("github", self.github_search),
                ("arxiv", self.arxiv_search),
                ("news", self.news_search),
                ("web", self.web_search),
            )
            if tool is not None
        )

    def status(self) -> list[ToolHealth]:
        return [
            ToolHealth(name="github_search", configured=True),
            ToolHealth(name="github_analyzer", configured=True),
            ToolHealth(name="arxiv_search", configured=True),
            ToolHealth(name="news_search", configured=self.news_search is not None),
            ToolHealth(name="web_search", configured=self.web_search is not None),
        ]
