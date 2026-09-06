from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from research_radar.config import AppSettings
from research_radar.digest import DigestEngine
from research_radar.schemas import (
    DigestItemSynthesis,
    DigestSynthesis,
    NewsResult,
    PaperResult,
    RepositoryResult,
    ResearchDigest,
    SearchBatch,
    SourceType,
)
from research_radar.state import DigestStore
from research_radar.telegram.formatter import format_digest
from research_radar.tools.registry import ToolRegistry


async def test_digest_collects_ranks_and_suppresses_already_sent_evidence(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        digest_topics="efficient local AI",
        digest_search_queries="efficient local AI",
        digest_items_per_source=2,
        digest_state_path=str(tmp_path) + "/radar.sqlite3",
        news_feed_urls="https://example.com/feed",
    )
    repo = RepositoryResult(
        full_name="example/tiny-ai",
        description="Efficient local AI for CPU laptops",
        url="https://github.com/example/tiny-ai",
        stars=100,
        forks=10,
        language="Python",
        topics=["local-ai"],
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        updated_at=datetime(2026, 8, 27, tzinfo=UTC),
        pushed_at=datetime(2026, 8, 27, tzinfo=UTC),
    )
    paper = PaperResult(
        arxiv_id="2608.12345",
        title="Efficient Local AI",
        authors=["A. Researcher"],
        abstract="CPU-only inference for old laptops.",
        categories=["cs.AI"],
        primary_category="cs.AI",
        published_at=datetime(2026, 8, 26, tzinfo=UTC),
        updated_at=datetime(2026, 8, 26, tzinfo=UTC),
        url="https://arxiv.org/abs/2608.12345",
    )
    news = NewsResult(
        title="Free local AI release",
        url="https://example.com/free-local-ai",
        description="A lightweight open-source model.",
        source_name="Example",
        published_at=datetime(2026, 8, 27, tzinfo=UTC),
        relevance_score=0.9,
    )

    github_search = MagicMock()
    github_search.search = AsyncMock(
        return_value=SearchBatch(query="efficient local AI", source=SourceType.GITHUB, items=[repo])
    )
    arxiv_search = MagicMock()
    arxiv_search.search = AsyncMock(
        return_value=SearchBatch(query="efficient local AI", source=SourceType.ARXIV, items=[paper])
    )
    news_search = MagicMock()
    news_search.search = AsyncMock(
        return_value=SearchBatch(query="efficient local AI", source=SourceType.NEWS, items=[news])
    )
    registry = ToolRegistry(
        github_search=github_search,
        github_analyzer=MagicMock(),
        arxiv_search=arxiv_search,
        news_search=news_search,
    )
    store = DigestStore(settings.digest_state_path)
    engine = DigestEngine(settings=settings, tools=registry, store=store)

    first = await engine.generate()
    assert first.news == [news]
    assert first.repositories[0].full_name == "example/tiny-ai"
    assert first.papers == [paper]

    await engine.record_sent(first)
    second = await engine.generate()
    assert second.evidence_urls() == []
    assert await engine.is_due(now=datetime(2026, 8, 28, 1, tzinfo=UTC)) is False


async def test_digest_ai_explains_each_item_using_bounded_source_evidence(
    tmp_path: Path,
) -> None:
    settings = AppSettings(
        _env_file=None,
        digest_state_path=str(tmp_path / "radar.sqlite3"),
    )
    repo = RepositoryResult(
        full_name="example/agent-api",
        description="API for an AI agent",
        url="https://github.com/example/agent-api",
        stars=42,
        forks=4,
        language="Python",
        topics=["agents"],
        license_name="MIT",
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        updated_at=datetime(2026, 8, 27, tzinfo=UTC),
    )
    paper = PaperResult(
        arxiv_id="2608.00001",
        title="Small Agent Models",
        authors=["A. Author"],
        abstract="A routing method reduces unnecessary model calls in agent workflows.",
        categories=["cs.AI"],
        primary_category="cs.AI",
        published_at=datetime(2026, 8, 26, tzinfo=UTC),
        updated_at=datetime(2026, 8, 26, tzinfo=UTC),
        url="https://arxiv.org/abs/2608.00001",
    )
    news = NewsResult(
        title="Agent API released",
        url="https://example.com/agent-api",
        description="The project released an API for local agent workflows.",
        source_name="Example News",
    )

    def synthesized_item(evidence_id: str, stack: list[str]) -> DigestItemSynthesis:
        return DigestItemSynthesis(
            evidence_id=evidence_id,
            what_it_is="Sebuah komponen AI yang dijelaskan oleh sumber.",
            purpose="Membantu pengguna menjalankan alur kerja agent.",
            how_it_works="Permintaan diarahkan ke komponen yang sesuai berdasarkan tugas.",
            tech_stack=stack,
            key_takeaway="Kontribusi utamanya adalah routing yang lebih selektif.",
            why_it_matters="Inferensi: pemanggilan model dapat menjadi lebih hemat.",
            caveat="Bukti ringkas belum menyediakan benchmark independen.",
            evidence_quality="medium",
        )

    synthesis = DigestSynthesis(
        overview="Ketiga sumber menunjukkan fokus pada alur agent yang lebih efisien.",
        items=[
            synthesized_item("news-1", []),
            synthesized_item("github-1", ["Python", "FastAPI"]),
            synthesized_item("arxiv-1", []),
        ],
    )
    router = MagicMock()
    router.generate_structured = AsyncMock(
        return_value=SimpleNamespace(data=synthesis, model="gemini-test")
    )
    analyzer = MagicMock()
    analyzer.readme = AsyncMock(
        return_value="# Agent API\nBuilt with Python and FastAPI for local workflows."
    )
    registry = ToolRegistry(
        github_search=MagicMock(),
        github_analyzer=analyzer,
        arxiv_search=MagicMock(),
    )
    engine = DigestEngine(
        settings=settings,
        tools=registry,
        store=DigestStore(settings.digest_state_path),
        llm_router=router,
    )

    result = await engine._synthesize(
        ResearchDigest(
            topics=["AI agents"],
            news=[news],
            repositories=[repo],
            papers=[paper],
        )
    )

    assert result.news[0].insight is not None
    assert result.repositories[0].insight is not None
    assert result.repositories[0].insight.tech_stack == ["Python", "FastAPI"]
    assert result.papers[0].insight is not None
    assert result.synthesis_model == "gemini-test"
    analyzer.readme.assert_awaited_once_with("example/agent-api")
    request = router.generate_structured.await_args.args[0]
    assert "Built with Python and FastAPI" in request.prompt
    assert "jangan menebak detail" in request.prompt

    rendered = format_digest(result)
    assert "Fungsi & Cara Kerja:" in rendered
    assert "Stack:" in rendered
    assert "Python, FastAPI" in rendered
    assert "Dampak & Catatan:" in rendered
    assert "Bukti: Sedang" in rendered


async def test_digest_content_aware_deduplication_allows_updated_content(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        digest_state_path=str(tmp_path / "radar_dedup.sqlite3"),
        digest_topics="agents",
        digest_search_queries="agents",
        digest_items_per_source=2,
    )
    repo_v1 = RepositoryResult(
        full_name="example/agent-v1",
        description="Version 1 initial description",
        url="https://github.com/example/agent-v1",
        stars=10,
        forks=0,
        topics=["agents"],
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        updated_at=datetime(2026, 8, 1, tzinfo=UTC),
    )

    github_search = MagicMock()
    github_search.search = AsyncMock(
        return_value=SearchBatch(query="agents", source=SourceType.GITHUB, items=[repo_v1])
    )
    arxiv_search = MagicMock()
    arxiv_search.search = AsyncMock(
        return_value=SearchBatch(query="agents", source=SourceType.ARXIV, items=[])
    )

    registry = ToolRegistry(
        github_search=github_search,
        github_analyzer=MagicMock(),
        arxiv_search=arxiv_search,
    )
    store = DigestStore(settings.digest_state_path)
    engine = DigestEngine(settings=settings, tools=registry, store=store)

    # 1. First run collects and sends repo_v1
    first = await engine.generate()
    assert len(first.repositories) == 1
    assert first.repositories[0].full_name == "example/agent-v1"
    assert first.run_id is not None
    await engine.record_sent(first)

    # 2. Second run with SAME content -> should be suppressed
    second = await engine.generate()
    assert len(second.repositories) == 0

    # 3. Third run with UPDATED content on same URL -> should be collected!
    repo_v2 = RepositoryResult(
        full_name="example/agent-v1",
        description="Version 2 major update with new features",
        url="https://github.com/example/agent-v1",
        stars=500,
        forks=50,
        topics=["agents", "v2"],
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        updated_at=datetime(2026, 8, 28, tzinfo=UTC),
    )
    github_search.search = AsyncMock(
        return_value=SearchBatch(query="agents", source=SourceType.GITHUB, items=[repo_v2])
    )

    third = await engine.generate()
    assert len(third.repositories) == 1
    assert third.repositories[0].description == "Version 2 major update with new features"
    # An unsent revision remains eligible on subsequent runs.
    retry = await engine.generate()
    assert retry.evidence_urls() == third.evidence_urls()
    await engine.record_sent(third)
    assert not (await engine.generate()).repositories


async def test_digest_resilience_on_partial_collector_failure(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        digest_state_path=str(tmp_path / "radar_partial.sqlite3"),
        digest_topics="ai",
        digest_search_queries="ai",
    )
    repo = RepositoryResult(
        full_name="example/working-repo",
        description="Working repo",
        url="https://github.com/example/working-repo",
        stars=10,
        forks=0,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        updated_at=datetime(2026, 8, 1, tzinfo=UTC),
    )

    github_search = MagicMock()
    github_search.search = AsyncMock(
        return_value=SearchBatch(query="ai", source=SourceType.GITHUB, items=[repo])
    )
    # ArXiv fails with an exception (e.g. 503 or network error)
    arxiv_search = MagicMock()
    arxiv_search.search = AsyncMock(side_effect=RuntimeError("ArXiv API 503 Service Unavailable"))

    registry = ToolRegistry(
        github_search=github_search,
        github_analyzer=MagicMock(),
        arxiv_search=arxiv_search,
    )
    store = DigestStore(settings.digest_state_path)
    engine = DigestEngine(settings=settings, tools=registry, store=store)

    digest = await engine.generate()
    # Repository was successfully collected despite arXiv failure
    assert len(digest.repositories) == 1
    # Warning recorded
    assert any("arXiv gagal" in w for w in digest.warnings)


async def test_digest_all_collectors_fail_graceful_handling(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        digest_state_path=str(tmp_path / "radar_all_fail.sqlite3"),
        digest_topics="ai",
        digest_search_queries="ai",
    )
    github_search = MagicMock()
    github_search.search = AsyncMock(side_effect=ConnectionError("GitHub down"))
    arxiv_search = MagicMock()
    arxiv_search.search = AsyncMock(side_effect=ConnectionError("ArXiv down"))

    registry = ToolRegistry(
        github_search=github_search,
        github_analyzer=MagicMock(),
        arxiv_search=arxiv_search,
    )
    store = DigestStore(settings.digest_state_path)
    engine = DigestEngine(settings=settings, tools=registry, store=store)

    digest = await engine.generate()

    # Zero items collected, but structured digest returned without crashing
    assert digest.evidence_urls() == []
    assert len(digest.warnings) >= 2
    assert any("GitHub gagal" in w for w in digest.warnings)
    assert any("arXiv gagal" in w for w in digest.warnings)

    rendered = format_digest(digest)
    assert "Belum ada temuan baru sejak digest terakhir." in rendered
    assert "Data parsial" in rendered


async def test_digest_force_resend_and_dry_run_semantics(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        digest_state_path=str(tmp_path / "radar_flags.sqlite3"),
        digest_topics="agents",
        digest_search_queries="agents",
    )
    repo = RepositoryResult(
        full_name="example/agent-tool",
        description="An agent tool",
        url="https://github.com/example/agent-tool",
        stars=10,
        forks=0,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        updated_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    github_search = MagicMock()
    github_search.search = AsyncMock(
        return_value=SearchBatch(query="agents", source=SourceType.GITHUB, items=[repo])
    )
    arxiv_search = MagicMock()
    arxiv_search.search = AsyncMock(
        return_value=SearchBatch(query="agents", source=SourceType.ARXIV, items=[])
    )
    registry = ToolRegistry(
        github_search=github_search,
        github_analyzer=MagicMock(),
        arxiv_search=arxiv_search,
    )
    store = DigestStore(settings.digest_state_path)
    engine = DigestEngine(settings=settings, tools=registry, store=store)

    # 1. Run digest initially
    digest1 = await engine.generate()
    assert len(digest1.repositories) == 1
    await engine.record_sent(digest1)

    # Cooldown should now be active
    assert await engine.is_due(now=datetime(2026, 8, 28, 1, tzinfo=UTC)) is False

    # 2. Normal run without force/resend -> suppressed
    digest2 = await engine.generate(force=False)
    assert len(digest2.repositories) == 0

    # 3. Resend=True -> bypasses deduplication and resends
    digest3 = await engine.generate(force=True)
    assert len(digest3.repositories) == 1


async def test_balanced_budget_backfill_and_unsent_candidates(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        digest_topics="AI",
        digest_search_queries="AI",
        digest_state_path=tmp_path / "balanced.sqlite3",
    )
    now = datetime.now(UTC)
    repos = [
        RepositoryResult(
            full_name=f"lab/repo-{i}",
            url=f"https://github.com/lab/repo-{i}",
            stars=i,
            forks=0,
            created_at=now,
            updated_at=now,
        )
        for i in range(20)
    ]
    papers = [
        PaperResult(
            arxiv_id=f"2609.{i:05d}",
            title=f"AI paper {i}",
            abstract="AI research",
            authors=["Author"],
            categories=["cs.AI"],
            primary_category="cs.AI",
            published_at=now,
            updated_at=now,
            url=f"https://arxiv.org/abs/2609.{i:05d}",
        )
        for i in range(20)
    ]
    news = [
        NewsResult(title=f"Model release {i}", url=f"https://lab{i}.test/release", published_at=now)
        for i in range(20)
    ]
    registry = ToolRegistry(
        github_search=SimpleNamespace(
            search=AsyncMock(
                return_value=SearchBatch(query="AI", source=SourceType.GITHUB, items=repos)
            )
        ),
        github_analyzer=MagicMock(),
        arxiv_search=SimpleNamespace(
            search=AsyncMock(
                return_value=SearchBatch(query="AI", source=SourceType.ARXIV, items=papers)
            )
        ),
        news_search=SimpleNamespace(
            search=AsyncMock(
                return_value=SearchBatch(query="AI", source=SourceType.NEWS, items=news)
            )
        ),
    )
    engine = DigestEngine(
        settings=settings, tools=registry, store=DigestStore(settings.digest_state_path)
    )
    first = await engine.generate()
    assert (len(first.news), len(first.repositories), len(first.papers)) == (5, 5, 5)
    # A preview or failed send must not consume candidates.
    preview_again = await engine.generate()
    assert preview_again.evidence_urls() == first.evidence_urls()
    await engine.record_sent(first)
    second = await engine.generate()
    assert len(second.evidence_urls()) == 15
    assert set(second.evidence_urls()).isdisjoint(first.evidence_urls())
    # Vacant paper/repo slots are reused for actual news.
    registry.github_search.search.return_value.items = []
    registry.arxiv_search.search.return_value.items = []
    filled = await engine.generate(force=True)
    assert len(filled.news) == 15
    assert not filled.repositories and not filled.papers


async def test_news_search_queries_and_repository_links_are_separate(tmp_path: Path) -> None:
    from research_radar.schemas import WebResult

    settings = AppSettings(
        _env_file=None,
        digest_search_queries="local LLM",
        digest_state_path=tmp_path / "web.sqlite3",
    )
    registry = ToolRegistry(
        github_search=SimpleNamespace(
            search=AsyncMock(
                return_value=SearchBatch(query="local LLM", source=SourceType.GITHUB, items=[])
            )
        ),
        github_analyzer=MagicMock(),
        arxiv_search=SimpleNamespace(
            search=AsyncMock(
                return_value=SearchBatch(query="local LLM", source=SourceType.ARXIV, items=[])
            )
        ),
        web_search=SimpleNamespace(
            search=AsyncMock(
                return_value=SearchBatch(
                    query="models",
                    source=SourceType.WEB,
                    items=[
                        WebResult(
                            description="Model news",
                            title="Repo",
                            url="https://github.com/lab/model",
                        ),
                        WebResult(
                            description="Model news",
                            title="Paper",
                            url="https://arxiv.org/abs/2609.00001",
                        ),
                        WebResult(
                            description="Model news",
                            title="Model launch",
                            url="https://lab.test/launch",
                        ),
                    ],
                )
            )
        ),
    )
    engine = DigestEngine(
        settings=settings, tools=registry, store=DigestStore(settings.digest_state_path)
    )
    result = await engine.generate()
    assert [item.title for item in result.news] == ["Model launch"]
    assert [call.args[0] for call in registry.web_search.search.await_args_list] == list(
        settings.digest_news_queries
    )
