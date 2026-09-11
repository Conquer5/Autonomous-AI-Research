"""Safe, compact Telegram HTML rendering and message splitting."""

from __future__ import annotations

import html
from zoneinfo import ZoneInfo

from research_radar.schemas import (
    DigestItemInsight,
    PaperResult,
    RepositoryAnalysis,
    RepositoryResult,
    ResearchDigest,
    SearchBatch,
    WebResult,
)
from research_radar.service import RadarStatus

TELEGRAM_SAFE_LIMIT = 3900


def truncate(value: str | None, limit: int) -> str:
    normalized = " ".join((value or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def split_plain_text(text: str, limit: int = TELEGRAM_SAFE_LIMIT) -> list[str]:
    """Split without losing characters, preferring paragraph/line/word boundaries."""

    if limit < 1:
        raise ValueError("limit must be positive")
    remaining = text
    chunks: list[str] = []
    while len(remaining) > limit:
        boundary = remaining.rfind("\n\n", 0, limit + 1)
        if boundary < limit // 3:
            boundary = remaining.rfind("\n", 0, limit + 1)
        if boundary < limit // 3:
            boundary = remaining.rfind(" ", 0, limit + 1)
        if boundary <= 0:
            boundary = limit
        chunks.append(remaining[:boundary].rstrip())
        remaining = remaining[boundary:].lstrip()
    if remaining or not chunks:
        chunks.append(remaining)
    return chunks


def split_html_blocks(text: str, limit: int = TELEGRAM_SAFE_LIMIT) -> list[str]:
    """Split formatter-produced HTML at balanced paragraph boundaries."""

    blocks = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for block in blocks:
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(block) > limit:
            # Formatter blocks are expected to be short. Fall back to safe plain text.
            plain = html.unescape(block.replace("<b>", "").replace("</b>", ""))
            chunks.extend(html.escape(part) for part in split_plain_text(plain, limit))
            current = ""
        else:
            current = block
    if current:
        chunks.append(current)
    return chunks or [""]


def _link(url: object, label: str) -> str:
    return f'<a href="{html.escape(str(url), quote=True)}">{html.escape(label)}</a>'


def format_repositories(batch: SearchBatch[RepositoryResult]) -> str:
    blocks = [f"<b>💻 GitHub repositories</b>\nQuery: {html.escape(batch.query)}"]
    if not batch.items:
        blocks.append("Tidak ada repository yang ditemukan.")
    for index, repo in enumerate(batch.items, 1):
        signals = repo.signals
        score_line = ""
        if signals:
            score_line = (
                f"\nRelevance {signals.relevance_score:.2f} · Activity "
                f"{signals.activity_score:.2f} · Depth {signals.technical_depth_score:.2f}"
            )
        blocks.append(
            f"<b>{index}. {_link(repo.url, repo.full_name)}</b>\n"
            f"⭐ {repo.stars:,} · Forks {repo.forks:,} · {html.escape(repo.language or 'n/a')}\n"
            f"{html.escape(truncate(repo.description, 240))}{score_line}"
        )
    return "\n\n".join(blocks)


def format_papers(batch: SearchBatch[PaperResult]) -> str:
    blocks = [f"<b>📚 arXiv papers</b>\nQuery: {html.escape(batch.query)}"]
    if not batch.items:
        blocks.append("Tidak ada paper yang ditemukan pada rentang waktu ini.")
    for index, paper in enumerate(batch.items, 1):
        authors = truncate(", ".join(paper.authors), 150)
        primary_category = html.escape(paper.primary_category or "n/a")
        blocks.append(
            f"<b>{index}. {_link(paper.url, paper.title)}</b>\n"
            f"{paper.published_at.date().isoformat()} · {primary_category}\n"
            f"{html.escape(authors)}\n{html.escape(truncate(paper.abstract, 280))}"
        )
    return "\n\n".join(blocks)


def format_web_results(batch: SearchBatch[WebResult]) -> str:
    blocks = [f"<b>🌐 Web results</b>\nQuery: {html.escape(batch.query)}"]
    if not batch.items:
        blocks.append("Tidak ada hasil web yang ditemukan.")
    for index, item in enumerate(batch.items, 1):
        blocks.append(
            f"<b>{index}. {_link(item.url, item.title)}</b>\n"
            f"Source quality (heuristic): {item.source_quality_score:.2f}\n"
            f"{html.escape(truncate(item.description, 280))}"
        )
    return "\n\n".join(blocks)


def format_repository_analysis(analysis: RepositoryAnalysis) -> str:
    repo = analysis.repository
    release = analysis.latest_release
    release_text = (
        f"{html.escape(release.tag_name)} ({release.published_at.date().isoformat()})"
        if release and release.published_at
        else html.escape(release.tag_name)
        if release
        else "Tidak ada"
    )
    commit_lines = [
        f"• {_link(commit.url, commit.sha[:7])} — {html.escape(truncate(commit.message, 100))}"
        for commit in analysis.recent_commits
    ]
    readme_status = (
        "yes, truncated" if analysis.readme_truncated else "yes" if analysis.readme else "no"
    )
    warning_text = "\n".join(f"• {html.escape(item)}" for item in analysis.warnings)
    blocks = [
        f"<b>🔬 Repository analysis</b>\n{_link(repo.url, repo.full_name)}",
        (
            f"⭐ {repo.stars:,} · Forks {repo.forks:,} · Issues {repo.open_issues:,}\n"
            f"Language: {html.escape(repo.language or 'n/a')}\n"
            f"Latest release: {release_text}\n"
            f"README captured: {readme_status}"
        ),
    ]
    if commit_lines:
        blocks.append("<b>Recent commits</b>\n" + "\n".join(commit_lines))
    if warning_text:
        blocks.append("<b>Partial data</b>\n" + warning_text)
    return "\n\n".join(blocks)


def format_status(status: RadarStatus) -> str:
    runtime_state = (
        "healthy"
        if status.runtime.healthy
        else "unhealthy"
        if status.runtime.healthy is False
        else "unknown"
    )
    tool_lines = [
        f"• {html.escape(tool.name)}: {'configured' if tool.configured else 'not configured'}"
        for tool in status.tools
    ]
    metrics = status.metrics
    return (
        "<b>System Status</b>\n"
        f"Hermes: {runtime_state}\n"
        f"Tasks: {metrics.total_tasks} ({metrics.successful_tasks} success, "
        f"{metrics.failed_tasks} failed)\n"
        f"Runtime calls: {metrics.runtime_calls}\n"
        f"LLM calls: {metrics.llm_calls}\n"
        f"Tool calls: {metrics.tool_calls}\n"
        f"Retries: {metrics.retries}\n"
        f"Average latency: {metrics.average_duration_ms / 1000:.2f}s\n\n"
        "<b>Tools</b>\n" + "\n".join(tool_lines)
    )


def _quality_badge(quality: str) -> str:
    mapping = {"high": "🟢 Bukti: Tinggi", "medium": "🟡 Bukti: Sedang", "low": "⚪ Bukti: Rendah"}
    label = mapping.get(quality, f"Bukti: {quality}")
    return f" · <i>{label}</i>"


def _format_digest_insight(insight: DigestItemInsight) -> str:
    function_text = (
        f"{insight.what_it_is.rstrip('.')} — {insight.how_it_works}"
        if insight.how_it_works
        else insight.what_it_is
    )
    lines = [f"• 📌 <b>Fungsi & Cara Kerja:</b> {html.escape(function_text)}"]

    if insight.tech_stack:
        stack_str = ", ".join(insight.tech_stack)
        lines.append(f"• 🛠️ <b>Stack:</b> {html.escape(stack_str)}")

    why = insight.why_it_matters
    if insight.claim_type == "interpretation" and not why.startswith("Inferensi:"):
        why = f"Inferensi: {why}"
    impact_caveat = f"{why.rstrip('.')} (Catatan: {insight.caveat})"
    lines.append(f"• 💡 <b>Dampak & Catatan:</b> {html.escape(impact_caveat)}")

    return "\n".join(lines)


def _fallback_source_summary(value: str | None) -> str:
    return "• <b>Ringkasan sumber (belum dianalisis AI):</b> " + html.escape(
        truncate(value, 260) or "Tidak tersedia."
    )


def format_digest(digest: ResearchDigest) -> str:
    generated = digest.generated_at.astimezone(ZoneInfo("Asia/Jakarta"))
    blocks = [
        "<b>🔥 Autonomous AI Research Radar</b>\n"
        f"{generated:%d-%m-%Y %H:%M} WIB · berita, repository, dan riset terbaru"
    ]
    if digest.overview:
        blocks.append(f"<b>Ringkasan AI</b>\n{html.escape(truncate(digest.overview, 1000))}")

    if digest.news:
        lines = ["<b>📰 Perkembangan teknologi</b>"]
        for index, item in enumerate(digest.news, 1):
            date_text = item.published_at.date().isoformat() if item.published_at else "tanggal n/a"
            source = html.escape(item.source_name or item.source.value)
            quality_badge = _quality_badge(item.insight.evidence_quality) if item.insight else ""
            explanation = (
                _format_digest_insight(item.insight)
                if item.insight
                else _fallback_source_summary(item.description)
            )
            lines.append(
                f"<b>{index}. {_link(item.url, item.title)}</b>\n"
                f"{date_text} · {source}{quality_badge}\n{explanation}"
            )
        blocks.append("\n\n".join(lines))

    if digest.repositories:
        lines = ["<b>💻 Repository GitHub baru</b>"]
        for index, repo in enumerate(digest.repositories, 1):
            license_text = html.escape(repo.license_name or "lisensi belum terdeteksi")
            quality_badge = _quality_badge(repo.insight.evidence_quality) if repo.insight else ""
            explanation = (
                _format_digest_insight(repo.insight)
                if repo.insight
                else _fallback_source_summary(repo.description)
            )
            lang = html.escape(repo.language or "n/a")
            lines.append(
                f"<b>{index}. {_link(repo.url, repo.full_name)}</b>\n"
                f"⭐ {repo.stars:,} · {lang} · {license_text}{quality_badge}\n"
                f"{explanation}"
            )
        blocks.append("\n\n".join(lines))

    if digest.papers:
        lines = ["<b>📚 Paper arXiv terbaru</b>"]
        for index, paper in enumerate(digest.papers, 1):
            quality_badge = _quality_badge(paper.insight.evidence_quality) if paper.insight else ""
            explanation = (
                _format_digest_insight(paper.insight)
                if paper.insight
                else _fallback_source_summary(paper.abstract)
            )
            lines.append(
                f"<b>{index}. {_link(paper.url, paper.title)}</b>\n"
                f"{paper.published_at.date().isoformat()} · "
                f"{html.escape(paper.primary_category or 'n/a')}{quality_badge}\n"
                f"{explanation}"
            )
        blocks.append("\n\n".join(lines))

    if not digest.evidence_urls():
        blocks.append("Belum ada temuan baru sejak digest terakhir.")
    if digest.warnings:
        warning_lines = "\n".join(
            f"• {html.escape(truncate(item, 180))}" for item in digest.warnings[:4]
        )
        blocks.append(f"<b>Data parsial</b>\n{warning_lines}")
    if digest.synthesis_model:
        blocks.append(f"Sintesis: <code>{html.escape(digest.synthesis_model)}</code>")
    return "\n\n".join(blocks)


def format_research_result(result: object) -> str:
    """Format structured autonomous research result for Telegram."""
    from research_radar.consensus.models import ConsensusLevel
    from research_radar.research.models import ResearchMode, ResearchSynthesisResult

    if not isinstance(result, ResearchSynthesisResult):
        return html.escape(str(result))

    mode_label = "Mendalam (Deep Research)" if result.mode == ResearchMode.DEEP else "Cepat (Quick)"
    consensus_badge = {
        ConsensusLevel.STRONG: "🟢 Konsensus Kuat",
        ConsensusLevel.MODERATE: "🟡 Konsensus Moderat",
        ConsensusLevel.MIXED: "🟠 Bukti Beragam / Disputed",
        ConsensusLevel.WEAK: "⚪ Konsensus Lemah",
        ConsensusLevel.INSUFFICIENT: "⚪ Bukti Belum Cukup",
    }.get(result.consensus, "⚪ Belum Ditentukan")

    blocks = [
        f"<b>🔬 Autonomous AI Research · {mode_label}</b>\n"
        f"<b>Fokus:</b> {html.escape(result.question)}\n"
        f"<b>Status Konsensus:</b> {consensus_badge}"
    ]

    if result.state.focus is not None:
        focus = result.state.focus
        if focus.since:
            blocks.append(
                f"<b>Jendela sumber:</b> {focus.since} – {focus.as_of} (UTC)\n"
                + html.escape(focus.window_note)
            )
        else:
            blocks.append(
                f"<b>Diperiksa:</b> {focus.as_of} (UTC); sumber historis dapat disertakan."
            )

    if result.answer:
        blocks.append(f"<b>Ringkasan Riset</b>\n{html.escape(result.answer)}")

    if result.agreements:
        agreements_lines = [f"• {html.escape(a)}" for a in result.agreements[:3]]
        blocks.append("<b>🤝 Kesepakatan Sumber</b>\n" + "\n".join(agreements_lines))

    if result.disagreements or result.dissenting_findings:
        disagreement_lines = []
        for d in result.disagreements[:2]:
            disagreement_lines.append(f"• {html.escape(d)}")
        for dissent in result.dissenting_findings[:2]:
            disagreement_lines.append(f"• <i>{html.escape(dissent)}</i>")
        if disagreement_lines:
            blocks.append(
                "<b>⚡ Perbedaan & Pandangan Dissent</b>\n" + "\n".join(disagreement_lines)
            )

    if result.safe_conclusion:
        blocks.append(f"<b>🛡️ Kesimpulan Konservatif</b>\n{html.escape(result.safe_conclusion)}")

    if result.key_findings:
        finding_lines = []
        for f in result.key_findings[:5]:
            if f.startswith("Inferensi:"):
                finding_lines.append(f"• <i>{html.escape(f)}</i>")
            else:
                finding_lines.append(f"• {html.escape(f)}")
        blocks.append("<b>💡 Temuan Kunci</b>\n" + "\n".join(finding_lines))

    if result.evidence_sources:
        evidence_lines = []
        for item in result.evidence_sources[:6]:
            url = str(item.get("url", ""))
            title = str(item.get("title", "")) or url
            auth = str(item.get("authority", "sumber")).upper()
            if url:
                evidence_lines.append(f"• {_link(url, title)} (<code>{html.escape(auth)}</code>)")
            else:
                evidence_lines.append(f"• {html.escape(title)}")
        blocks.append("<b>📚 Sumber Bukti Terverifikasi</b>\n" + "\n".join(evidence_lines))

    if result.uncertainties:
        uncertainty_lines = [f"• {html.escape(u)}" for u in result.uncertainties[:3]]
        blocks.append("<b>⚠️ Keterbatasan & Ketidakpastian</b>\n" + "\n".join(uncertainty_lines))

    confidence_badge = {
        "high": "Tinggi 🟢",
        "medium": "Sedang 🟡",
        "low": "Awal 🔴",
    }.get(result.confidence.value, result.confidence.value)

    state = result.state
    blocks.append(
        f"<i>Status: {html.escape(result.stop_reason.value)} · "
        f"Iterasi: {state.iterations}/{state.budget.max_iterations} · "
        f"Bukti: {len(result.evidence_sources)} · "
        f"Tingkat Keyakinan: {confidence_badge}</i>"
    )

    return "\n\n".join(blocks)
