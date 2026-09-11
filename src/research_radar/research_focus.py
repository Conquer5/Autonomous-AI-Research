"""Bounded research focus and temporal evidence policy; no network or model calls."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from importlib.resources import files

from pydantic import BaseModel, Field


class ResearchFocus(BaseModel):
    track: str = "general"
    skill: str = "research-methodology"
    skill_version: str = "1.0.0"
    as_of: date = Field(default_factory=lambda: datetime.now(UTC).date())
    since: date | None = None
    window_note: str = ""


def resolve_focus(
    question: str, *, recent_days: int = 14, today: date | None = None
) -> ResearchFocus:
    """Recognize explicit rolling windows; preserve historical/general research access."""
    today = today or datetime.now(UTC).date()
    text = question.lower()
    focus = ResearchFocus(as_of=today)
    if re.search(r"\b(token|codex|coding|hemat|efisien|efficiency|mcp|optimasi)\b", text):
        focus.track, focus.skill = "agent_efficiency", "agent-efficiency"
    elif re.search(r"\b(model|gemini|gpt|gratis|free|pricing|harga|kuota)\b", text):
        focus.track, focus.skill = "model_access", "model-access"
    elif re.search(r"\b(ai|rilis|release|perkembangan|news|berita)\b", text):
        focus.track, focus.skill = "ai_developments", "ai-developments"

    match = re.search(r"\b(\d+)\s*(hari|days?|minggu|weeks?)\s*(terakhir|last)?\b", text)
    days: int | None = None
    if match:
        days = int(match[1]) * (7 if match[2].startswith(("minggu", "week")) else 1)
    elif re.search(r"\b(hari ini|today)\b", text):
        days = 1
    elif re.search(r"\b(minggu ini|pekan ini|this week)\b", text):
        days = 7
    elif re.search(r"\b(terbaru|terkini|sekarang|latest|recent|current|baru)\b", text):
        days = recent_days
    if days is not None:
        bounded_days = max(1, min(days, 365))
        focus.since = today - timedelta(days=bounded_days - 1)
        focus.window_note = "Tanggal GitHub menandai aktivitas kode, bukan tanggal rilis."
        if bounded_days != days:
            focus.window_note += " Jendela dibatasi menjadi 1–365 hari."
    return focus


def skill_text(skill: str) -> str:
    if skill not in SKILLS:
        raise ValueError("Unknown research skill")
    return files("research_radar").joinpath("skills", skill, "SKILL.md").read_text("utf-8")


SKILLS = {
    "research-methodology": "Perencanaan riset berbasis bukti",
    "ai-developments": "Perkembangan AI dan kebaruan sumber",
    "agent-efficiency": "Efisiensi coding agent dan rancangan eksperimen",
    "model-access": "Model, harga, kuota, dan akses gratis",
}

RESEARCH_DECISION_POLICY = """Answer in concise Indonesian. Explain practical relevance to
Codex/Hermes/Gemini only when supported. Separate developer claims, independent measurements,
and untested hypotheses. Never assert that skills make Flash generally equal to a stronger model.
For efficiency, compare cost per successful task, correctness, total tokens, retries, latency,
and human intervention on matched tasks; never invent savings or test results. For free models,
distinguish free API quota, trial credits and open weights; require current official evidence
for model ID, price, quota, access conditions and hardware. Unknown means unverified.
Recommend a small reproducible experiment when evidence is insufficient, clearly labeled proposed.
Repository activity is not a new release. Retrieval time is not publication time. An undated
source does not prove recency, and a search snippet does not verify a price or benchmark.
"""


def temporal_status(value: datetime | None, focus: ResearchFocus) -> str:
    if value is None:
        return "undated"
    day = value.replace(tzinfo=UTC).date() if value.tzinfo is None else value.astimezone(UTC).date()
    if day > focus.as_of:
        return "future"
    if focus.since and day < focus.since:
        return "outside_window"
    return "in_window"
