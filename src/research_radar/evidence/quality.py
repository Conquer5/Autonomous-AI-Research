from __future__ import annotations

from urllib.parse import urlparse

from .models import SourceAuthority


def classify_source_authority(
    source_type: str,
    url: str,
    metadata: dict[str, object] | None = None,
) -> tuple[SourceAuthority, float]:
    """Classify source authority and return (authority, score).

    Rules:
    - github repos by known orgs (google, meta, microsoft, openai, etc) -> PRIMARY, 0.95
    - github repos general -> COMMUNITY, 0.60
    - arxiv papers -> ACADEMIC, 0.90
    - Official documentation sites (docs.*, developer.*) -> OFFICIAL, 0.92
    - First-party blogs (deepmind.google, research.google, huggingface.co/blog) -> PRIMARY, 0.88
    - Known tech news (techcrunch, wired, arstechnica, theverge) -> SECONDARY, 0.65
    - General news/web -> SECONDARY, 0.55
    - edu/gov domains -> OFFICIAL, 0.85
    - Unknown -> UNKNOWN, 0.40
    """
    if not url:
        return SourceAuthority.UNKNOWN, 0.40

    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    path = parsed.path.lower()

    # Github
    if source_type == "github" or netloc == "github.com":
        known_orgs = {"google", "meta", "microsoft", "openai", "deepmind", "anthropic"}
        parts = path.strip("/").split("/")
        if len(parts) >= 1 and parts[0] in known_orgs:
            return SourceAuthority.PRIMARY, 0.95
        return SourceAuthority.COMMUNITY, 0.60

    # Arxiv
    if source_type == "arxiv" or "arxiv.org" in netloc:
        return SourceAuthority.ACADEMIC, 0.90

    # First-party / Known Orgs Blogs
    primary_blogs = [
        "deepmind.google",
        "research.google",
        "openai.com/research",
        "openai.com/blog",
        "huggingface.co/blog",
        "ai.meta.com/blog",
    ]
    for pb in primary_blogs:
        if pb in netloc or (netloc in ("huggingface.co", "openai.com") and "/blog" in path):
            return SourceAuthority.PRIMARY, 0.88

    # Official docs
    if (
        netloc.startswith("docs.")
        or netloc.startswith("developer.")
        or netloc.startswith("developers.")
    ):
        return SourceAuthority.OFFICIAL, 0.92

    # EDU / GOV
    if netloc.endswith(".edu") or netloc.endswith(".gov") or ".edu." in netloc or ".gov." in netloc:
        return SourceAuthority.OFFICIAL, 0.85

    # Known Tech News
    tech_news = ["techcrunch.com", "wired.com", "arstechnica.com", "theverge.com"]
    if any(news in netloc for news in tech_news):
        return SourceAuthority.SECONDARY, 0.65

    # General news/web if valid url
    if source_type in ("news", "web") or "." in netloc:
        return SourceAuthority.SECONDARY, 0.55

    return SourceAuthority.UNKNOWN, 0.40
