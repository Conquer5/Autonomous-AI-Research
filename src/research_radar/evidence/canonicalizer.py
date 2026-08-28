from __future__ import annotations

import hashlib
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


def canonicalize_url(url: str) -> str:
    """Normalize URL for deduplication.

    - Strip fragments (#...)
    - Strip trailing slashes
    - Normalize http -> https where applicable
    - Remove common tracking params (utm_*, ref, source, fbclid, gclid)
    - Normalize github.com URLs (strip .git suffix)
    - Lowercase the hostname
    """
    if not url:
        return url

    parsed = urlparse(url)
    scheme = "https" if parsed.scheme in ("http", "https") else parsed.scheme
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")

    if netloc == "github.com" and path.endswith(".git"):
        path = path[:-4]

    query_params = parse_qsl(parsed.query, keep_blank_values=True)
    filtered_query = []
    tracking_prefixes = ("utm_", "ref", "source", "fbclid", "gclid")

    for k, v in query_params:
        if not any(k.lower().startswith(prefix) for prefix in tracking_prefixes):
            filtered_query.append((k, v))

    query = urlencode(filtered_query)

    # urlunparse expects: scheme, netloc, path, params, query, fragment
    canonical = urlunparse((scheme, netloc, path, parsed.params, query, ""))
    return canonical


def content_fingerprint(source_type: str, metadata: dict[str, object]) -> str:
    """Generate SHA-256 content fingerprint for evidence.

    Source-aware hashing:
    - github: hash(description + readme_excerpt + latest_release_tag + topics + license)
    - arxiv: hash(title + abstract + categories + updated_at_iso)
    - news/web: hash(title + description)

    Volatile metrics (stars, forks, watchers, temporary scores) are explicitly
    excluded to prevent false version updates.
    """
    hasher = hashlib.sha256()

    if source_type == "github":
        description = str(metadata.get("description", ""))
        readme = str(metadata.get("readme_excerpt", ""))
        release = str(metadata.get("latest_release_tag", ""))
        raw_topics = metadata.get("topics")
        topics_list = (
            [str(t) for t in raw_topics] if isinstance(raw_topics, (list, tuple, set)) else []
        )
        topics = str(sorted(topics_list))
        license_name = str(metadata.get("license", ""))
        payload = f"{description}|{readme}|{release}|{topics}|{license_name}"
    elif source_type == "arxiv":
        title = str(metadata.get("title", ""))
        abstract = str(metadata.get("abstract", ""))
        raw_cats = metadata.get("categories")
        cats_list = [str(c) for c in raw_cats] if isinstance(raw_cats, (list, tuple, set)) else []
        categories = str(sorted(cats_list))
        updated = str(metadata.get("updated_at_iso", ""))
        payload = f"{title}|{abstract}|{categories}|{updated}"
    else:  # news/web
        title = str(metadata.get("title", ""))
        description = str(metadata.get("description", ""))
        payload = f"{title}|{description}"

    hasher.update(payload.encode("utf-8"))
    return hasher.hexdigest()
