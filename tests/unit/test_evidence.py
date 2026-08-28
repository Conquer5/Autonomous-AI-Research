"""Comprehensive tests for Evidence Registry, Canonicalization, Quality, and Versioning."""

from __future__ import annotations

from pathlib import Path

from research_radar.evidence.canonicalizer import canonicalize_url, content_fingerprint
from research_radar.evidence.models import (
    EvidenceStatus,
    SourceAuthority,
)
from research_radar.evidence.quality import classify_source_authority
from research_radar.evidence.registry import EvidenceRegistry


def test_canonicalize_url_normalizes_and_strips_tracking() -> None:
    # Fragment stripping
    assert (
        canonicalize_url("https://github.com/fastapi/fastapi#readme")
        == "https://github.com/fastapi/fastapi"
    )
    # Trailing slash
    assert (
        canonicalize_url("https://arxiv.org/abs/2608.12345/") == "https://arxiv.org/abs/2608.12345"
    )
    # Git suffix stripping for github
    assert (
        canonicalize_url("https://github.com/google/gemma.git") == "https://github.com/google/gemma"
    )
    # HTTP to HTTPS
    assert canonicalize_url("http://deepmind.google/research") == "https://deepmind.google/research"
    # Tracking parameters removal
    dirty = (
        "https://techcrunch.com/article?utm_source=twitter&utm_medium=social&ref=newsletter&id=123"
    )
    clean = canonicalize_url(dirty)
    assert "utm_source" not in clean
    assert "utm_medium" not in clean
    assert "ref" not in clean
    assert "id=123" in clean


def test_content_fingerprint_deterministic_and_source_aware() -> None:
    gh_meta_1 = {
        "description": "Fast AI Agent",
        "readme_excerpt": "Hello",
        "latest_release_tag": "v1.0",
    }
    gh_meta_2 = {
        "description": "Fast AI Agent",
        "readme_excerpt": "Hello",
        "latest_release_tag": "v1.0",
    }
    gh_meta_3 = {
        "description": "Fast AI Agent",
        "readme_excerpt": "Hello v2",
        "latest_release_tag": "v1.1",
    }

    fp1 = content_fingerprint("github", gh_meta_1)
    fp2 = content_fingerprint("github", gh_meta_2)
    fp3 = content_fingerprint("github", gh_meta_3)

    assert fp1 == fp2
    assert fp1 != fp3
    assert len(fp1) == 64  # SHA-256 hex string


def test_classify_source_authority_deterministic() -> None:
    auth, score = classify_source_authority("github", "https://github.com/google/research")
    assert auth == SourceAuthority.PRIMARY
    assert score >= 0.90

    auth, score = classify_source_authority("github", "https://github.com/randomuser/project")
    assert auth == SourceAuthority.COMMUNITY
    assert score == 0.60

    auth, score = classify_source_authority("arxiv", "https://arxiv.org/abs/2608.00001")
    assert auth == SourceAuthority.ACADEMIC
    assert score == 0.90

    auth, score = classify_source_authority("web", "https://docs.anthropic.com/en/docs")
    assert auth == SourceAuthority.OFFICIAL
    assert score >= 0.90

    auth, score = classify_source_authority("news", "https://techcrunch.com/2026/08/ai-news")
    assert auth == SourceAuthority.SECONDARY
    assert score == 0.65


def test_evidence_registry_lifecycle_and_versioning(tmp_path: Path) -> None:
    db_path = tmp_path / "test_registry.sqlite3"
    registry = EvidenceRegistry(db_path)

    url = "https://github.com/example/research-core"
    meta_v1 = {"description": "v1 description", "stars": 100}
    hash_v1 = content_fingerprint("github", meta_v1)

    # 1. Register NEW evidence
    record1, status1 = registry.register(
        url=url,
        source_type="github",
        title="example/research-core",
        content_hash=hash_v1,
        source_authority=SourceAuthority.COMMUNITY,
        authority_score=0.60,
        metadata=meta_v1,
    )
    assert status1 == EvidenceStatus.NEW
    assert record1.canonical_url == url
    assert record1.content_hash == hash_v1

    # 2. Register DUPLICATE (same content hash)
    record2, status2 = registry.register(
        url=url,
        source_type="github",
        title="example/research-core",
        content_hash=hash_v1,
        metadata=meta_v1,
    )
    assert status2 == EvidenceStatus.DUPLICATE
    assert record2.evidence_id == record1.evidence_id

    # 3. Register UPDATED (changed content hash)
    meta_v2 = {"description": "v2 major release description", "stars": 500}
    hash_v2 = content_fingerprint("github", meta_v2)
    record3, status3 = registry.register(
        url=url,
        source_type="github",
        title="example/research-core",
        content_hash=hash_v2,
        metadata=meta_v2,
    )
    assert status3 == EvidenceStatus.UPDATED
    assert record3.evidence_id == record1.evidence_id
    assert record3.content_hash == hash_v2

    # 4. Verify version history preserves snapshots
    versions = registry.get_versions(record1.evidence_id)
    assert len(versions) == 2
    assert versions[0].content_hash == hash_v1
    assert versions[1].content_hash == hash_v2


def test_evidence_registry_delivery_history(tmp_path: Path) -> None:
    db_path = tmp_path / "test_delivery.sqlite3"
    registry = EvidenceRegistry(db_path)

    url = "https://arxiv.org/abs/2608.11111"
    meta = {"abstract": "Test abstract"}
    content_hash = content_fingerprint("arxiv", meta)

    record, _ = registry.register(
        url=url,
        source_type="arxiv",
        title="Test Paper",
        content_hash=content_hash,
        metadata=meta,
    )

    assert not registry.has_been_delivered(record.evidence_id)
    undelivered = registry.get_undelivered()
    assert len(undelivered) == 1
    assert undelivered[0].evidence_id == record.evidence_id

    # Record delivery
    delivery = registry.record_delivery(
        run_id="run-123",
        evidence_id=record.evidence_id,
        evidence_version=content_hash,
        delivery_status="sent",
    )
    assert delivery.run_id == "run-123"
    assert delivery.evidence_id == record.evidence_id
    assert registry.has_been_delivered(record.evidence_id)

    # Verify no longer undelivered
    assert len(registry.get_undelivered()) == 0


def test_volatile_metrics_do_not_trigger_updated_version(tmp_path: Path) -> None:
    db_path = tmp_path / "test_volatile.sqlite3"
    registry = EvidenceRegistry(db_path)
    url = "https://github.com/example/fast-rag"

    meta_t0 = {
        "description": "Fast RAG agent framework",
        "readme_excerpt": "Initial readme documentation",
        "latest_release_tag": "v1.0.0",
        "topics": ["rag", "ai"],
        "license": "MIT",
        "stars": 50,
        "forks": 5,
    }
    hash_t0 = content_fingerprint("github", meta_t0)

    # Day 0: Register initial
    rec0, status0 = registry.register(
        url=url,
        source_type="github",
        title="example/fast-rag",
        content_hash=hash_t0,
        metadata=meta_t0,
    )
    assert status0 == EvidenceStatus.NEW

    # Day 1: Stars increase from 50 to 5000 (viral repo), but content is unchanged
    meta_t1 = dict(meta_t0)
    meta_t1["stars"] = 5000
    meta_t1["forks"] = 600
    hash_t1 = content_fingerprint("github", meta_t1)

    # Fingerprint must be IDENTICAL despite 100x star growth
    assert hash_t1 == hash_t0

    rec1, status1 = registry.register(
        url=url,
        source_type="github",
        title="example/fast-rag",
        content_hash=hash_t1,
        metadata=meta_t1,
    )
    # Must remain DUPLICATE (no false update alert)
    assert status1 == EvidenceStatus.DUPLICATE
    assert len(registry.get_versions(rec0.evidence_id)) == 1

    # Day 2: Substantive update (v2.0.0 released with new README)
    meta_t2 = dict(meta_t0)
    meta_t2["readme_excerpt"] = "New v2 architecture with streaming reasoning"
    meta_t2["latest_release_tag"] = "v2.0.0"
    hash_t2 = content_fingerprint("github", meta_t2)

    assert hash_t2 != hash_t0

    rec2, status2 = registry.register(
        url=url,
        source_type="github",
        title="example/fast-rag",
        content_hash=hash_t2,
        metadata=meta_t2,
    )
    # Must trigger UPDATED
    assert status2 == EvidenceStatus.UPDATED
    assert len(registry.get_versions(rec0.evidence_id)) == 2
