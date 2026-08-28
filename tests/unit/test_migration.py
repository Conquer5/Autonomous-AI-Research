"""Tests for Schema Migration Safety and Backward Compatibility."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from research_radar.evidence.registry import EvidenceRegistry


def test_fresh_schema_creation(tmp_path: Path) -> None:
    db_path = tmp_path / "fresh.sqlite3"
    registry = EvidenceRegistry(db_path)
    assert registry.path == db_path

    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

    assert "schema_version" in tables
    assert "evidence" in tables
    assert "evidence_versions" in tables
    assert "delivery_history" in tables
    assert "digest_runs_v2" in tables
    assert "digest_lock" in tables


def test_v1_to_v2_migration_preserves_existing_data(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"

    # 1. Create a legacy v1 database
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE seen_evidence (
                url TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                first_sent_at REAL NOT NULL,
                last_sent_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE digest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sent_at REAL NOT NULL,
                item_count INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO seen_evidence VALUES (?, ?, ?, ?)",
            ("https://github.com/example/legacy-repo", "github", 1720000000.0, 1720000100.0),
        )
        conn.execute(
            "INSERT INTO digest_runs(sent_at, item_count) VALUES (?, ?)",
            (1720000100.0, 1),
        )
        conn.commit()

    # 2. Instantiate EvidenceRegistry which triggers migration
    registry = EvidenceRegistry(db_path)

    # 3. Check that legacy data is preserved in new evidence table
    record = registry.get_by_url("https://github.com/example/legacy-repo")
    assert record is not None
    assert record.canonical_url == "https://github.com/example/legacy-repo"
    assert record.source_type == "github"

    # Check that legacy tables are still intact
    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM seen_evidence")
        assert cursor.fetchone()[0] == 1
        cursor.execute("SELECT count(*) FROM digest_runs")
        assert cursor.fetchone()[0] == 1


def test_migration_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "idempotent.sqlite3"

    # Initialize once
    registry1 = EvidenceRegistry(db_path)
    assert registry1.path == db_path

    # Initialize second time on same DB (should not fail or duplicate)
    registry2 = EvidenceRegistry(db_path)
    assert registry2.path == db_path

    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM schema_version")
        assert cursor.fetchone()[0] == 1


def test_legacy_migrated_evidence_first_rediscovery_is_duplicate(tmp_path: Path) -> None:
    db_path = tmp_path / "rediscover.sqlite3"

    # 1. Create legacy DB with a seen URL
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE seen_evidence (
                url TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                first_sent_at REAL NOT NULL,
                last_sent_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE digest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sent_at REAL NOT NULL,
                item_count INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO seen_evidence VALUES (?, ?, ?, ?)",
            ("https://github.com/example/old-tool", "github", 1720000000.0, 1720000100.0),
        )
        conn.commit()

    # 2. Run migration
    registry = EvidenceRegistry(db_path)

    # 3. Rediscover the legacy evidence for the first time
    meta = {
        "description": "An existing tool discovered previously",
        "readme_excerpt": "Hello world",
        "latest_release_tag": "v1.0.0",
    }
    from research_radar.evidence.canonicalizer import content_fingerprint
    from research_radar.evidence.models import EvidenceStatus

    fingerprint = content_fingerprint("github", meta)
    record, status = registry.register(
        url="https://github.com/example/old-tool",
        source_type="github",
        title="example/old-tool",
        content_hash=fingerprint,
        metadata=meta,
    )

    # Must be DUPLICATE so users are NOT spammed with legacy items
    assert status == EvidenceStatus.DUPLICATE
    assert record.content_hash == fingerprint

    # Filter unseen should also filter it out
    filter_res = registry.filter_unseen_or_updated(
        [("https://github.com/example/old-tool", "github", fingerprint)]
    )
    assert filter_res["https://github.com/example/old-tool"] == EvidenceStatus.DUPLICATE
