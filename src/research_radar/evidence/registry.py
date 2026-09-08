from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from research_radar.utils.deadline import bounded_timeout

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]

from .canonicalizer import canonicalize_url
from .models import (
    DeliveryRecord,
    DigestRunRecord,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceVersion,
    SourceAuthority,
)


def _parse_datetime(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(float(val), tz=UTC)
    if isinstance(val, str):
        val_str = val.strip()
        try:
            ts = float(val_str)
            return datetime.fromtimestamp(ts, tz=UTC)
        except ValueError:
            dt = datetime.fromisoformat(val_str)
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=UTC)
    return None


class EvidenceRegistry:
    """Content-aware evidence registry with version tracking."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock_file = self.path.parent / f"{self.path.stem}.lock"
        self._lock_fd: int | None = None
        self._ensure_schema_if_needed()

    def _connect(self) -> sqlite3.Connection:
        """Connect and ensure schema exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=bounded_timeout(30.0))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema_if_needed(self) -> None:
        with contextlib.closing(self._connect()) as conn:
            self._ensure_schema(conn)

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Create tables and run migrations."""
        cursor = conn.cursor()

        # Check if schema_version exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        if not cursor.fetchone():
            self._create_tables(conn)
            # Potentially migrate from v1 (seen_evidence, digest_runs) if they exist
            self._migrate_v1_data(conn)
        else:
            cursor.execute("SELECT version FROM schema_version ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row and row["version"] < 2:
                # Add migrations here when needed
                pass

        conn.commit()

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY,
                version INTEGER NOT NULL,
                applied_at TIMESTAMP NOT NULL
            );

            INSERT INTO schema_version (version, applied_at) VALUES (2, CURRENT_TIMESTAMP);

            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id TEXT PRIMARY KEY,
                canonical_url TEXT UNIQUE NOT NULL,
                source_type TEXT NOT NULL,
                title TEXT,
                first_seen_at TIMESTAMP NOT NULL,
                last_seen_at TIMESTAMP NOT NULL,
                content_hash TEXT NOT NULL,
                published_at TIMESTAMP,
                updated_at TIMESTAMP,
                source_authority TEXT,
                authority_score REAL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS evidence_versions (
                version_id TEXT PRIMARY KEY,
                evidence_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                captured_at TIMESTAMP NOT NULL,
                metadata_snapshot TEXT,
                FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id)
            );

            CREATE TABLE IF NOT EXISTS delivery_history (
                delivery_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                evidence_id TEXT NOT NULL,
                evidence_version TEXT,
                delivered_at TIMESTAMP NOT NULL,
                delivery_status TEXT NOT NULL,
                FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id)
            );

            CREATE TABLE IF NOT EXISTS digest_runs_v2 (
                run_id TEXT PRIMARY KEY,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                mode TEXT,
                force BOOLEAN,
                dry_run BOOLEAN,
                sources_attempted INTEGER,
                sources_succeeded INTEGER,
                sources_failed INTEGER,
                items_collected INTEGER,
                items_new INTEGER,
                items_updated INTEGER,
                items_duplicate INTEGER,
                items_ranked INTEGER,
                items_published INTEGER,
                status TEXT
            );

            CREATE TABLE IF NOT EXISTS digest_lock (
                id INTEGER PRIMARY KEY,
                locked_by TEXT NOT NULL,
                locked_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_evidence_url ON evidence(canonical_url);
            CREATE INDEX IF NOT EXISTS idx_delivery_evidence ON delivery_history(evidence_id);
            CREATE INDEX IF NOT EXISTS idx_delivery_run ON delivery_history(run_id);
        """)

    def _migrate_v1_data(self, conn: sqlite3.Connection) -> None:
        """Migrate data from old seen_evidence and digest_runs tables."""
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='seen_evidence'")
        if not cursor.fetchone():
            return

        # Migrate seen_evidence → evidence (v1 columns: url, source, first_sent_at, last_sent_at)
        cursor.execute("""
            INSERT OR IGNORE INTO evidence (
                evidence_id, canonical_url, source_type, title,
                first_seen_at, last_seen_at, content_hash,
                source_authority, authority_score, metadata
            )
            SELECT
                hex(randomblob(16)), url, source, '',
                first_sent_at, last_sent_at, '',
                'unknown', 0.5, '{}'
            FROM seen_evidence
        """)

        # Migrate digest_runs → digest_runs_v2 (v1 columns: id, sent_at, item_count)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='digest_runs'")
        if cursor.fetchone():
            cursor.execute("""
                INSERT OR IGNORE INTO digest_runs_v2 (
                    run_id, started_at, completed_at, status, items_published
                )
                SELECT
                    'legacy-' || CAST(id AS TEXT), sent_at, sent_at, 'completed', item_count
                FROM digest_runs
            """)

    def register(
        self,
        *,
        url: str,
        source_type: str,
        title: str = "",
        content_hash: str = "",
        published_at: datetime | None = None,
        source_authority: SourceAuthority = SourceAuthority.UNKNOWN,
        authority_score: float = 0.5,
        metadata: dict[str, object] | None = None,
    ) -> tuple[EvidenceRecord, EvidenceStatus]:
        """Register evidence, returning record and status (NEW/UPDATED/DUPLICATE)."""
        c_url = canonicalize_url(url)
        meta_str = json.dumps(metadata or {})
        now = datetime.now(UTC)

        with contextlib.closing(self._connect()) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM evidence WHERE canonical_url = ?", (c_url,))
            row = cursor.fetchone()

            if not row:
                evidence_id = uuid4().hex
                cursor.execute(
                    """
                    INSERT INTO evidence (
                        evidence_id, canonical_url, source_type, title,
                        first_seen_at, last_seen_at, content_hash, published_at,
                        source_authority, authority_score, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        evidence_id,
                        c_url,
                        source_type,
                        title,
                        now.isoformat(),
                        now.isoformat(),
                        content_hash,
                        published_at.isoformat() if published_at else None,
                        source_authority.value,
                        authority_score,
                        meta_str,
                    ),
                )

                # Insert initial version
                cursor.execute(
                    """
                    INSERT INTO evidence_versions (
                        version_id, evidence_id, content_hash, captured_at, metadata_snapshot
                    ) VALUES (?, ?, ?, ?, ?)
                """,
                    (uuid4().hex, evidence_id, content_hash, now.isoformat(), meta_str),
                )

                conn.commit()
                status = EvidenceStatus.NEW

                record = EvidenceRecord(
                    evidence_id=evidence_id,
                    canonical_url=c_url,
                    source_type=source_type,
                    title=title,
                    first_seen_at=now,
                    last_seen_at=now,
                    content_hash=content_hash,
                    published_at=published_at,
                    source_authority=source_authority,
                    authority_score=authority_score,
                    metadata=metadata or {},
                )

                return record, status

            evidence_id = row["evidence_id"]
            existing_hash = row["content_hash"]
            first_seen_at = _parse_datetime(row["first_seen_at"]) or now

            if not existing_hash:
                # Conservative legacy migration policy:
                # First rediscovery of a legacy migrated URL establishes the baseline
                # content_hash without falsely triggering an UPDATED alert to users.
                cursor.execute(
                    """
                    UPDATE evidence SET 
                        last_seen_at = ?,
                        content_hash = ?,
                        title = COALESCE(NULLIF(title, ''), ?),
                        metadata = ?
                    WHERE evidence_id = ?
                """,
                    (now.isoformat(), content_hash, title, meta_str, evidence_id),
                )
                cursor.execute(
                    """
                    INSERT INTO evidence_versions (
                        version_id, evidence_id, content_hash, captured_at, metadata_snapshot
                    ) VALUES (?, ?, ?, ?, ?)
                """,
                    (uuid4().hex, evidence_id, content_hash, now.isoformat(), meta_str),
                )
                conn.commit()
                status = EvidenceStatus.DUPLICATE
            elif existing_hash == content_hash:
                cursor.execute(
                    """
                    UPDATE evidence SET last_seen_at = ? WHERE evidence_id = ?
                """,
                    (now.isoformat(), evidence_id),
                )
                conn.commit()
                status = EvidenceStatus.DUPLICATE
            else:
                cursor.execute(
                    """
                    UPDATE evidence SET 
                        last_seen_at = ?,
                        content_hash = ?,
                        title = ?,
                        updated_at = ?,
                        metadata = ?
                    WHERE evidence_id = ?
                """,
                    (now.isoformat(), content_hash, title, now.isoformat(), meta_str, evidence_id),
                )

                cursor.execute(
                    """
                    INSERT INTO evidence_versions (
                        version_id, evidence_id, content_hash, captured_at, metadata_snapshot
                    ) VALUES (?, ?, ?, ?, ?)
                """,
                    (uuid4().hex, evidence_id, content_hash, now.isoformat(), meta_str),
                )
                conn.commit()
                status = EvidenceStatus.UPDATED

            record = EvidenceRecord(
                evidence_id=evidence_id,
                canonical_url=c_url,
                source_type=row["source_type"],
                title=title if status == EvidenceStatus.UPDATED else (row["title"] or ""),
                first_seen_at=first_seen_at,
                last_seen_at=now,
                content_hash=content_hash,
                published_at=_parse_datetime(row["published_at"]),
                updated_at=now
                if status == EvidenceStatus.UPDATED
                else _parse_datetime(row["updated_at"]),
                source_authority=SourceAuthority(row["source_authority"])
                if row["source_authority"]
                else SourceAuthority.UNKNOWN,
                authority_score=row["authority_score"]
                if row["authority_score"] is not None
                else 0.5,
                metadata=json.loads(meta_str)
                if status == EvidenceStatus.UPDATED
                else json.loads(row["metadata"] or "{}"),
            )
            return record, status

    def get_by_url(self, canonical_url: str) -> EvidenceRecord | None:
        """Look up evidence by canonical URL."""
        with contextlib.closing(self._connect()) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evidence WHERE canonical_url = ?", (canonical_url,))
            row = cursor.fetchone()
            if not row:
                return None

            return EvidenceRecord(
                evidence_id=row["evidence_id"],
                canonical_url=row["canonical_url"],
                source_type=row["source_type"],
                title=row["title"] or "",
                first_seen_at=_parse_datetime(row["first_seen_at"]) or datetime.now(UTC),
                last_seen_at=_parse_datetime(row["last_seen_at"]) or datetime.now(UTC),
                content_hash=row["content_hash"] or "",
                published_at=_parse_datetime(row["published_at"]),
                updated_at=_parse_datetime(row["updated_at"]),
                source_authority=SourceAuthority(row["source_authority"])
                if row["source_authority"]
                else SourceAuthority.UNKNOWN,
                authority_score=row["authority_score"]
                if row["authority_score"] is not None
                else 0.5,
                metadata=json.loads(row["metadata"] or "{}"),
            )

    def get_versions(self, evidence_id: str) -> list[EvidenceVersion]:
        """Get version history for evidence."""
        with contextlib.closing(self._connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM evidence_versions WHERE evidence_id = ? ORDER BY captured_at ASC",
                (evidence_id,),
            )
            return [
                EvidenceVersion(
                    version_id=r["version_id"],
                    evidence_id=r["evidence_id"],
                    content_hash=r["content_hash"],
                    captured_at=_parse_datetime(r["captured_at"]) or datetime.now(UTC),
                    metadata_snapshot=json.loads(r["metadata_snapshot"])
                    if r["metadata_snapshot"]
                    else {},
                )
                for r in cursor.fetchall()
            ]

    def get_undelivered(self, run_id: str | None = None) -> list[EvidenceRecord]:
        """Get evidence records not yet delivered."""
        with contextlib.closing(self._connect()) as conn:
            cursor = conn.cursor()
            query = (
                "SELECT e.* FROM evidence e "
                "LEFT JOIN delivery_history d "
                "ON e.evidence_id = d.evidence_id AND d.delivery_status = 'sent' "
                "WHERE d.evidence_id IS NULL"
            )
            cursor.execute(query)
            return [
                EvidenceRecord(
                    evidence_id=row["evidence_id"],
                    canonical_url=row["canonical_url"],
                    source_type=row["source_type"],
                    title=row["title"] or "",
                    first_seen_at=_parse_datetime(row["first_seen_at"]) or datetime.now(UTC),
                    last_seen_at=_parse_datetime(row["last_seen_at"]) or datetime.now(UTC),
                    content_hash=row["content_hash"] or "",
                    published_at=_parse_datetime(row["published_at"]),
                    updated_at=_parse_datetime(row["updated_at"]),
                    source_authority=SourceAuthority(row["source_authority"])
                    if row["source_authority"]
                    else SourceAuthority.UNKNOWN,
                    authority_score=row["authority_score"]
                    if row["authority_score"] is not None
                    else 0.5,
                    metadata=json.loads(row["metadata"] or "{}"),
                )
                for row in cursor.fetchall()
            ]

    def record_delivery(
        self,
        *,
        run_id: str,
        evidence_id: str,
        evidence_version: str = "",
        delivery_status: str = "sent",
    ) -> DeliveryRecord:
        """Record that evidence was delivered."""
        now = datetime.now(UTC)
        delivery_id = uuid4().hex

        with contextlib.closing(self._connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO delivery_history (
                    delivery_id, run_id, evidence_id,
                    evidence_version, delivered_at, delivery_status
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    delivery_id,
                    run_id,
                    evidence_id,
                    evidence_version,
                    now.isoformat(),
                    delivery_status,
                ),
            )
            conn.commit()

        return DeliveryRecord(
            delivery_id=delivery_id,
            run_id=run_id,
            evidence_id=evidence_id,
            evidence_version=evidence_version,
            delivered_at=now,
            delivery_status=delivery_status,
        )

    def has_been_delivered(self, evidence_id: str, *, evidence_version: str | None = None) -> bool:
        """Check if evidence has ever been successfully delivered."""
        with contextlib.closing(self._connect()) as conn:
            cursor = conn.cursor()
            query = (
                "SELECT 1 FROM delivery_history WHERE evidence_id = ? AND delivery_status = 'sent'"
            )
            params = [evidence_id]
            if evidence_version is not None:
                query += " AND evidence_version = ?"
                params.append(evidence_version)
            cursor.execute(query + " LIMIT 1", params)
            return cursor.fetchone() is not None

    def start_run(self, run: DigestRunRecord) -> None:
        """Persist the start of a digest run."""
        with contextlib.closing(self._connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO digest_runs_v2 (
                    run_id, started_at, mode, force, dry_run,
                    sources_attempted, sources_succeeded, sources_failed,
                    items_collected, items_new, items_updated, items_duplicate,
                    items_ranked, items_published, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run.run_id,
                    run.started_at.isoformat(),
                    run.mode,
                    run.force,
                    run.dry_run,
                    run.sources_attempted,
                    run.sources_succeeded,
                    run.sources_failed,
                    run.items_collected,
                    run.items_new,
                    run.items_updated,
                    run.items_duplicate,
                    run.items_ranked,
                    run.items_published,
                    run.status,
                ),
            )
            conn.commit()

    def finish_run(
        self, run_id: str, *, status: str, completed_at: datetime | None = None, **counters: int
    ) -> None:
        """Update run with completion status and counters."""
        now_str = (completed_at or datetime.now(UTC)).isoformat()
        with contextlib.closing(self._connect()) as conn:
            cursor = conn.cursor()

            updates = ["completed_at = ?", "status = ?"]
            params: list[object] = [now_str, status]

            for k, v in counters.items():
                if k in [
                    "sources_attempted",
                    "sources_succeeded",
                    "sources_failed",
                    "items_collected",
                    "items_new",
                    "items_updated",
                    "items_duplicate",
                    "items_ranked",
                    "items_published",
                ]:
                    updates.append(f"{k} = ?")
                    params.append(v)

            params.append(run_id)
            query = f"UPDATE digest_runs_v2 SET {', '.join(updates)} WHERE run_id = ?"
            cursor.execute(query, params)
            conn.commit()

    def acquire_lock(self, run_id: str, ttl_seconds: float = 600) -> bool:
        """Try to acquire the digest execution lock. Returns True if acquired."""
        now = datetime.now(UTC)
        expires_at = datetime.fromtimestamp(now.timestamp() + ttl_seconds, tz=UTC)

        flock_acquired = False
        if fcntl is not None and self._lock_fd is None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(self._lock_file, os.O_CREAT | os.O_RDWR, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._lock_fd = fd
                flock_acquired = True
            except (BlockingIOError, OSError):
                return False

        with contextlib.closing(self._connect()) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT locked_by, expires_at FROM digest_lock WHERE id = 1")
            row = cursor.fetchone()

            if row:
                current_owner = row["locked_by"]
                current_expires = _parse_datetime(row["expires_at"]) or now
                if current_owner != run_id and now <= current_expires and not flock_acquired:
                    # Still locked by another active process
                    return False

            cursor.execute(
                """
                INSERT INTO digest_lock (id, locked_by, locked_at, expires_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    locked_by = excluded.locked_by,
                    locked_at = excluded.locked_at,
                    expires_at = excluded.expires_at
            """,
                (run_id, now.isoformat(), expires_at.isoformat()),
            )
            conn.commit()

        return True

    def release_lock(self, run_id: str) -> None:
        """Release the digest execution lock."""
        with contextlib.closing(self._connect()) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM digest_lock WHERE id = 1 AND locked_by = ?", (run_id,))
            conn.commit()

        if fcntl is not None and self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None

    def filter_unseen_or_updated(
        self, evidence_items: list[tuple[str, str, str]]
    ) -> dict[str, EvidenceStatus]:
        """For each (url, source_type, content_hash), return status.

        Returns dict mapping canonical_url -> EvidenceStatus.
        Items not in registry -> NEW
        Items with same hash -> DUPLICATE
        Items with different hash -> UPDATED
        """
        results = {}
        with contextlib.closing(self._connect()) as conn:
            cursor = conn.cursor()
            for url, _source_type, content_hash in evidence_items:
                c_url = canonicalize_url(url)
                cursor.execute(
                    "SELECT content_hash FROM evidence WHERE canonical_url = ?", (c_url,)
                )
                row = cursor.fetchone()
                if not row:
                    results[c_url] = EvidenceStatus.NEW
                elif not row["content_hash"] or row["content_hash"] == content_hash:
                    results[c_url] = EvidenceStatus.DUPLICATE
                else:
                    results[c_url] = EvidenceStatus.UPDATED

        return results
