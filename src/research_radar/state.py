"""Tiny SQLite state store used to prevent duplicate autonomous digests."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path


class DigestStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_evidence (
                url TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                first_sent_at REAL NOT NULL,
                last_sent_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS digest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sent_at REAL NOT NULL,
                item_count INTEGER NOT NULL
            )
            """
        )
        return connection

    def _last_sent_at(self) -> datetime | None:
        with closing(self._connect()) as connection:
            with connection:
                row = connection.execute("SELECT MAX(sent_at) FROM digest_runs").fetchone()
        timestamp = row[0] if row else None
        return datetime.fromtimestamp(timestamp, tz=UTC) if timestamp is not None else None

    async def last_sent_at(self) -> datetime | None:
        # Queries are tiny and bounded; keeping them in-process also works in
        # constrained environments where thread creation is unavailable.
        return self._last_sent_at()

    def _filter_unseen(self, urls: list[str]) -> set[str]:
        if not urls:
            return set()
        with closing(self._connect()) as connection:
            with connection:
                rows = connection.execute("SELECT url FROM seen_evidence").fetchall()
        seen = {row[0] for row in rows}
        return {url for url in urls if url not in seen}

    async def filter_unseen(self, urls: list[str]) -> set[str]:
        return self._filter_unseen(urls)

    def _record_sent(self, evidence: list[tuple[str, str]], sent_at: datetime) -> None:
        timestamp = sent_at.astimezone(UTC).timestamp()
        with closing(self._connect()) as connection:
            with connection:
                connection.executemany(
                    """
                    INSERT INTO seen_evidence(url, source, first_sent_at, last_sent_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET last_sent_at=excluded.last_sent_at
                    """,
                    [(url, source, timestamp, timestamp) for url, source in evidence],
                )
                connection.execute(
                    "INSERT INTO digest_runs(sent_at, item_count) VALUES (?, ?)",
                    (timestamp, len(evidence)),
                )
                connection.execute(
                    "DELETE FROM seen_evidence WHERE last_sent_at < ?",
                    (timestamp - 365 * 86400,),
                )

    async def record_sent(self, evidence: list[tuple[str, str]], sent_at: datetime) -> None:
        self._record_sent(evidence, sent_at)
