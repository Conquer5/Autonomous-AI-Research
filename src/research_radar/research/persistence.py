"""SQLite persistence for autonomous research runs and query records."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from research_radar.research.models import (
    ResearchSynthesisResult,
)
from research_radar.utils.deadline import bounded_timeout


class ResearchStore:
    """Stores research executions and query provenance."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=bounded_timeout(30.0))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    research_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stop_reason TEXT,
                    iterations INTEGER NOT NULL,
                    tool_calls INTEGER NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    confidence TEXT NOT NULL,
                    consensus_level TEXT,
                    contradictions_count INTEGER DEFAULT 0,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    summary TEXT
                )
                """
            )

            # Idempotent column migrations for existing databases
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(research_runs)")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            if "consensus_level" not in existing_cols:
                conn.execute("ALTER TABLE research_runs ADD COLUMN consensus_level TEXT")
            if "contradictions_count" not in existing_cols:
                conn.execute(
                    "ALTER TABLE research_runs ADD COLUMN contradictions_count INTEGER DEFAULT 0"
                )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_queries (
                    id TEXT PRIMARY KEY,
                    research_id TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    query TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    result_count INTEGER NOT NULL,
                    new_evidence_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    executed_at TEXT NOT NULL,
                    FOREIGN KEY (research_id) REFERENCES research_runs(research_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_queries_research "
                "ON research_queries(research_id)"
            )
            for table in ("research_runs", "research_queries"):
                columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                if "diagnostics" not in columns:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN diagnostics TEXT")
            conn.commit()

    def save_run(self, result: ResearchSynthesisResult) -> None:
        """Persist a completed or partial research run and its query history."""
        now_str = datetime.now(UTC).isoformat()
        state = result.state
        completed_str = state.completed_at.isoformat() if state.completed_at else now_str
        contra_count = len(result.consensus_report.contradictions) if result.consensus_report else 0

        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO research_runs (
                    research_id, run_id, question, mode, status, stop_reason,
                    iterations, tool_calls, evidence_count, confidence,
                    consensus_level, contradictions_count,
                    started_at, completed_at, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(research_id) DO UPDATE SET
                    status = excluded.status,
                    stop_reason = excluded.stop_reason,
                    iterations = excluded.iterations,
                    tool_calls = excluded.tool_calls,
                    evidence_count = excluded.evidence_count,
                    confidence = excluded.confidence,
                    consensus_level = excluded.consensus_level,
                    contradictions_count = excluded.contradictions_count,
                    completed_at = excluded.completed_at,
                    summary = excluded.summary
                """,
                (
                    state.research_id,
                    state.run_id,
                    state.question,
                    state.mode.value,
                    state.status.value,
                    state.stop_reason.value if state.stop_reason else None,
                    state.iterations,
                    state.tool_calls,
                    len(state.evidence_ids),
                    result.confidence.value,
                    result.consensus.value,
                    contra_count,
                    state.started_at.isoformat(),
                    completed_str,
                    result.answer[:500],
                ),
            )

            # Replace executed queries idempotently
            conn.execute("DELETE FROM research_queries WHERE research_id = ?", (state.research_id,))
            for q in state.queries_executed:
                conn.execute(
                    """
                    INSERT INTO research_queries (
                        id, research_id, tool, query, iteration,
                        result_count, new_evidence_count, status, executed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid4().hex,
                        state.research_id,
                        q.tool,
                        q.query,
                        q.iteration,
                        q.result_count,
                        q.new_evidence_count,
                        q.status,
                        now_str,
                    ),
                )

            conn.execute(
                "UPDATE research_runs SET diagnostics = ? WHERE research_id = ?",
                (
                    json.dumps(
                        {
                            "coverage": state.coverage,
                            "uncertainties": state.uncertainties,
                            "provider_failures": [f.model_dump() for f in state.provider_failures],
                        }
                    ),
                    state.research_id,
                ),
            )
            for q in state.queries_executed:
                conn.execute(
                    "UPDATE research_queries SET diagnostics = ? WHERE research_id = ? "
                    "AND tool = ? AND query = ? AND iteration = ?",
                    (
                        q.model_dump_json(exclude={"query"}),
                        state.research_id,
                        q.tool,
                        q.query,
                        q.iteration,
                    ),
                )
            conn.commit()

    def get_run(self, research_id: str) -> dict[str, object] | None:
        """Fetch a research run summary by research_id."""
        with contextlib.closing(self._connect()) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM research_runs WHERE research_id = ?", (research_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)

    def get_queries(self, research_id: str) -> list[dict[str, object]]:
        """Fetch all executed queries for a research run."""
        with contextlib.closing(self._connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM research_queries WHERE research_id = ? "
                "ORDER BY iteration ASC, executed_at ASC",
                (research_id,),
            )
            return [dict(row) for row in cursor.fetchall()]
