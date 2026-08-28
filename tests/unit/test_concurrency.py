"""Tests for Digest Concurrency Lock Protection."""

from __future__ import annotations

import time
from pathlib import Path

from research_radar.evidence.registry import EvidenceRegistry


def test_single_lock_acquisition_and_release(tmp_path: Path) -> None:
    db_path = tmp_path / "lock_test.sqlite3"
    registry = EvidenceRegistry(db_path)

    run_1 = "run-001"
    # Acquire lock
    assert registry.acquire_lock(run_1, ttl_seconds=60) is True

    # Same owner can re-acquire / heartbeat
    assert registry.acquire_lock(run_1, ttl_seconds=60) is True

    # Release lock
    registry.release_lock(run_1)

    # Another run can now acquire
    run_2 = "run-002"
    assert registry.acquire_lock(run_2, ttl_seconds=60) is True
    registry.release_lock(run_2)


def test_second_concurrent_run_is_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "lock_conflict.sqlite3"
    registry = EvidenceRegistry(db_path)

    run_1 = "run-001"
    run_2 = "run-002"

    # Run 1 holds the lock
    assert registry.acquire_lock(run_1, ttl_seconds=300) is True

    # Run 2 attempts to acquire and is rejected
    assert registry.acquire_lock(run_2, ttl_seconds=300) is False

    # Once Run 1 finishes, Run 2 can proceed
    registry.release_lock(run_1)
    assert registry.acquire_lock(run_2, ttl_seconds=300) is True
    registry.release_lock(run_2)


def test_expired_lock_can_be_taken_by_new_run(tmp_path: Path) -> None:
    db_path = tmp_path / "lock_expiry.sqlite3"
    registry = EvidenceRegistry(db_path)

    run_1 = "crashed-run-001"
    run_2 = "new-run-002"

    # Run 1 acquires with very short TTL (e.g. 0.01s)
    assert registry.acquire_lock(run_1, ttl_seconds=0.05) is True

    # Wait for expiry
    time.sleep(0.1)

    # Run 2 should now be able to take over the expired lock
    assert registry.acquire_lock(run_2, ttl_seconds=60) is True
    registry.release_lock(run_2)
