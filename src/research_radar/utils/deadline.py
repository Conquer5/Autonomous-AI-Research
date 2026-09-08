"""Workflow deadline shared with synchronous SQLite busy waits."""

from contextvars import ContextVar
from time import perf_counter

workflow_deadline: ContextVar[float | None] = ContextVar("workflow_deadline", default=None)


def bounded_timeout(default: float) -> float:
    deadline = workflow_deadline.get()
    return default if deadline is None else max(0.0, min(default, deadline - perf_counter()))
