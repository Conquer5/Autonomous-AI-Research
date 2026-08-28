"""In-process Phase 1/2 execution traces and aggregate status."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class TraceStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class ExecutionTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str
    run_id: str | None = None
    user_id: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    status: TraceStatus = TraceStatus.RUNNING
    runtime_calls: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    tool_calls: dict[str, int] = Field(default_factory=dict)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)


class TraceSummary(BaseModel):
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    running_tasks: int
    runtime_calls: int
    llm_calls: int
    tool_calls: int
    retries: int
    average_duration_ms: float


class TraceStore:
    """Concurrency-safe bounded trace store for early phases."""

    def __init__(self, max_traces: int = 500) -> None:
        self._traces: dict[str, ExecutionTrace] = {}
        self._lock = asyncio.Lock()
        self._max_traces = max_traces

    async def start(
        self, task_id: str, user_id: str | None = None, run_id: str | None = None
    ) -> ExecutionTrace:
        trace = ExecutionTrace(task_id=task_id, user_id=user_id, run_id=run_id)
        async with self._lock:
            if len(self._traces) >= self._max_traces:
                oldest = min(self._traces.values(), key=lambda item: item.started_at)
                del self._traces[oldest.trace_id]
            self._traces[trace.trace_id] = trace
        return trace.model_copy(deep=True)

    async def record_runtime_call(
        self, trace_id: str, *, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        async with self._lock:
            trace = self._traces[trace_id]
            trace.runtime_calls += 1
            trace.input_tokens += input_tokens
            trace.output_tokens += output_tokens

    async def record_llm_call(
        self, trace_id: str, *, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        async with self._lock:
            trace = self._traces[trace_id]
            trace.llm_calls += 1
            trace.input_tokens += input_tokens
            trace.output_tokens += output_tokens

    async def record_tool_call(self, trace_id: str, tool_name: str) -> None:
        async with self._lock:
            trace = self._traces[trace_id]
            trace.tool_calls[tool_name] = trace.tool_calls.get(tool_name, 0) + 1

    async def record_retry(self, trace_id: str) -> None:
        async with self._lock:
            self._traces[trace_id].retries += 1

    async def record_error(self, trace_id: str, error: str) -> None:
        async with self._lock:
            self._traces[trace_id].errors.append(error)

    async def finish(self, trace_id: str, status: TraceStatus) -> ExecutionTrace:
        now = datetime.now(UTC)
        async with self._lock:
            trace = self._traces[trace_id]
            trace.completed_at = now
            trace.duration_ms = max(0.0, (now - trace.started_at).total_seconds() * 1000)
            trace.status = status
            return trace.model_copy(deep=True)

    async def get(self, trace_id: str) -> ExecutionTrace | None:
        async with self._lock:
            trace = self._traces.get(trace_id)
            return trace.model_copy(deep=True) if trace else None

    async def summary(self) -> TraceSummary:
        async with self._lock:
            traces = list(self._traces.values())
        durations = [trace.duration_ms for trace in traces if trace.duration_ms is not None]
        return TraceSummary(
            total_tasks=len(traces),
            successful_tasks=sum(trace.status == TraceStatus.SUCCESS for trace in traces),
            failed_tasks=sum(trace.status == TraceStatus.FAILED for trace in traces),
            running_tasks=sum(trace.status == TraceStatus.RUNNING for trace in traces),
            runtime_calls=sum(trace.runtime_calls for trace in traces),
            llm_calls=sum(trace.llm_calls for trace in traces),
            tool_calls=sum(sum(trace.tool_calls.values()) for trace in traces),
            retries=sum(trace.retries for trace in traces),
            average_duration_ms=sum(durations) / len(durations) if durations else 0.0,
        )
