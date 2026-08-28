from __future__ import annotations

import json
import logging

from research_radar.observability.logging import JsonFormatter, redact_text
from research_radar.observability.tracing import TraceStatus, TraceStore


def test_log_redaction_removes_common_secret_shapes() -> None:
    assert "abc123" not in redact_text("Authorization: Bearer abc123")
    assert "secret-value" not in redact_text("api_key=secret-value")
    telegram_log = (
        "POST https://api.telegram.org/bot123456789:AATelegramSecret/getUpdates HTTP/1.1 200 OK"
    )
    redacted_telegram_log = redact_text(telegram_log)
    assert "AATelegramSecret" not in redacted_telegram_log
    assert "https://api.telegram.org/bot[REDACTED]/getUpdates" in redacted_telegram_log

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request complete",
        args=(),
        exc_info=None,
    )
    record.api_token = "secret"
    record.event = telegram_log
    payload = json.loads(JsonFormatter().format(record))
    assert payload["api_token"] == "[REDACTED]"
    assert "AATelegramSecret" not in payload["event"]


async def test_trace_store_aggregates_task_metrics() -> None:
    store = TraceStore()
    trace = await store.start("task-1", "user-1")
    await store.record_runtime_call(trace.trace_id, input_tokens=10, output_tokens=5)
    await store.record_tool_call(trace.trace_id, "github")
    await store.finish(trace.trace_id, TraceStatus.SUCCESS)

    summary = await store.summary()
    assert summary.total_tasks == 1
    assert summary.successful_tasks == 1
    assert summary.runtime_calls == 1
    assert summary.tool_calls == 1
