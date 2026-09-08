"""Safe retrieval diagnostics and bounded transport shared by source adapters."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import socket
import sqlite3
import time
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field, ValidationError

from research_radar.exceptions import ConfigurationError, ExternalServiceError, InvalidResponseError
from research_radar.observability.logging import current_run_id
from research_radar.utils.retry import RetryPolicy, call_with_retry

if TYPE_CHECKING:
    from research_radar.schemas import SearchBatch

BatchT = TypeVar("BatchT", bound="SearchBatch[Any]")
SearchParams = ParamSpec("SearchParams")

logger = logging.getLogger(__name__)


class RetrievalFailure(BaseModel):
    provider: str
    operation: str = "search"
    failure_category: str
    retryable: bool = False
    http_status_class: str | None = None
    error_code: str
    attempt: int = 0


class RetrievalOperation(BaseModel):
    provider: str
    operation: str = "search"
    call_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str | None = None
    iteration: int = 0
    query_hash: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    latency_ms: float = 0
    attempts: int = 0
    retries: int = 0
    result_count: int = 0
    partial_result: bool = False
    status: str = "success"
    failures: list[RetrievalFailure] = Field(default_factory=list)


current_call_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "retrieval_call_id", default=None
)

operation_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "retrieval_deadline", default=None
)

active_operation: contextvars.ContextVar[RetrievalOperation | None] = contextvars.ContextVar(
    "retrieval_operation", default=None
)
operation_sink: contextvars.ContextVar[list[RetrievalOperation] | None] = contextvars.ContextVar(
    "retrieval_sink", default=None
)
current_iteration: contextvars.ContextVar[int] = contextvars.ContextVar(
    "retrieval_iteration", default=0
)


class RetrievalError(ExternalServiceError):
    def __init__(self, failure: RetrievalFailure, status_code: int | None = None) -> None:
        super().__init__(
            failure.provider,
            f"{failure.provider}: {failure.failure_category}",
            status_code=status_code,
            retryable=failure.retryable,
        )
        self.failure = failure


def classify_failure(provider: str, exc: BaseException) -> RetrievalFailure:
    if isinstance(exc, RetrievalError):
        return exc.failure.model_copy()
    category, retryable, status = "unexpected_exception", False, None
    cause = exc
    while cause.__cause__ is not None:
        cause = cause.__cause__
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        headers = exc.response.headers
        # Inspect only a documented error code, never persist provider payloads.
        code = ""
        try:
            data = exc.response.json()
            if isinstance(data, dict) and isinstance(data.get("error"), dict):
                code = str(data["error"].get("code", ""))
        except (ValueError, TypeError):
            pass
        if code in {"QUOTA_LIMITED", "QUOTA_EXCEEDED"}:
            category = "quota_exhaustion"
        elif status == 401:
            category = "authentication_failure"
        elif status == 403 and (
            headers.get("x-ratelimit-remaining") == "0" or "retry-after" in headers
        ):
            category, retryable = "rate_limiting", True
        elif status == 403:
            category = "authorization_failure"
        elif status == 429:
            category, retryable = "http_429", True
        elif status in {400, 422}:
            category = "invalid_query"
        elif status >= 500:
            category, retryable = "http_5xx", True
        else:
            category = "http_4xx"
    elif isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        category, retryable = "timeout", True
    elif isinstance(cause, socket.gaierror):
        category, retryable = "dns_failure", True
    elif isinstance(exc, httpx.NetworkError):
        category, retryable = "connection_failure", True
    elif isinstance(exc, httpx.RequestError):
        category, retryable = "network_failure", True
    elif isinstance(exc, sqlite3.Error):
        category = "storage_failure"
    elif isinstance(exc, ConfigurationError):
        category = "provider_unavailable"
    elif isinstance(exc, json.JSONDecodeError):
        category = "malformed_response"
    elif isinstance(exc, ET.ParseError):
        category = "parser_failure"
    elif isinstance(exc, (ValidationError, KeyError, TypeError, AttributeError)):
        category = "schema_drift"
    elif isinstance(exc, InvalidResponseError):
        return (
            classify_failure(provider, exc.__cause__)
            if exc.__cause__
            else RetrievalFailure(
                provider=provider,
                failure_category="malformed_response",
                error_code="InvalidResponseError",
            )
        )
    elif isinstance(exc, ExternalServiceError) and exc.__cause__:
        return classify_failure(provider, exc.__cause__)
    elif isinstance(exc, ValueError):
        category = "invalid_query"
    return RetrievalFailure(
        provider=provider,
        failure_category=category,
        retryable=retryable,
        http_status_class=f"{status // 100}xx" if status else None,
        error_code=f"HTTP_{status}" if status else type(exc).__name__,
    )


def retry_after(exc: Exception) -> float:
    if not isinstance(exc, httpx.HTTPStatusError):
        return 0
    headers = exc.response.headers
    delays = [0.0]
    value = headers.get("retry-after")
    if value:
        try:
            delays.append(float(value))
        except ValueError:
            try:
                delays.append(parsedate_to_datetime(value).timestamp() - time.time())
            except (ValueError, TypeError, OverflowError):
                pass
    if headers.get("x-ratelimit-remaining") == "0":
        try:
            delays.append(float(headers.get("x-ratelimit-reset", "0")) - time.time())
        except ValueError:
            pass
    return max(delays)


async def request_with_retry(
    provider: str,
    operation: Callable[[], Awaitable[httpx.Response]],
    *,
    policy: RetryPolicy,
    timeout_seconds: float = 30,
    resource_id: str | None = None,
) -> httpx.Response:
    attempt = 0
    trace = active_operation.get()
    request_id = uuid4().hex

    async def request() -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if trace is not None:
            trace.attempts += 1
        started = time.perf_counter()
        failure = None
        status = None
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await operation()
            status = f"{response.status_code // 100}xx"
            return response
        except asyncio.CancelledError:
            failure = RetrievalFailure(
                provider=provider,
                failure_category="cancelled",
                error_code="CANCELLED",
                attempt=attempt,
            )
            raise
        except Exception as exc:
            failure = classify_failure(provider, exc)
            failure.attempt = attempt
            raise
        finally:
            logger.info(
                "Provider request finished",
                extra={
                    "event": "provider_request",
                    "provider": provider,
                    "request_id": request_id,
                    "resource_id": resource_id,
                    "started_at": datetime.fromtimestamp(
                        time.time() - (time.perf_counter() - started), UTC
                    ).isoformat(),
                    "run_id": current_run_id.get(),
                    "call_id": trace.call_id if trace else None,
                    "iteration": current_iteration.get(),
                    "attempt": attempt,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "failure_category": failure.failure_category if failure else None,
                    "retryable": failure.retryable if failure else False,
                    "http_status_class": failure.http_status_class if failure else status,
                },
            )

    def on_retry(exc: Exception, attempt: int) -> None:
        if trace is not None:
            trace.retries += 1

    try:
        return await call_with_retry(
            request,
            policy=policy,
            should_retry=lambda exc: (
                classify_failure(provider, exc).retryable
                and retry_after(exc) <= policy.max_delay_seconds
            ),
            on_retry=on_retry,
            minimum_delay=retry_after,
        )
    except Exception as exc:
        failure = classify_failure(provider, exc)
        failure.attempt = attempt
        raise RetrievalError(
            failure, getattr(getattr(exc, "response", None), "status_code", None)
        ) from exc


def observed_search(
    provider: str,
) -> Callable[
    [Callable[SearchParams, Awaitable[BatchT]]], Callable[SearchParams, Awaitable[BatchT]]
]:
    """Observe normalization as well as transport, without storing queries or payloads."""

    def decorate(
        function: Callable[SearchParams, Awaitable[BatchT]],
    ) -> Callable[SearchParams, Awaitable[BatchT]]:
        @wraps(function)
        async def wrapped(*args: SearchParams.args, **kwargs: SearchParams.kwargs) -> BatchT:
            query = args[1] if len(args) > 1 else kwargs.get("query", kwargs.get("topics", ""))
            trace = RetrievalOperation(
                provider=provider,
                call_id=current_call_id.get() or uuid4().hex,
                run_id=current_run_id.get(),
                iteration=current_iteration.get(),
                query_hash=hashlib.sha256(str(query).encode()).hexdigest()[:16],
            )
            token = active_operation.set(trace)
            start = time.perf_counter()
            try:
                batch = await function(*args, **kwargs)
                trace.result_count = len(batch.items)
                trace.partial_result = batch.partial
                trace.failures.extend(batch.failures)
                trace.status = "partial" if batch.partial else "success" if batch.items else "empty"
                if not batch.items and not trace.failures:
                    trace.failures.append(
                        RetrievalFailure(
                            provider=provider,
                            failure_category="empty_result",
                            error_code="EMPTY_RESULT",
                        )
                    )
                return batch
            except asyncio.CancelledError:
                trace.status = "cancelled"
                raise
            except Exception as exc:
                trace.status = "failed"
                failure = classify_failure(provider, exc)
                trace.failures.append(failure)
                if type(exc) is ValueError:
                    raise
                raise RetrievalError(failure, getattr(exc, "status_code", None)) from exc
            finally:
                trace.latency_ms = round((time.perf_counter() - start) * 1000, 2)
                sink = operation_sink.get()
                if sink is not None:
                    sink.append(trace)
                logger.info(
                    "Retrieval finished",
                    extra={"event": "retrieval_operation", **trace.model_dump(mode="json")},
                )
                active_operation.reset(token)

        return wrapped

    return decorate
