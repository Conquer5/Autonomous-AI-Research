"""Authenticated HTTP adapter for an isolated Hermes Agent runtime."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from uuid import uuid4

import httpx

from research_radar.agent.base import HermesRuntime
from research_radar.exceptions import ExternalServiceError, InvalidResponseError
from research_radar.schemas import RuntimeResponse, TokenUsage, ToolHealth
from research_radar.utils.retry import RetryPolicy, call_with_retry

logger = logging.getLogger(__name__)


def _retryable_http_error(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, TimeoutError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class HermesHttpRuntime(HermesRuntime):
    """Drive Hermes through its supported OpenAI-compatible API server."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str = "hermes-agent",
        timeout_seconds: float = 120.0,
        retry_policy: RetryPolicy | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.retry_policy = retry_policy or RetryPolicy()
        self._authorization = f"Bearer {api_key}"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def run(
        self,
        prompt: str,
        *,
        session_key: str,
        system_instruction: str | None = None,
        request_id: str | None = None,
    ) -> RuntimeResponse:
        resolved_request_id = request_id or uuid4().hex
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model_name, "messages": messages, "stream": False}

        async def operation() -> httpx.Response:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": self._authorization,
                        "X-Hermes-Session-Key": session_key,
                        "Idempotency-Key": resolved_request_id,
                    },
                )
            response.raise_for_status()
            return response

        started = perf_counter()
        try:
            response = await call_with_retry(
                operation,
                policy=self.retry_policy,
                should_retry=_retryable_http_error,
                on_retry=lambda exc, attempt: logger.warning(
                    "Hermes runtime retry",
                    extra={
                        "event": "runtime_retry",
                        "request_id": resolved_request_id,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                    },
                ),
            )
        except Exception as exc:
            status_code = (
                exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            )
            raise ExternalServiceError(
                "hermes",
                "Hermes Agent sedang tidak dapat dijangkau. Silakan coba lagi.",
                status_code=status_code,
                retryable=_retryable_http_error(exc),
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"]
            usage_data = body.get("usage") or {}
            usage = TokenUsage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                cached_tokens=usage_data.get("cached_tokens", 0),
            )
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise InvalidResponseError(
                "hermes",
                "Hermes Agent mengembalikan respons yang tidak valid.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        return RuntimeResponse(
            text=text,
            model=body.get("model", self.model_name),
            usage=usage,
            latency_ms=(perf_counter() - started) * 1000,
            request_id=resolved_request_id,
        )

    async def health(self) -> ToolHealth:
        root_url = self.base_url.removesuffix("/v1")
        try:
            response = await self._client.get(
                f"{root_url}/health", headers={"Authorization": self._authorization}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return ToolHealth(
                name="hermes",
                configured=True,
                healthy=False,
                detail=type(exc).__name__,
            )
        return ToolHealth(name="hermes", configured=True, healthy=True)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class FakeHermesRuntime(HermesRuntime):
    """Deterministic runtime used in unit tests and local UI development."""

    def __init__(self, response_text: str = "Fake Hermes response") -> None:
        self.response_text = response_text
        self.calls: list[tuple[str, str]] = []

    async def run(
        self,
        prompt: str,
        *,
        session_key: str,
        system_instruction: str | None = None,
        request_id: str | None = None,
    ) -> RuntimeResponse:
        del system_instruction
        resolved_request_id = request_id or uuid4().hex
        self.calls.append((prompt, session_key))
        return RuntimeResponse(
            text=self.response_text,
            latency_ms=0,
            request_id=resolved_request_id,
        )

    async def health(self) -> ToolHealth:
        return ToolHealth(name="hermes", configured=True, healthy=True)

    async def aclose(self) -> None:
        return None
