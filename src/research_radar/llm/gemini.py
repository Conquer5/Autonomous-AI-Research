"""Google Gemini implementation using the supported Google Gen AI SDK."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Any, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from research_radar.exceptions import ExternalServiceError, InvalidResponseError
from research_radar.llm.base import LLMProvider
from research_radar.schemas import (
    LLMRequest,
    LLMResponse,
    StructuredLLMResponse,
    TokenUsage,
)
from research_radar.utils.retry import RetryPolicy, call_with_retry

logger = logging.getLogger(__name__)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    code = getattr(exc, "status_code", getattr(exc, "code", None))
    return code == 429 or (isinstance(code, int) and code >= 500)


def _is_rate_limit_or_quota(exc: Exception) -> bool:
    code = getattr(exc, "status_code", getattr(exc, "code", None))
    if code == 429:
        return True
    msg = str(exc).lower()
    return "resource_exhausted" in msg or "429" in msg or "quota" in msg or "rate limit" in msg


class GeminiProvider(LLMProvider):
    """Single gateway for all direct application-owned Gemini calls."""

    def __init__(
        self,
        *,
        api_key: str,
        default_model: str,
        fallback_models: tuple[str, ...] = (),
        rpm_limit: int | None = 12,
        timeout_seconds: float = 60.0,
        retry_policy: RetryPolicy | None = None,
        client: Any | None = None,
    ) -> None:
        self.default_model = default_model
        self.fallback_models = fallback_models
        self.rpm_limit = rpm_limit
        self._min_interval_seconds = 60.0 / rpm_limit if rpm_limit and rpm_limit > 0 else 0.0
        self._last_request_time = 0.0
        self._rate_lock = asyncio.Lock()
        self.timeout_seconds = timeout_seconds
        self.retry_policy = retry_policy or RetryPolicy()
        self._owns_client = client is None
        self._client = client if client is not None else genai.Client(api_key=api_key).aio

    async def _pace_rate(self) -> None:
        if self._min_interval_seconds <= 0:
            return
        async with self._rate_lock:
            now = asyncio.get_running_loop().time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval_seconds:
                await asyncio.sleep(self._min_interval_seconds - elapsed)
            self._last_request_time = asyncio.get_running_loop().time()

    def _config(
        self,
        request: LLMRequest,
        model: str,
        response_schema: type[BaseModel] | None = None,
    ) -> types.GenerateContentConfig:
        # Gemini 3.5+ manages sampling alongside thinking level; Google recommends
        # removing legacy sampling parameters during migration.
        temperature = (
            None
            if model.startswith(("gemini-3.5-", "gemini-3.6-", "gemini-3.7-"))
            else request.temperature
        )
        return types.GenerateContentConfig(
            system_instruction=request.system_instruction,
            temperature=temperature,
            max_output_tokens=request.max_output_tokens,
            response_mime_type="application/json" if response_schema else None,
            response_schema=response_schema,
        )

    async def _execute(
        self, request: LLMRequest, response_schema: type[BaseModel] | None = None
    ) -> tuple[Any, float, str]:
        primary_model = request.model or self.default_model
        candidates: list[str] = [primary_model]
        for fallback in self.fallback_models:
            if fallback and fallback not in candidates:
                candidates.append(fallback)

        last_exc: Exception | None = None
        for index, model in enumerate(candidates):
            started = perf_counter()
            target_model = model

            async def operation(m: str = target_model) -> Any:
                await self._pace_rate()
                async with asyncio.timeout(self.timeout_seconds):
                    return await self._client.models.generate_content(
                        model=m,
                        contents=request.prompt,
                        config=self._config(request, m, response_schema),
                    )

            def _log_retry(exc: Exception, attempt: int, m: str = target_model) -> None:
                logger.warning(
                    "Gemini call retry",
                    extra={
                        "event": "llm_retry",
                        "provider": "gemini",
                        "model": m,
                        "request_id": request.request_id,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                    },
                )

            try:
                response = await call_with_retry(
                    operation,
                    policy=self.retry_policy,
                    should_retry=_is_retryable,
                    on_retry=_log_retry,
                )
                return response, (perf_counter() - started) * 1000, target_model
            except Exception as exc:
                last_exc = exc
                is_quota_or_rate_limit = _is_rate_limit_or_quota(exc)
                has_next_fallback = index + 1 < len(candidates)

                if is_quota_or_rate_limit and has_next_fallback:
                    next_model = candidates[index + 1]
                    logger.warning(
                        "Gemini model %s rate limited / quota exhausted. Falling back to %s",
                        model,
                        next_model,
                        extra={
                            "event": "llm_model_fallback",
                            "provider": "gemini",
                            "failed_model": model,
                            "fallback_model": next_model,
                            "request_id": request.request_id,
                            "error": str(exc),
                        },
                    )
                    continue

                raise ExternalServiceError(
                    "gemini",
                    "Gemini sedang tidak dapat merespons. Silakan coba lagi.",
                    retryable=_is_retryable(exc),
                    detail=f"{type(exc).__name__} ({model}): {exc}",
                ) from exc

        if last_exc is not None:
            raise ExternalServiceError(
                "gemini",
                "Gemini sedang tidak dapat merespons. Silakan coba lagi.",
                retryable=_is_retryable(last_exc),
                detail=f"{type(last_exc).__name__}: {last_exc}",
            ) from last_exc
        raise ExternalServiceError("gemini", "Gemini provider tidak memiliki model yang tersedia.")

    @staticmethod
    def _usage(response: Any) -> TokenUsage:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return TokenUsage()
        return TokenUsage(
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            total_tokens=getattr(usage, "total_token_count", 0) or 0,
            cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        response, latency_ms, requested_model = await self._execute(request)
        text = getattr(response, "text", None)
        if text is None:
            raise InvalidResponseError(
                "gemini",
                "Gemini mengembalikan respons kosong.",
                detail="GenerateContentResponse.text was None",
            )
        return LLMResponse(
            text=text,
            model=getattr(response, "model_version", None) or requested_model,
            usage=self._usage(response),
            latency_ms=latency_ms,
            request_id=request.request_id,
            provider_response_id=getattr(response, "response_id", None),
        )

    async def generate_structured(
        self, request: LLMRequest, schema: type[SchemaT]
    ) -> StructuredLLMResponse[SchemaT]:
        response, latency_ms, requested_model = await self._execute(request, schema)
        try:
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, schema):
                data = parsed
            elif parsed is not None:
                data = schema.model_validate(parsed)
            else:
                text = getattr(response, "text", None)
                if not text:
                    raise ValueError("structured response had no text or parsed value")
                data = schema.model_validate_json(text)
        except (ValidationError, ValueError, TypeError) as exc:
            raise InvalidResponseError(
                "gemini",
                "Gemini mengembalikan data yang tidak valid.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        return StructuredLLMResponse[SchemaT](
            data=data,
            model=getattr(response, "model_version", None) or requested_model,
            usage=self._usage(response),
            latency_ms=latency_ms,
            request_id=request.request_id,
            provider_response_id=getattr(response, "response_id", None),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
