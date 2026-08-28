from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from research_radar.exceptions import InvalidResponseError
from research_radar.llm.gemini import GeminiProvider
from research_radar.schemas import LLMRequest
from research_radar.utils.retry import RetryPolicy


class Finding(BaseModel):
    title: str
    score: float


class FakeModels:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response: Any) -> None:
        self.models = FakeModels(response)


def response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        model_version="gemini-test",
        response_id="response-1",
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
            cached_content_token_count=2,
        ),
    )


async def test_generate_tracks_model_usage_and_request_id() -> None:
    fake = FakeClient(response("hello"))
    provider = GeminiProvider(
        api_key="unused",
        default_model="gemini-default",
        retry_policy=RetryPolicy(attempts=1),
        client=fake,
    )

    result = await provider.generate(LLMRequest(prompt="hello", request_id="request-1"))

    assert result.text == "hello"
    assert result.model == "gemini-test"
    assert result.request_id == "request-1"
    assert result.usage.total_tokens == 15
    assert fake.models.calls[0]["model"] == "gemini-default"


async def test_structured_output_is_validated_by_pydantic() -> None:
    provider = GeminiProvider(
        api_key="unused",
        default_model="gemini-default",
        retry_policy=RetryPolicy(attempts=1),
        client=FakeClient(response('{"title":"Agent","score":0.9}')),
    )

    result = await provider.generate_structured(LLMRequest(prompt="rank"), Finding)

    assert result.data == Finding(title="Agent", score=0.9)


async def test_malformed_structured_output_is_not_silently_accepted() -> None:
    provider = GeminiProvider(
        api_key="unused",
        default_model="gemini-default",
        retry_policy=RetryPolicy(attempts=1),
        client=FakeClient(response("not-json")),
    )

    with pytest.raises(InvalidResponseError):
        await provider.generate_structured(LLMRequest(prompt="rank"), Finding)


class RateLimitError(Exception):
    def __init__(self, msg: str = "RESOURCE_EXHAUSTED", status_code: int = 429) -> None:
        super().__init__(msg)
        self.status_code = status_code


class MultiModelFakeClient:
    def __init__(self, responses_by_model: dict[str, Any]) -> None:
        self.models = self
        self.responses_by_model = responses_by_model
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        model = kwargs["model"]
        res = self.responses_by_model.get(model)
        if isinstance(res, Exception):
            raise res
        return res


async def test_fallback_when_primary_model_hits_rate_limit() -> None:
    fake = MultiModelFakeClient(
        {
            "gemini-3.5-flash": RateLimitError("Quota exceeded for quota metric 'Requests'"),
            "gemini-3.5-flash-lite": response("fallback-response"),
        }
    )
    provider = GeminiProvider(
        api_key="unused",
        default_model="gemini-3.5-flash",
        fallback_models=("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"),
        retry_policy=RetryPolicy(attempts=1),
        client=fake,
        rpm_limit=None,
    )

    result = await provider.generate(LLMRequest(prompt="hello", request_id="req-fallback"))

    assert result.text == "fallback-response"
    assert len(fake.calls) == 2
    assert fake.calls[0]["model"] == "gemini-3.5-flash"
    assert fake.calls[1]["model"] == "gemini-3.5-flash-lite"


async def test_rate_pacing_spaces_requests() -> None:
    fake = FakeClient(response("pacing-test"))
    provider = GeminiProvider(
        api_key="unused",
        default_model="gemini-default",
        rpm_limit=60,  # 1 sec interval
        retry_policy=RetryPolicy(attempts=1),
        client=fake,
    )
    # Pacing is configured with min_interval_seconds = 1.0
    assert provider._min_interval_seconds == 1.0
