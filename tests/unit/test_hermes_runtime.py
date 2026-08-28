from __future__ import annotations

import json

import httpx
import pytest

from research_radar.agent.hermes_runtime import HermesHttpRuntime
from research_radar.exceptions import InvalidResponseError
from research_radar.utils.retry import RetryPolicy


async def test_hermes_http_mapping_and_security_headers() -> None:
    seen_request: httpx.Request | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(
            200,
            json={
                "model": "hermes-agent",
                "choices": [{"message": {"content": "radar answer"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    runtime = HermesHttpRuntime(
        base_url="http://hermes.local/v1",
        api_key="secret",
        retry_policy=RetryPolicy(attempts=1),
        client=client,
    )

    result = await runtime.run(
        "research agents", session_key="telegram:42", request_id="request-42"
    )

    assert result.text == "radar answer"
    assert result.usage.total_tokens == 12
    assert seen_request is not None
    assert seen_request.headers["authorization"] == "Bearer secret"
    assert seen_request.headers["x-hermes-session-key"] == "telegram:42"
    assert seen_request.headers["idempotency-key"] == "request-42"
    body = json.loads(seen_request.content)
    assert body["messages"][-1]["content"] == "research agents"
    await client.aclose()


async def test_invalid_hermes_response_is_rejected() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": []}))
    client = httpx.AsyncClient(transport=transport)
    runtime = HermesHttpRuntime(
        base_url="http://hermes.local/v1",
        api_key="secret",
        retry_policy=RetryPolicy(attempts=1),
        client=client,
    )

    with pytest.raises(InvalidResponseError):
        await runtime.run("hello", session_key="test")
    await client.aclose()
