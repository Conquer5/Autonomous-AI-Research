from __future__ import annotations

from research_radar.utils.retry import RetryPolicy, call_with_retry


async def test_retry_is_bounded_and_eventually_returns() -> None:
    calls = 0
    delays: list[float] = []

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TimeoutError("temporary")
        return "ok"

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    result = await call_with_retry(
        operation,
        policy=RetryPolicy(attempts=3, base_delay_seconds=0, max_delay_seconds=0, jitter_seconds=0),
        should_retry=lambda exc: isinstance(exc, TimeoutError),
        sleep=fake_sleep,
    )

    assert result == "ok"
    assert calls == 3
    assert delays == [0, 0]
