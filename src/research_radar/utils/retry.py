"""Small bounded async retry helper with exponential backoff."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    jitter_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be at least 1")
        if min(self.base_delay_seconds, self.max_delay_seconds, self.jitter_seconds) < 0:
            raise ValueError("retry delays must not be negative")


async def call_with_retry(
    operation: Callable[[], Awaitable[ResultT]],
    *,
    policy: RetryPolicy,
    should_retry: Callable[[Exception], bool],
    on_retry: Callable[[Exception, int], None] | None = None,
    minimum_delay: Callable[[Exception], float] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> ResultT:
    """Execute an operation with bounded exponential backoff."""

    for attempt in range(1, policy.attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            if attempt >= policy.attempts or not should_retry(exc):
                raise
            if on_retry is not None:
                on_retry(exc, attempt)
            base = min(
                policy.max_delay_seconds,
                policy.base_delay_seconds * (2 ** (attempt - 1)),
            )
            delay = min(policy.max_delay_seconds, base + random.uniform(0, policy.jitter_seconds))
            if minimum_delay is not None:
                delay = max(delay, minimum_delay(exc))
            await sleep(delay)
    raise RuntimeError("retry loop exhausted unexpectedly")
