"""Agent-runtime boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod

from research_radar.schemas import RuntimeResponse, ToolHealth


class HermesRuntime(ABC):
    @abstractmethod
    async def run(
        self,
        prompt: str,
        *,
        session_key: str,
        system_instruction: str | None = None,
        request_id: str | None = None,
    ) -> RuntimeResponse:
        """Execute one Hermes turn."""

    @abstractmethod
    async def health(self) -> ToolHealth:
        """Check runtime reachability without executing a model turn."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release runtime resources."""
