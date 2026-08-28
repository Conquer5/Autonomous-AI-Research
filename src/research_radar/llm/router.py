"""Small workload-based Gemini model router."""

from __future__ import annotations

from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel

from research_radar.llm.base import LLMProvider
from research_radar.schemas import LLMRequest, LLMResponse, StructuredLLMResponse

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class Workload(StrEnum):
    FAST = "fast"
    DEFAULT = "default"
    REASONING = "reasoning"


class LLMRouter:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        default_model: str,
        fast_model: str | None = None,
        reasoning_model: str | None = None,
    ) -> None:
        self.provider = provider
        self.models = {
            Workload.DEFAULT: default_model,
            Workload.FAST: fast_model or default_model,
            Workload.REASONING: reasoning_model or default_model,
        }

    async def generate(self, request: LLMRequest, workload: Workload) -> LLMResponse:
        routed = request.model_copy(update={"model": request.model or self.models[workload]})
        return await self.provider.generate(routed)

    async def generate_structured(
        self,
        request: LLMRequest,
        schema: type[SchemaT],
        workload: Workload,
    ) -> StructuredLLMResponse[SchemaT]:
        routed = request.model_copy(update={"model": request.model or self.models[workload]})
        return await self.provider.generate_structured(routed, schema)
