"""Provider-neutral LLM contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from research_radar.schemas import LLMRequest, LLMResponse, StructuredLLMResponse

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate an unstructured response."""

    @abstractmethod
    async def generate_structured(
        self, request: LLMRequest, schema: type[SchemaT]
    ) -> StructuredLLMResponse[SchemaT]:
        """Generate and validate a structured response."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release provider resources."""
