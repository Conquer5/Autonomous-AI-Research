"""Domain exceptions with messages that are safe to expose at API boundaries."""

from __future__ import annotations


class RadarError(Exception):
    """Base exception for expected application failures."""

    def __init__(self, public_message: str, *, detail: str | None = None) -> None:
        super().__init__(detail or public_message)
        self.public_message = public_message
        self.detail = detail


class ConfigurationError(RadarError):
    """Raised when a requested component is not configured."""


class ExternalServiceError(RadarError):
    """Raised when an upstream service fails or returns an invalid response."""

    def __init__(
        self,
        service: str,
        public_message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        detail: str | None = None,
    ) -> None:
        super().__init__(public_message, detail=detail)
        self.service = service
        self.status_code = status_code
        self.retryable = retryable


class AuthorizationError(RadarError):
    """Raised for a denied user or operation."""


class InvalidResponseError(ExternalServiceError):
    """Raised when an upstream response cannot be normalized."""
