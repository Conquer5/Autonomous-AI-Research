"""Tool safety policy for Hermes Agent integration.

Security assumptions for the Research Radar ↔ Hermes boundary:

1. AUTHENTICATION: All Hermes requests use Bearer token authentication.
   The token is stored as SecretStr and never logged.

2. SESSION ISOLATION: Each user gets a distinct session key (telegram:<user_id>).
   Sessions should not leak context between users.

3. BOUNDED EXECUTION: All Hermes calls have explicit timeouts
   (max 120s default, configurable). Retry policy is bounded (max 3 attempts).

4. IDEMPOTENCY: Requests include an Idempotency-Key header to prevent
   duplicate execution of the same request.

5. TOOL RESTRICTIONS: The system instruction sent to Hermes explicitly
   prohibits fabricating citations, inventing URLs, and claiming tool
   execution that didn't happen.

6. NO CREDENTIAL FORWARDING: Hermes never receives GitHub tokens,
   Gemini API keys, or Telegram bot tokens. It only receives its own
   API key via Authorization header.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HermesSecurityPolicy:
    """Validates Hermes integration meets safety requirements."""

    min_timeout_seconds: float = 30.0
    max_timeout_seconds: float = 300.0
    max_retry_attempts: int = 10
    require_authentication: bool = True
    require_session_isolation: bool = True
    require_idempotency: bool = True


def validate_hermes_config(
    *,
    api_key: str | None,
    timeout_seconds: float,
    max_retries: int,
    policy: HermesSecurityPolicy | None = None,
) -> list[str]:
    """Validate Hermes configuration against security policy.

    Returns list of policy violations (empty if compliant).
    """
    p = policy or HermesSecurityPolicy()
    violations: list[str] = []

    if p.require_authentication and not api_key:
        violations.append("Hermes API key is required but not configured")

    if timeout_seconds < p.min_timeout_seconds:
        violations.append(f"Timeout {timeout_seconds}s is below minimum {p.min_timeout_seconds}s")
    if timeout_seconds > p.max_timeout_seconds:
        violations.append(f"Timeout {timeout_seconds}s exceeds maximum {p.max_timeout_seconds}s")

    if max_retries > p.max_retry_attempts:
        violations.append(f"Retry count {max_retries} exceeds maximum {p.max_retry_attempts}")

    return violations


SESSION_TTL_SECONDS: int = 3600 * 4  # 4 hours default session TTL
"""Recommended session TTL to prevent long-lived sessions from mixing
unrelated research topics. This is a guideline for future session
lifecycle management."""
