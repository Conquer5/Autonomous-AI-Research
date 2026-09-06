from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_radar.config import AppSettings
from research_radar.exceptions import ConfigurationError


def test_settings_parse_telegram_allowlist_and_redact_secrets() -> None:
    settings = AppSettings(
        _env_file=None,
        telegram_allowed_user_ids="123, 456,123",
        telegram_bot_token="telegram-secret",
        gemini_api_key="gemini-secret",
    )

    assert settings.telegram_allowed_user_ids == frozenset({123, 456})
    assert "telegram-secret" not in repr(settings)
    assert "gemini-secret" not in repr(settings)


def test_empty_allowlist_fails_when_telegram_is_started() -> None:
    settings = AppSettings(_env_file=None, telegram_bot_token="token")

    with pytest.raises(ConfigurationError, match="Allowlist"):
        settings.require_telegram_token()


def test_non_numeric_allowlist_is_rejected() -> None:
    with pytest.raises(ValidationError, match="numeric"):
        AppSettings(_env_file=None, telegram_allowed_user_ids="123,nope")


def test_low_resource_digest_and_gemini_defaults() -> None:
    settings = AppSettings(_env_file=None)

    assert settings.gemini_model == "gemini-3.5-flash-lite"
    assert settings.gemini_fast_model == "gemini-3.5-flash-lite"
    assert settings.gemini_reasoning_model == "gemini-3.5-flash-lite"
    assert settings.gemini_fallback_models == (
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
    )
    assert settings.digest_items_per_source == 5
    assert settings.digest_max_items == 15
    assert settings.gemini_rpm_limit == 12
    assert settings.max_concurrency == 3
    assert "free open-source AI models and tools" in settings.digest_topics
    assert "local LLM" in settings.digest_search_queries
    assert "https://deepmind.google/blog/rss.xml" in settings.news_feed_urls
