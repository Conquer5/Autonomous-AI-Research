"""Environment-backed application configuration."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from research_radar.exceptions import ConfigurationError


def _parse_user_ids(value: object) -> frozenset[int]:
    if value is None or value == "":
        return frozenset()
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_items = list(value)
    else:
        raise ValueError("must be a comma-separated list of Telegram user IDs")
    try:
        parsed = frozenset(int(item) for item in raw_items)
    except (TypeError, ValueError) as exc:
        raise ValueError("must contain only numeric Telegram user IDs") from exc
    if any(user_id <= 0 for user_id in parsed):
        raise ValueError("Telegram user IDs must be positive integers")
    return parsed


TelegramUserIds = Annotated[frozenset[int], BeforeValidator(_parse_user_ids)]


def _parse_csv_strings(value: object) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        items: Iterable[object] = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = value
    else:
        raise ValueError("must be a comma-separated list")
    return tuple(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))


CsvStrings = Annotated[tuple[str, ...], BeforeValidator(_parse_csv_strings)]

DEFAULT_DIGEST_TOPICS = (
    "coding agent efficiency and token optimization",
    "AI agents and agent frameworks",
    "efficient local AI and CPU-only inference",
    "free open-source AI models and tools",
)
DEFAULT_DIGEST_SEARCH_QUERIES = (
    "coding agent",
    "context optimization",
    "local LLM",
    "open source AI model",
)
DEFAULT_GEMINI_FALLBACK_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
)
DEFAULT_NEWS_QUERIES = (
    "coding agent token efficiency tools",
    "AI model free tier pricing availability",
    "AI model release launch announcement GPT Gemini",
)
DEFAULT_NEWS_FEEDS = (
    "https://news.google.com/rss/search?q=AI+model+release+when:14d&hl=en-US&gl=US&ceid=US:en",
    "https://openai.com/news/rss.xml",
    "https://deepmind.google/blog/rss.xml",
    "https://research.google/blog/rss/",
    "https://huggingface.co/blog/feed.xml",
    "https://github.blog/feed/",
    "https://blog.cloudflare.com/rss/",
)


class AppSettings(BaseSettings):
    """Validated settings loaded from process environment and optional `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
        enable_decoding=False,
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    request_timeout_seconds: float = Field(default=120.0, gt=0, le=300)
    max_retries: int = Field(default=3, ge=1, le=10)
    max_concurrency: int = Field(default=3, ge=1, le=20)

    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_fast_model: str | None = "gemini-3.5-flash-lite"
    gemini_reasoning_model: str | None = "gemini-3.5-flash-lite"
    gemini_fallback_models: CsvStrings = DEFAULT_GEMINI_FALLBACK_MODELS
    gemini_rpm_limit: int = Field(default=12, ge=1, le=60)

    hermes_api_url: HttpUrl = HttpUrl("http://127.0.0.1:8642/v1")
    hermes_api_key: SecretStr | None = None
    hermes_model_name: str = "hermes-agent"
    research_planner_backend: Literal["auto", "gemini", "hermes", "deterministic"] = "auto"
    research_recent_days: int = Field(default=14, ge=1, le=365)

    telegram_bot_token: SecretStr | None = None
    telegram_allowed_user_ids: TelegramUserIds = frozenset()

    github_token: SecretStr | None = None
    github_api_url: HttpUrl = HttpUrl("https://api.github.com")
    github_api_version: str = "2026-03-10"

    arxiv_api_url: HttpUrl = HttpUrl("https://export.arxiv.org/api/query")
    arxiv_min_interval_seconds: float = Field(default=3.0, ge=0, le=60)

    brave_search_api_key: SecretStr | None = None
    brave_search_api_url: HttpUrl = HttpUrl("https://api.search.brave.com/res/v1/web/search")

    digest_topics: CsvStrings = DEFAULT_DIGEST_TOPICS
    digest_search_queries: CsvStrings = DEFAULT_DIGEST_SEARCH_QUERIES
    digest_items_per_source: int = Field(default=5, ge=1, le=10)
    digest_max_items: int = Field(default=15, ge=1, le=30)
    digest_news_queries: CsvStrings = DEFAULT_NEWS_QUERIES
    digest_repository_days: int = Field(default=30, ge=1, le=365)
    digest_paper_days: int = Field(default=30, ge=1, le=365)
    digest_news_days: int = Field(default=14, ge=1, le=90)
    digest_min_interval_hours: float = Field(default=12, ge=0, le=168)
    digest_state_path: Path = Path(".data/research-radar.sqlite3")
    news_feed_urls: CsvStrings = DEFAULT_NEWS_FEEDS

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("invalid log level")
        return normalized

    def require_gemini_key(self) -> str:
        if self.gemini_api_key is None:
            raise ConfigurationError("Gemini belum dikonfigurasi. Isi GEMINI_API_KEY.")
        return self.gemini_api_key.get_secret_value()

    def require_hermes_key(self) -> str:
        if self.hermes_api_key is None:
            raise ConfigurationError("Hermes belum dikonfigurasi. Isi HERMES_API_KEY.")
        return self.hermes_api_key.get_secret_value()

    def require_telegram_token(self) -> str:
        if self.telegram_bot_token is None:
            raise ConfigurationError("Telegram belum dikonfigurasi. Isi TELEGRAM_BOT_TOKEN.")
        if not self.telegram_allowed_user_ids:
            raise ConfigurationError("Allowlist Telegram kosong. Isi TELEGRAM_ALLOWED_USER_IDS.")
        return self.telegram_bot_token.get_secret_value()

    def require_brave_key(self) -> str:
        if self.brave_search_api_key is None:
            raise ConfigurationError("Web search belum dikonfigurasi. Isi BRAVE_SEARCH_API_KEY.")
        return self.brave_search_api_key.get_secret_value()
