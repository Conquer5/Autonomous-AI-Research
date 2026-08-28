# Phase 1–2 Completion Status

Completed on 2026-08-27.

> Historical snapshot: autonomous digest, RSS, SQLite deduplication, and systemd
> scheduling were added after this Phase 2 checkpoint. See
> [Autonomous Digest](autonomous-digest.md) for the current workflow.

## Delivered

- Python 3.12 package and development environment metadata.
- Pydantic environment configuration with secret types and fail-closed Telegram
  allowlist.
- Provider-neutral `LLMProvider` and async `GeminiProvider` with structured
  output, usage metadata, timeout, retry, and cleanup.
- `HermesRuntime` contract, authenticated HTTP adapter, stable sessions,
  idempotency keys, health check, and fake runtime.
- Async Telegram bot with all requested commands, natural-language messages,
  progress action, long-message splitting, HTML escaping, and safe failures.
- GitHub repository search, heuristic signals, and concurrent progressive
  repository analyzer.
- arXiv Atom search with keyword/category/date filtering and request pacing.
- Brave web search with date filtering, provider limits, compact results, and
  source-priority heuristics.
- Explicit tool registry, application service, structured JSON logging, secret
  redaction, and bounded in-process execution traces.
- Unit tests, opt-in live integration tests, and a credential-aware smoke script.
- Architecture, Hermes, tool-system, environment, command, security, and roadmap
  documentation.

## Verified Locally

```text
ruff check: passed
ruff format --check: passed
mypy src: passed (30 source files)
pytest: 21 passed, 5 live tests skipped
pip check: no broken requirements
offline smoke behavior: passed (safe opt-in skip)
```

## Pending Live Verification

Live verification was not claimed. It requires operator-owned credentials and a
running Hermes process:

```bash
RUN_LIVE_TESTS=1 .venv/bin/pytest -m live tests/integration
RUN_LIVE_TESTS=1 .venv/bin/python scripts/smoke.py all
```

The live suite covers Gemini generation, Hermes health, public GitHub search,
public arXiv search, and Brave Search. Telegram delivery is verified by starting
the bot and sending an allowlisted message after its token is configured.

## Phase Boundary

No Phase 3 research pipeline is claimed. Complex natural-language turns go to
Hermes, while explicit source commands use one application-owned source at a
time. Cross-source planning, bounded parallel collection, evidence
deduplication, ranking, and synthesis belong to Phase 3.
