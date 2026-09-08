# Research Tool System (Phase 2)

> Current reliability update (2026-09-08): the application now has bounded research orchestration, consensus/verification, evidence persistence, and retrieval diagnostics. See the [hardening report](retrieval-hardening.md) for current contracts, tests and live measurements. Phase 1–2 descriptions below are historical.

Each research source is independent and returns a `SearchBatch[T]` containing
typed Pydantic items, retrieval time, partial-result status, and warnings. Raw
HTML and unbounded documents never cross the tool boundary.

## GitHub

`GitHubSearchTool` uses `/search/repositories` and supports creation date,
activity date, language, limit, and sort. `GitHubRepositoryAnalyzer` concurrently
fetches repository metadata, preferred README, latest published release, and
recent commits. Optional endpoint failures produce warnings; missing repository
metadata fails the task.

The analyzer caps README context at 12,000 characters by default. Later phases
will retrieve tree/files only after relevance selection.

Repository signals are intentionally heuristic:

- recency: repository age over a one-year horizon;
- activity: days since last push over a 90-day horizon;
- popularity: logarithmic stars/forks;
- growth: approximate stars per repository age, not measured star history;
- relevance: query token overlap across metadata;
- technical depth: size, topics, language, license, and archive state.

## arXiv

`ArxivSearchTool` calls the official query endpoint and parses Atom XML with the
standard library. It supports keyword, supported CS categories, submission date,
and result limits. A single rate lock ensures the configured minimum interval
between requests.

## Brave Web Search

`BraveWebSearchTool` caps queries to provider limits, uses SafeSearch, supports a
custom date range, and returns at most 20 compact results. Only three extra
snippets per result are retained.

Source-quality scoring is a routing heuristic. A high score means “inspect this
primary-looking source first,” not “the claim is proven.” Phase 4 verification
must open and inspect sources before attaching claims.

## Failure Behavior

- Network, timeout, `429`, and `5xx` errors receive bounded exponential retry.
- Permanent `4xx` errors fail immediately.
- User-facing messages do not contain credentials or stack traces.
- Detailed error type is retained in structured application logs.
- One optional GitHub sub-resource failure does not discard repository metadata.

