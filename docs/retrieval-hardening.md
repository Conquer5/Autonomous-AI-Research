# Retrieval reliability hardening — 8 September 2026

## A. Diagnosis

The committed historical pilot (`eval/reports/live_pilot_v1.json`, 28 August 2026)
found evidence in **1 of 16 runs**. It recorded **zero collector exceptions** despite
15 runs without evidence. That supports an empty-retrieval diagnosis; it does not
establish that upstream services were down. Historical telemetry cannot attribute
those empty results to individual causes.

Source inspection found concrete contributors:

- arXiv wrapped the entire research query in one exact phrase. Multiple independent
  concepts therefore had to appear consecutively. Queries now use individually
  escaped terms joined with AND, preserving category/date filters. This follows
  the [arXiv query contract](https://github.com/arXiv/arxiv-docs/blob/develop/source/help/api/user-manual.md).
- Absent optional tools fell through as successful searches with zero results.
  The planner can still request absent Brave; this is now an explicit availability
  failure with a bounded alternative-provider fallback.
- Empty results had no same-iteration deterministic recovery. Batch partial flags
  and warnings were discarded by the orchestrator.
- RSS had no retry policy. Malformed entries could discard a whole feed, while
  a slow feed could hold up healthy feeds.
- GitHub/Brave assumed dictionary envelopes and complete entry schemas. Unexpected
  envelopes could become empty successes or unclassified exceptions.
- Wall time was checked only between iterations, excluding planning and final LLM
  phases. SQLite busy waits could independently wait 30 seconds.
- Final evidence IDs were not synchronized on every early stop. Novelty counted
  global database discovery rather than evidence newly collected in this run.
- Deterministic synthesis could describe weak/insufficient consensus as consistent.
  Research service traces reported success even for partial or failed results.

A simple authenticated GitHub search and public arXiv search both succeeded outside
the restricted network sandbox before implementation. The sandboxed probe eventually
returned two connection errors and an executor-shutdown timeout warning.
Environment restrictions and application failures must therefore be distinguished.

Baseline gates, before edits:

| Gate | Baseline |
|---|---|
| Ruff lint | PASS |
| Ruff format | FAIL: trailing blank line in `tests/unit/test_evidence.py` |
| Mypy | PASS: 65 source files |
| Pytest | 143 passed, 3 failed, 5 live tests skipped |
| Pip check | No broken requirements; cache-directory warning only |

The three pytest failures were stale evaluation assertions: schema 1.0 versus the
committed 1.1 contract, and zero versus `None` for unmeasured grounding/citations.
These assertions were aligned with the existing contract; metrics were not changed
to manufacture passing results. The baseline formatting issue was also corrected.

## B. Changes

| Files | Purpose |
|---|---|
| `src/research_radar/retrieval.py` | Shared classification, safe typed diagnostics, observed searches, bounded HTTP retry, request/operation correlation |
| `src/research_radar/utils/retry.py`, `utils/deadline.py` | Backoff/jitter cap, server-requested delay, shared workflow deadline |
| `src/research_radar/tools/github.py`, `arxiv.py`, `news.py`, `web.py` | Integrate reliable transport; validate envelopes/entries; preserve valid partial data; deduplicate; handle 204 |
| `src/research_radar/schemas.py`, `research/models.py` | Add backward-compatible failure, latency, operation, coverage and fallback fields |
| `src/research_radar/research/orchestrator.py`, `research/planner.py` | Count bounded fallback queries; enforce await deadlines; preserve partial status, evidence IDs and uncertainty; restore request contexts |
| `src/research_radar/evidence/registry.py`, `research/persistence.py` | Bound SQLite lock waits; persist diagnostics in two additive JSON columns without changing evidence IDs or delivery/version tables |
| `src/research_radar/consensus/extractor.py`, `consensus/synthesis.py` | Safe exception logging; conservative deterministic conclusions; preserve verification lineage |
| `src/research_radar/observability/logging.py`, `service.py` | Structured nested JSON redaction; shared request/run correlation; accurate service trace status |
| `src/research_radar/bootstrap.py` | Give RSS the configured retry policy |
| `src/research_radar/eval/models.py`, `eval/metrics.py` | Record attempts, retries, empty/nonempty/partial queries, latency and failure categories |
| `scripts/run_live_pilot.py`, `scripts/smoke.py`, `scripts/compare_retrieval_runs.py` | Isolated timestamped pilot artifacts and source manifest; failure-isolated smoke checks; reproducible historical comparison |
| `tests/unit/test_retrieval_reliability.py`, `test_research_orchestrator.py` | Failure matrix and pipeline persistence/budget/lineage regression tests |
| `tests/integration/test_live_services.py` | Load the same settings as the application and add RSS live coverage |
| `tests/unit/test_eval.py`, `test_evidence.py` | Correct the documented baseline gate failures |
| `README.md`, `docs/architecture.md`, `docs/tool-system.md`, this report | Document current contracts and results |

No evidence-ID algorithm, content fingerprint algorithm, delivery history, or
versioning schema was replaced. The unrelated untracked user document was untouched.

## C. Retrieval Reliability

Classification includes authentication, authorization, known quota exhaustion,
429, other 4xx, 5xx, timeout, connection, DNS/network, malformed JSON, parser errors,
schema drift, invalid queries, unavailable providers, empty results, partial
responses, storage failures, unexpected exceptions, and cancellation.

Only transient HTTP/network failures are retried. Invalid credentials, invalid
queries, deterministic parsing/schema failures, and known quota exhaustion are
not retried. Exponential backoff and jitter are capped. GitHub rate-limit headers
are distinguished from ordinary 403 responses. Server `Retry-After` and reset
requirements are honored; when the wait exceeds the backoff cap, the call returns
a classified failure instead of retrying early. See
[GitHub's official retry guidance](https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api).

Every attempted search records a query hash, start time, latency, attempts/retries,
result count, status, partial flag and safe failure codes. Correlation follows
service request/run ID → iteration → tool call ID → provider request ID. Feed
resource hashes identify individual configured feeds without exposing their URLs.
SQLite query diagnostics preserve the operation records; raw responses, authorization
headers and exception payloads are not stored in diagnostics. Persisted research
queries use the existing query lineage field with credential-pattern redaction.

Fallback retains at most the first two distinctive query terms. A failed/empty
original search can enqueue one fallback; fallback steps cannot recursively enqueue
more steps. Provider failures select an available alternative. Executed-query
deduplication and the original query/tool budgets apply to every fallback.

GitHub and arXiv intentionally retrieve only one bounded page, at most the caller's
limit (five per research search). More available pages are not automatically fetched;
this avoids multiplying network calls. RSS deduplicates canonical URLs while
preserving case-sensitive paths. Malformed entries do not discard valid siblings.

QUICK remains **1 iteration / 5 tools / 5 queries / 15 evidence / 60s / 3 LLM calls**.
DEEP remains **3 / 12 / 10 / 30 / 180s / 8**. Planning gets at most one quarter of
remaining workflow time; retrieval waits include queueing, throttling and retries.
Synthesis degrades deterministically when time/call capacity is exhausted. SQLite
busy waits use the same remaining deadline. Short synchronous verification and
finalization still have normal scheduling/I/O overhead; this is not a hard-real-time
execution guarantee.

## D. Tests

Final offline gates: **227 passed, 6 live tests skipped**, Ruff lint/format PASS,
Mypy PASS (67 source files), pip dependency check PASS. There are **81 new offline
regression cases**, plus the RSS live test.

Coverage includes 200, 204, 400, 401, 403, 404, 429, 500, 502 and 503 across all four
adapters; timeout/connect/DNS errors; malformed JSON/XML; changed/missing schemas;
duplicate entries; long rate-limit waits; retry recovery/exhaustion; optional
credentials; quota exhaustion; one/multiple/all failed providers; partial data;
query/evidence/time limits; SQLite lock contention; persistence and evidence lineage;
structured redaction; keyword-call contract preservation; and cancellation.

The offline evaluation executed **32 cases / 64 mode runs**, with **5/5 adversarial
fixtures passing**. Simulated results are not evidence of live retrieval quality.

Live checks on the final code: **5 passed, Brave skipped**. Smoke passed Gemini,
Hermes, GitHub, arXiv, and RSS; the final RSS smoke retained three results while
reporting one feed connection failure. This was an observed partial failure, not a
full-coverage success. No Telegram messages were sent by these checks.

## E. Metrics

Final pilot: [report](../eval/hardening/20260908T101701Z/reports/live_pilot_v1.md),
[raw metrics](../eval/hardening/20260908T101701Z/reports/live_pilot_v1.json),
[comparison JSON](../eval/hardening/20260908T101701Z/comparison.json),
[validation](../eval/hardening/20260908T101701Z/validation.json), and
[source/configuration manifest](../eval/hardening/20260908T101701Z/run_manifest.json).

| Metric | Before: 28 Aug | After: 8 Sep, final code |
|---|---:|---:|
| Runs with evidence | 1/16 (6.25%) | 16/16 (100%) |
| Average evidence items per run | 0.31 | 11.81 |
| Reported collector failure rate | 0/49 (incomplete accounting) | 24/81 (29.63%) |
| Actual HTTP attempts ending in error | Not measured | 1/58 (1.72%) |
| Retry rate / HTTP attempts | Not measured | 0/58 (0%) |
| Average run latency | 6.58s | 14.96s |
| Partial-result run rate | 1/16 (6.25%) | 16/16 (100%) |
| Empty search operations | Not measured | 15/58 |
| Recorded budget violations | 0 | 0 |
| Adversarial fixtures | 5/5 | 5/5 |

Of the 24 failed collector steps, 23 were unavailable Brave and one was a GitHub
query rejected as invalid. That invalid query received one HTTP attempt, with no
permanent-error retry. All 16 runs disclosed limited coverage. No transient HTTP
failure occurred during this final pilot, so retry recovery is supported by the
failure fixtures, not by this live sample. RSS isolation was exercised separately
by the real connection failure during smoke testing.

All **16 runs and 81 query records** were persisted. All **189 evidence references**
resolved to registered IDs with canonical URLs and content fingerprints. Operation
IDs matched their query IDs, and the final source hash matched the pilot manifest.

The earlier hardening pilot also found evidence in 16/16 runs (12.69 items/run,
17.19s average). Its [retained report](../eval/hardening/20260908T095708Z/reports/live_pilot_v1.json)
shows normal run-to-run variation; the final-code report above is the acceptance run.

Reproduce the comparison without calling any external provider:

```bash
.venv/bin/python scripts/compare_retrieval_runs.py \
  eval/reports/live_pilot_v1.json \
  eval/hardening/20260908T101701Z/reports/live_pilot_v1.json
```

Review the [human evaluation packet](../eval/hardening/20260908T101701Z/results/live_pilot_human_review.md)
before treating increased evidence yield as increased answer quality.

Definitions: retrieval success means at least one evidence item in a run, not a
judgment of relevance, correctness or completeness. Collector failures include
explicitly unavailable optional tools; the historical implementation silently
reported those as successes. Retry telemetry was not present in the historical
report and must not be read as zero. Compare dates, provider availability and
upstream nondeterminism before making causal claims.

## F. Risks

- Brave has no configured key, so its live authentication/quota behavior was tested
  with fixtures only. Missing coverage is explicitly reported.
- More retrieved evidence does not demonstrate better answers. The generated human
  review packet still needs relevance/completeness/grounding review. Relaxed queries
  can broaden the result set beyond the user's intended scope.
- The planner can still choose an unavailable provider and consume a tool slot.
  Failures are isolated and alternatives are bounded, but this costs coverage budget.
- Exhausted server rate limits may require waits longer than QUICK/DEEP can afford;
  such calls fail safely rather than exceed the workflow budget.
- Unexpected new provider error codes may initially use broader HTTP/schema categories.
  DNS attribution is specific only when the transport preserves a `gaierror` cause.
- A locked/unwritable SQLite database cannot guarantee persistence. Busy waits are
  bounded, evidence already collected survives, and failed persistence is disclosed.
- The current registry/content and verification methods remain heuristic and do not
  independently open full articles to prove every factual statement. This work does
  not establish production readiness for unattended high-stakes research.

## G. Next Priority

Use the pilot's human review packet to assess whether the additional evidence is
relevant and sufficient. Then make planning aware of configured provider capability
so it avoids spending calls on absent Brave, while reporting the resulting coverage
limits. Inspect provider-specific query qualifiers as well: the final pilot captured
a rejected GitHub `repo:` query, and simplified-query relevance remains unscored. Repeat the same pilot with matched provider availability and retain the source
manifest. Track empty-query rates, collector availability and latency over multiple
runs before choosing caching, wider pagination or budget changes.
