# AI developments and efficiency radar — 11 September 2026

The product follows AI developments to help the operator decide what to try with
Codex, Hermes and Gemini, based on evidence and cost per successful task.
It covers AI releases, coding-agent efficiency, model access, and efficient local
inference. General and historical research questions remain supported.

## Implemented behavior

- Topic selection chooses one versioned guide: research-methodology,
  ai-developments, agent-efficiency or model-access. Guides are packaged under
  `src/research_radar/skills/` and loaded using Python package resources.
- `RESEARCH_PLANNER_BACKEND=auto`: QUICK uses the fast Gemini route when configured;
  DEEP uses Hermes when configured. With only Hermes, QUICK also uses Hermes.
  With only Gemini, both use Gemini. With neither, planning is deterministic.
  Explicit `gemini`, `hermes`, and `deterministic` overrides are supported.
- The application sends the selected guide content in the system instruction of
  one planning request. Hermes returns a JSON plan; the application validates
  sources and limits, and retains ownership of retrieval and evidence verification.
  Invalid JSON or a failed planning turn uses deterministic fallback, without a
  second provider call. Planning retains the existing quarter-workflow timeout.
- Hermes planning sessions use a fresh random scope for each request. No previous
  user's conversation is deliberately loaded into the planning request.
- Missing providers are replaced or skipped before spending retrieval slots.
  The answer reports the missing coverage. This does not imply the substitute
  offers equivalent coverage (RSS cannot replace unrestricted web search).
- Planning backend, guide name/version and temporal focus are stored in run
  diagnostics. `/skills` lists these application guides without spending a model
  call. It does not claim they are installed in the global Hermes skill directory.
  Guide fields identify the selected policy; deterministic fallback does not send
  a guide to a model.
- Synthesis instructions require attribution of developer claims, conservative
  model comparisons, and official evidence for price/quota/free-access claims.
  Suggested experiments must be labeled unexecuted. This is a prompting contract,
  not proof that the current heuristic verifier enforces semantic correctness.

## What “latest” means

Natural-language research and `/research` / `/deepresearch` recognize `terbaru`,
`terkini`, `sekarang`, `latest`, `recent`, `current`, `hari ini`, `minggu ini`, and
numeric day/week windows. Default recent window: 14 inclusive UTC calendar days.
`hari ini` means the current UTC day; `minggu ini` means a rolling seven-day window.
Numeric windows are clamped to 1–365 days with an explicit note. Arbitrary date
ranges are not parsed; use a numeric rolling window for an enforced cutoff.

Source adapters receive date filters. Collected sources are additionally checked:
future dates are rejected; recent research also excludes undated or older sources
and reports reduced coverage. Fallback queries keep the same cutoff. Historical
questions without recency wording retain older and undated evidence.

For GitHub, the date filter uses last push (or update timestamp when push is
unavailable), not repository creation. Repository activity is never proof of a
new release. Paper dates refer to initial publication. Web/RSS dates are provider
metadata and are not independently verified publication dates. Retrieval date and
source date are carried separately in evidence sent to synthesis. The research
reply displays its UTC cutoff.

Digest retains its separate configured windows: repositories/papers 30 days,
news 14 days. Repository digest discovery still uses creation date; it is not a
release watcher. RSS entries without dates may remain discovery candidates in the
digest and must not establish recency. Direct source commands retain their own
existing behavior. Strict run-level filtering currently applies to research.

## Configuration and operation

New defaults include coding-agent and context-efficiency queries without increasing
the default GitHub/arXiv query count. Custom `.env` topic/query overrides are
preserved. The local workspace's two exact legacy sample values (DIGEST_TOPICS
and DIGEST_SEARCH_QUERIES) were migrated to the new focus. For other deployments,
copy the non-secret DIGEST settings from `.env.example` as appropriate.
Model IDs and credentials are unchanged.

Restart the bot after updating code/configuration. This implementation does not
restart running services, send Telegram messages, install candidate repositories,
or change the global Hermes installation. These are separate rollout actions.

## Cost boundary

`state.llm_calls` counts application model operations / Hermes turns, not every
internal model call or transport retry. A zero model budget uses deterministic
planning. Internal Hermes tools and token limits remain governed by the Hermes
server configuration; a system instruction is not a tool permission boundary.
Client timeout does not establish cancellation of remote server work.

Guides ask Hermes to plan without tool execution. Use a restricted Hermes server
profile for enforced isolation. Native skill discovery/installation and server-side
run cancellation are not claimed. See the official [API server contract](https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server)
and [skills contract](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills/).

## Roadmap and acceptance

Original Phase 3: bounded orchestration, capability-aware planning and temporal
filtering are implemented; real answer relevance still needs review.
Phase 4: consensus/verification exists but is heuristic; full-text grounding and
matched-task quality calibration remain open.
Phase 5: evidence and research execution persistence exists; reusable research
memory, per-user retrieval, compaction and retention are not complete.
Phase 6: versioned guides now participate in the planning workflow; automated
candidate evaluation, canary promotion and controlled learning remain open.
The later P0–P7 priority labels in WORKFLOW_DAN_RENCANA_PROYEK.txt are work order,
not substitutes for the original phase numbers.

Next experiment: select 10 representative tasks; compare the existing workflow,
Gemini with a selected guide/tool, and the stronger-model baseline where access
exists. Keep tasks and acceptance tests fixed. Measure success, input/output
usage, retries, elapsed time and human interventions. Report subscription quota
separately from API spend. Preserve model/version, configuration, source date,
failures and unmeasured values. No cost saving or model equivalence is established
by the offline implementation tests.

## Validation of this change

241 offline tests passed; six opt-in live tests skipped. Ruff lint/format, mypy
(68 source files), and git diff whitespace checks passed. Four guide resources
were read successfully through the package loader; each is under 3,000 characters.
Tests cover backend routing, invalid Hermes output, per-request session separation,
unavailable source handling without a wasted tool slot, time-window filtering,
repository activity dates, zero-model budgets and persisted policy provenance.
No live Gemini/Hermes turn or Telegram delivery was performed for this change.
