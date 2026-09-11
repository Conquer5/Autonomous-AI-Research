# Current architecture — 11 September 2026

Telegram handlers authorize users and pass natural-language, QUICK and DEEP
requests to `RadarService.research`, then `ResearchOrchestrator`.

1. Determine topic and any rolling date window.
2. Select one versioned guide and create a plan with the configured backend.
   Auto uses Gemini FAST for QUICK and Hermes for DEEP, with availability fallback.
3. Validate tools against configured adapters. Search within query/time/evidence
   budgets; keep recovery searches within the same temporal constraints.
4. Register evidence, retain source date separately from observation date, analyze
   consensus/contradictions and verify structured claims heuristically.
5. Return answer, citations and uncertainty; persist diagnostics to SQLite.

Application code owns retrieval and evidence persistence. Hermes is an external
HTTP runtime, used for planning and explicit agent utilities. Its internal tools,
memory and server permissions remain a separate boundary. A planning system
instruction is not server-side tool isolation or cancellation.

Guides are packaged application assets, injected into planning requests on demand;
this does not require or claim installation in `~/.hermes/skills`. `/skills` lists
the local application catalog. Each research run records backend and guide version.

Digest remains a distinct collect/rank/deduplicate/synthesize/deliver workflow.
Direct source commands use application adapters. `/memory` still asks Hermes for
its memory; SQLite research persistence does not yet implement conversational recall.

See [contracts and remaining work](ai-efficiency-radar.md),
[retrieval diagnostics](retrieval-hardening.md), and [tool adapters](tool-system.md).
