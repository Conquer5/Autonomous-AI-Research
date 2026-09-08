# Research Radar — Evaluation & Calibration Report (live_20260908_102101)

> 🔬 **LIVE RESEARCH EXECUTION — REAL-WORLD CALIBRATION DATA**

- **Execution Type:** `LIVE`
- **Generated At:** `2026-09-08T10:21:01.149966+00:00`
- **Dataset Version:** `1.1.0`
- **Evaluated:** `8` cases (16 total runs) across `6` categories

## 1. Research Mode Comparison (QUICK vs DEEP)

| Metric | QUICK Mode | DEEP Mode | Delta / Improvement |
|---|---|---|---|
| **Median Grounding Rate** | `93.8%` | `87.5%` | `-6.2%` |
| **Median Citation Coverage** | `100.0%` | `100.0%` | `+0.0%` |
| **Median Evidence Count** | `10.0` | `14.0` | `+4.0` items |
| **Median Independent Sources** | `7.0` | `12.0` | `+5.0` sources |
| **Median Contradictions Found** | `2.5` | `0.0` | `-2.5` |
| **Gold Contradiction Recall** | `83.3%` | `16.7%` | `-66.7%` |
| **Median Dissenting Points** | `1.5` | `0.0` | `-1.5` |
| **Median Tool Calls** | `4.0` | `6.0` | `+2.0` calls |
| **Median LLM Calls** | `2.0` | `3.0` | `+1.0` calls |
| **Median Latency (Wall Time)** | `10.01s` | `18.16s` | `+8.15s` |
| **Token Usage** | `NOT_MEASURED` | `NOT_MEASURED` | `N/A` |

## 2. Per-Category Breakdown

| Category | QUICK Evidence | DEEP Evidence | QUICK Grounding | DEEP Grounding |
|---|---|---|---|---|
| `agent_protocols` | `10.0` | `17.0` | `100.0%` | `87.5%` |
| `agents` | `8.0` | `15.5` | `75.0%` | `75.0%` |
| `evaluation_benchmarks` | `10.0` | `21.0` | `87.5%` | `87.5%` |
| `local_inference` | `5.0` | `13.0` | `100.0%` | `87.5%` |
| `models_reasoning` | `10.0` | `10.5` | `87.5%` | `81.2%` |
| `rag_retrieval` | `10.0` | `15.0` | `100.0%` | `100.0%` |

## 3. Per-Difficulty Breakdown

| Difficulty | QUICK Ev | DEEP Ev | QUICK Contradictions | DEEP Contradictions |
|---|---|---|---|---|
| `CONTROVERSIAL` | `10.0` | `14.5` | `8.5` | `0.0` |
| `EASY` | `10.0` | `12.5` | `0.0` | `0.0` |
| `HARD` | `8.0` | `15.0` | `2.5` | `0.0` |
| `MODERATE` | `7.5` | `17.0` | `4.0` | `0.5` |

## 4. Adversarial Fixture Results

| Fixture | Contradiction Detected | Dissent Preserved | Status |
|---|---|---|---|
| **official_vs_independent_benchmark** | `Yes` | `Yes` | ✅ PASS |
| **multi_paper_reproducibility** | `Yes` | `Yes` | ✅ PASS |
| **syndicated_copycat_resilience** | `Yes` | `Yes` | ✅ PASS |
| **valid_version_evolution** | `Yes` | `Yes` | ✅ PASS |
| **temporal_benchmark_discrepancy** | `Yes` | `No` | ✅ PASS |

## 5. Observed Failure Taxonomy

| Failure Category | Occurrence Count | Description |
|---|---|---|
| `SOURCE_QUALITY_FAILURE` | `3` | Diagnostic error category |
| `CONTRADICTION_FAILURE` | `3` | Diagnostic error category |

## 6. Routing & Calibration Recommendations

- EVIDENCE DEPTH: DEEP mode yielded +4.0 more evidence items and +5.0 independent sources.
- CONTRADICTION DISCOVERY: QUICK and DEEP showed similar contradiction recall.
- LATENCY OVERHEAD: DEEP mode required +8.15s latency and +2.0 additional tool calls.
- CALIBRATION: QUICK mode is recommended for EASY factual queries to conserve latency. DEEP mode is recommended for CONTROVERSIAL, HARD, or comparative queries.

## 7. Human Evaluation Rubric Guidelines

Score each category from **0 (poor)** to **2 (good)**:
- **RELEVANCE (0-2):** Does the answer address the core question directly?
- **COMPLETENESS (0-2):** Does it address important trade-offs?
- **EVIDENCE QUALITY (0-2):** Are sources authoritative?
- **CONSERVATIVENESS (0-2):** Does it avoid ungrounded claims?
- **USEFULNESS (0-2):** Would this enable an engineering decision?
- **DISSENT HANDLING (0-2):** Are conflicting benchmarks preserved?
