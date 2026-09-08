# Research Radar — Evaluation & Calibration Report (live_20260908_100146)

> 🔬 **LIVE RESEARCH EXECUTION — REAL-WORLD CALIBRATION DATA**

- **Execution Type:** `LIVE`
- **Generated At:** `2026-09-08T10:01:46.728884+00:00`
- **Dataset Version:** `1.1.0`
- **Evaluated:** `8` cases (16 total runs) across `6` categories

## 1. Research Mode Comparison (QUICK vs DEEP)

| Metric | QUICK Mode | DEEP Mode | Delta / Improvement |
|---|---|---|---|
| **Median Grounding Rate** | `93.8%` | `81.2%` | `-12.5%` |
| **Median Citation Coverage** | `100.0%` | `100.0%` | `+0.0%` |
| **Median Evidence Count** | `10.0` | `15.5` | `+5.5` items |
| **Median Independent Sources** | `7.5` | `11.0` | `+3.5` sources |
| **Median Contradictions Found** | `0.5` | `0.0` | `-0.5` |
| **Gold Contradiction Recall** | `66.7%` | `0.0%` | `-66.7%` |
| **Median Dissenting Points** | `0.5` | `0.0` | `-0.5` |
| **Median Tool Calls** | `4.0` | `6.0` | `+2.0` calls |
| **Median LLM Calls** | `2.0` | `3.0` | `+1.0` calls |
| **Median Latency (Wall Time)** | `10.21s` | `18.91s` | `+8.70s` |
| **Token Usage** | `NOT_MEASURED` | `NOT_MEASURED` | `N/A` |

## 2. Per-Category Breakdown

| Category | QUICK Evidence | DEEP Evidence | QUICK Grounding | DEEP Grounding |
|---|---|---|---|---|
| `agent_protocols` | `10.0` | `20.0` | `100.0%` | `100.0%` |
| `agents` | `10.0` | `18.0` | `81.2%` | `68.8%` |
| `evaluation_benchmarks` | `10.0` | `15.0` | `87.5%` | `62.5%` |
| `local_inference` | `8.0` | `18.0` | `100.0%` | `87.5%` |
| `models_reasoning` | `8.0` | `12.5` | `93.8%` | `87.5%` |
| `rag_retrieval` | `10.0` | `15.0` | `100.0%` | `100.0%` |

## 3. Per-Difficulty Breakdown

| Difficulty | QUICK Ev | DEEP Ev | QUICK Contradictions | DEEP Contradictions |
|---|---|---|---|---|
| `CONTROVERSIAL` | `8.0` | `15.5` | `4.5` | `0.0` |
| `EASY` | `10.0` | `12.5` | `0.0` | `0.0` |
| `HARD` | `10.0` | `20.0` | `3.0` | `0.0` |
| `MODERATE` | `9.0` | `16.5` | `4.0` | `0.0` |

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
| `SOURCE_QUALITY_FAILURE` | `4` | Diagnostic error category |
| `CONTRADICTION_FAILURE` | `3` | Diagnostic error category |

## 6. Routing & Calibration Recommendations

- EVIDENCE DEPTH: DEEP mode yielded +5.5 more evidence items and +3.5 independent sources.
- CONTRADICTION DISCOVERY: QUICK and DEEP showed similar contradiction recall.
- LATENCY OVERHEAD: DEEP mode required +8.70s latency and +2.0 additional tool calls.
- CALIBRATION: QUICK mode is recommended for EASY factual queries to conserve latency. DEEP mode is recommended for CONTROVERSIAL, HARD, or comparative queries.

## 7. Human Evaluation Rubric Guidelines

Score each category from **0 (poor)** to **2 (good)**:
- **RELEVANCE (0-2):** Does the answer address the core question directly?
- **COMPLETENESS (0-2):** Does it address important trade-offs?
- **EVIDENCE QUALITY (0-2):** Are sources authoritative?
- **CONSERVATIVENESS (0-2):** Does it avoid ungrounded claims?
- **USEFULNESS (0-2):** Would this enable an engineering decision?
- **DISSENT HANDLING (0-2):** Are conflicting benchmarks preserved?
