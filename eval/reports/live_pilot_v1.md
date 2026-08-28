# Research Radar — Evaluation & Calibration Report (live_20260828_092606)

> 🔬 **LIVE RESEARCH EXECUTION — REAL-WORLD CALIBRATION DATA**

- **Execution Type:** `LIVE`
- **Generated At:** `2026-08-28T09:26:06.068353+00:00`
- **Dataset Version:** `1.1.0`
- **Evaluated:** `8` cases (16 total runs) across `6` categories

## 1. Research Mode Comparison (QUICK vs DEEP)

| Metric | QUICK Mode | DEEP Mode | Delta / Improvement |
|---|---|---|---|
| **Median Grounding Rate** | `N/A` | `100.0%` | `N/A` |
| **Median Citation Coverage** | `N/A` | `100.0%` | `N/A` |
| **Median Evidence Count** | `0.0` | `0.0` | `+0.0` items |
| **Median Independent Sources** | `0.0` | `0.0` | `+0.0` sources |
| **Median Contradictions Found** | `0.0` | `0.0` | `+0.0` |
| **Gold Contradiction Recall** | `0.0%` | `0.0%` | `+0.0%` |
| **Median Dissenting Points** | `0.0` | `0.0` | `+0.0` |
| **Median Tool Calls** | `2.0` | `4.0` | `+2.0` calls |
| **Median LLM Calls** | `1.0` | `1.0` | `+0.0` calls |
| **Median Latency (Wall Time)** | `5.14s` | `5.09s` | `-0.05s` |
| **Token Usage** | `NOT_MEASURED` | `NOT_MEASURED` | `N/A` |

## 2. Per-Category Breakdown

| Category | QUICK Evidence | DEEP Evidence | QUICK Grounding | DEEP Grounding |
|---|---|---|---|---|
| `agent_protocols` | `0.0` | `5.0` | `N/A` | `100.0%` |
| `agents` | `0.0` | `0.0` | `N/A` | `N/A` |
| `evaluation_benchmarks` | `0.0` | `0.0` | `N/A` | `N/A` |
| `local_inference` | `0.0` | `0.0` | `N/A` | `N/A` |
| `models_reasoning` | `0.0` | `0.0` | `N/A` | `N/A` |
| `rag_retrieval` | `0.0` | `0.0` | `N/A` | `N/A` |

## 3. Per-Difficulty Breakdown

| Difficulty | QUICK Ev | DEEP Ev | QUICK Contradictions | DEEP Contradictions |
|---|---|---|---|---|
| `CONTROVERSIAL` | `0.0` | `0.0` | `0.0` | `0.0` |
| `EASY` | `0.0` | `0.0` | `0.0` | `0.0` |
| `HARD` | `0.0` | `2.5` | `0.0` | `0.0` |
| `MODERATE` | `0.0` | `0.0` | `0.0` | `0.0` |

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
| `RETRIEVAL_FAILURE` | `15` | Diagnostic error category |

## 6. Routing & Calibration Recommendations

- EVIDENCE DEPTH: Evidence volume difference between QUICK and DEEP was modest.
- CONTRADICTION DISCOVERY: QUICK and DEEP showed similar contradiction recall.
- CALIBRATION: QUICK mode is recommended for EASY factual queries to conserve latency. DEEP mode is recommended for CONTROVERSIAL, HARD, or comparative queries.

## 7. Human Evaluation Rubric Guidelines

Score each category from **0 (poor)** to **2 (good)**:
- **RELEVANCE (0-2):** Does the answer address the core question directly?
- **COMPLETENESS (0-2):** Does it address important trade-offs?
- **EVIDENCE QUALITY (0-2):** Are sources authoritative?
- **CONSERVATIVENESS (0-2):** Does it avoid ungrounded claims?
- **USEFULNESS (0-2):** Would this enable an engineering decision?
- **DISSENT HANDLING (0-2):** Are conflicting benchmarks preserved?
