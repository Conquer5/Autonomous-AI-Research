# Autonomous AI Research Radar — Evaluation & Calibration Report (eval_20260828_090827)

- **Generated At:** `2026-08-28T09:08:27.108367+00:00`
- **Dataset Version:** `1.0.0`
- **Total Test Cases:** `32` across `7` categories

## 1. Research Mode Comparison (QUICK vs DEEP)

| Metric | QUICK Mode | DEEP Mode | Delta / Improvement |
|---|---|---|---|
| **Median Grounding Rate** | `100.0%` | `100.0%` | `+0.0%` |
| **Median Evidence Count** | `2.0` | `5.0` | `+3.0` items |
| **Median Independent Sources** | `2.0` | `5.0` | `+3.0` sources |
| **Median Contradictions Found** | `0.0` | `0.0` | `+0.0` |
| **Contradiction Recall** | `0.0%` | `0.0%` | `+0.0%` |
| **Median Dissenting Points** | `0.0` | `0.0` | `+0.0` |
| **Median Tool Calls** | `1.0` | `4.0` | `+3.0` calls |
| **Median LLM Calls** | `0.0` | `2.0` | `+2.0` calls |
| **Median Latency (Wall Time)** | `1.00s` | `5.00s` | `+4.00s` |

## 2. Per-Category Breakdown

| Category | QUICK Evidence | DEEP Evidence | QUICK Grounding | DEEP Grounding |
|---|---|---|---|---|
| `agent_protocols` | `2.0` | `5.0` | `100.0%` | `100.0%` |
| `agents` | `2.0` | `5.0` | `100.0%` | `100.0%` |
| `ai_engineering` | `2.0` | `5.0` | `100.0%` | `100.0%` |
| `evaluation_benchmarks` | `2.0` | `5.0` | `100.0%` | `100.0%` |
| `local_inference` | `2.0` | `5.0` | `100.0%` | `100.0%` |
| `models_reasoning` | `2.0` | `5.0` | `100.0%` | `100.0%` |
| `rag_retrieval` | `2.0` | `5.0` | `100.0%` | `100.0%` |

## 3. Per-Difficulty Breakdown

| Difficulty | QUICK Evidence | DEEP Evidence | QUICK Contradictions | DEEP Contradictions |
|---|---|---|---|---|
| `CONTROVERSIAL` | `2.0` | `5.0` | `0.0` | `0.0` |
| `EASY` | `2.0` | `5.0` | `0.0` | `0.0` |
| `HARD` | `2.0` | `5.0` | `0.0` | `0.0` |
| `MODERATE` | `2.0` | `5.0` | `0.0` | `0.0` |

## 4. Adversarial Fixture Results

| Fixture | Contradiction Detected | Dissent Preserved | Confidence Capped | Status |
|---|---|---|---|---|
| **official_vs_independent_benchmark** | `Yes` | `Yes` | `Yes` | ✅ PASS |
| **multi_paper_reproducibility** | `Yes` | `Yes` | `Yes` | ✅ PASS |
| **syndicated_copycat_resilience** | `Yes` | `Yes` | `Yes` | ✅ PASS |
| **valid_version_evolution** | `Yes` | `Yes` | `No` | ✅ PASS |
| **temporal_benchmark_discrepancy** | `Yes` | `No` | `Yes` | ✅ PASS |

## 5. Observed Failure Taxonomy

| Failure Category | Occurrence Count | Description |
|---|---|---|
| `CONTRADICTION_FAILURE` | `12` | Diagnostic error category |

## 6. Routing & Calibration Recommendations

- DEEP BENEFICIAL: Questions with CONTROVERSIAL or HARD difficulty requiring cross-source synthesis.
- DEEP BENEFICIAL: Comparative queries across multiple frameworks or benchmarks.
- QUICK SUFFICIENT: EASY technical documentation and parameter lookup queries.
- QUICK SUFFICIENT: Single-repository or single-tool verification queries.

## 7. Human Evaluation Rubric Guidelines

Score each category from **0 (poor)** to **2 (good)**:
- **RELEVANCE (0-2):** Does the answer address the core question directly?
- **COMPLETENESS (0-2):** Does it address all important dimensions and trade-offs?
- **EVIDENCE QUALITY (0-2):** Are sources authoritative (primary/academic/official)?
- **CONSERVATIVENESS (0-2):** Does it avoid ungrounded or speculative claims?
- **USEFULNESS (0-2):** Would this enable an informed engineering decision?
- **DISSENT HANDLING (0-2):** Are conflicting benchmarks and minority findings preserved?
