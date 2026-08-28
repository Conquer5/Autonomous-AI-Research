"""Curated and versioned research evaluation dataset for Autonomous AI Research Radar."""

from __future__ import annotations

import json
from pathlib import Path

from research_radar.eval.models import (
    DifficultyLevel,
    EvaluationCase,
    EvaluationCategory,
    EvaluationDataset,
    ExpectedDisagreement,
    ResearchCharacteristics,
)

DEFAULT_EVALUATION_CASES: list[EvaluationCase] = [
    # 1. AI Agents
    EvaluationCase(
        id="agents_001",
        question="Is Model Context Protocol (MCP) becoming the standard for agent tooling?",
        category=EvaluationCategory.AGENTS,
        difficulty=DifficultyLevel.HARD,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            requires_recent_evidence=True,
            likely_disagreement=True,
            requires_primary_sources=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["vendor-adoption", "transport-limitations", "security-model"],
            notes="Vendor excitement vs developer concerns on auth and complexity.",
        ),
        evaluation_tags=["mcp", "agent-tooling", "industry-standard"],
        description="Assesses multi-vendor adoption and technical critique of MCP.",
    ),
    EvaluationCase(
        id="agents_002",
        question="What are the key architectural differences between AutoGen and CrewAI?",
        category=EvaluationCategory.AGENTS,
        difficulty=DifficultyLevel.MODERATE,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            requires_primary_sources=True,
            comparative=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["graph-execution", "role-playing"],
            notes="Architectural differences documented in official repositories.",
        ),
        evaluation_tags=["framework-comparison", "multi-agent", "autogen", "crewai"],
        description="Comparative analysis of execution graphs vs role-playing.",
    ),
    EvaluationCase(
        id="agents_003",
        question="Do multi-agent architectures consistently outperform single-agent systems?",
        category=EvaluationCategory.AGENTS,
        difficulty=DifficultyLevel.CONTROVERSIAL,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            likely_disagreement=True,
            benchmark_oriented=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["coordination-overhead", "token-cost-vs-accuracy"],
            notes="Contested debate: ensemble benefits vs token overhead.",
        ),
        evaluation_tags=["multi-agent-vs-single", "cost-benefit", "reasoning"],
        description="Controversial debate on coordination overhead vs accuracy.",
    ),
    EvaluationCase(
        id="agents_004",
        question="What is the primary role of memory persistence in LangGraph agents?",
        category=EvaluationCategory.AGENTS,
        difficulty=DifficultyLevel.EASY,
        characteristics=ResearchCharacteristics(
            requires_primary_sources=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["checkpoints", "state-recovery"],
            notes="Standard technical documentation lookup.",
        ),
        evaluation_tags=["langgraph", "memory", "checkpoints"],
        description="Factual technical documentation lookup.",
    ),
    EvaluationCase(
        id="agents_005",
        question="How reliable are autonomous code-editing agents in large existing repositories?",
        category=EvaluationCategory.AGENTS,
        difficulty=DifficultyLevel.HARD,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            likely_disagreement=True,
            benchmark_oriented=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["swe-bench-score-vs-real-productivity", "regression-risks"],
            notes="Divergence between benchmark pass rates and real defect rates.",
        ),
        evaluation_tags=["swe-bench", "coding-agents", "reliability"],
        description="Evaluates benchmark scores vs developer productivity studies.",
    ),
    # 2. Local AI / Inference
    EvaluationCase(
        id="local_001",
        question="What are the minimum VRAM requirements for running DeepSeek-R1 locally?",
        category=EvaluationCategory.LOCAL_INFERENCE,
        difficulty=DifficultyLevel.MODERATE,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            benchmark_oriented=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["gguf-quantization", "vram-limits"],
            notes="Quantization VRAM requirements are mathematically bounded.",
        ),
        evaluation_tags=["deepseek", "vram", "quantization", "gguf"],
        description="Evaluates precision requirements across 4-bit, 8-bit, and full weights.",
    ),
    EvaluationCase(
        id="local_002",
        question="vLLM vs Ollama: throughput and concurrency comparison for production serving.",
        category=EvaluationCategory.LOCAL_INFERENCE,
        difficulty=DifficultyLevel.MODERATE,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            comparative=True,
            benchmark_oriented=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["pagedattention-throughput", "developer-ergonomics"],
            notes="vLLM targets high concurrency; Ollama optimizes developer simplicity.",
        ),
        evaluation_tags=["vllm", "ollama", "throughput", "pagedattention"],
        description="Examines batching, PagedAttention, and developer UX tradeoffs.",
    ),
    EvaluationCase(
        id="local_003",
        question="Does Speculative Decoding yield consistent speedups across small edge devices?",
        category=EvaluationCategory.LOCAL_INFERENCE,
        difficulty=DifficultyLevel.CONTROVERSIAL,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            likely_disagreement=True,
            benchmark_oriented=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["memory-bandwidth-bottlenecks", "cpu-vs-gpu-speedup"],
            notes="GPU cluster speedup collapses on bandwidth-bound edge CPUs.",
        ),
        evaluation_tags=["speculative-decoding", "edge-ai", "latency"],
        description="Investigates divergence between cloud GPU and edge CPU speedups.",
    ),
    EvaluationCase(
        id="local_004",
        question="What quantization method is supported by llama.cpp for AMD GPUs?",
        category=EvaluationCategory.LOCAL_INFERENCE,
        difficulty=DifficultyLevel.EASY,
        characteristics=ResearchCharacteristics(
            requires_primary_sources=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["rocm-support", "k-quants"],
            notes="Deterministic documentation specification.",
        ),
        evaluation_tags=["llamacpp", "rocm", "amd"],
        description="Direct hardware backend compatibility lookup.",
    ),
    EvaluationCase(
        id="local_005",
        question="Is Apple Silicon unified memory competitive with discrete GPUs for 70B models?",
        category=EvaluationCategory.LOCAL_INFERENCE,
        difficulty=DifficultyLevel.HARD,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            likely_disagreement=True,
            comparative=True,
            benchmark_oriented=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["memory-capacity-vs-flops", "tokens-per-second"],
            notes="Mac Studio fits large models in VRAM cheaply, but FLOPS are slower.",
        ),
        evaluation_tags=["apple-silicon", "mac-studio", "vram-bandwidth"],
        description="Contrasts memory capacity advantages against raw compute FLOPS limits.",
    ),
    # 3. RAG / Retrieval
    EvaluationCase(
        id="rag_001",
        question="GraphRAG vs Hybrid Search: when does Knowledge Graph justify the indexing cost?",
        category=EvaluationCategory.RAG_RETRIEVAL,
        difficulty=DifficultyLevel.HARD,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            comparative=True,
            benchmark_oriented=True,
            likely_disagreement=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["global-synthesis-benefits", "indexing-cost"],
            notes="Superior global QA claims contrasted with production indexing cost.",
        ),
        evaluation_tags=["graphrag", "hybrid-search", "bm25", "indexing-cost"],
        description="Evaluates global query synthesis benefits vs indexing latency.",
    ),
    EvaluationCase(
        id="rag_002",
        question="Does Late Chunking outperform traditional chunking in retrieval benchmarks?",
        category=EvaluationCategory.RAG_RETRIEVAL,
        difficulty=DifficultyLevel.MODERATE,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            benchmark_oriented=True,
            requires_recent_evidence=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["jina-embeddings", "context-preservation"],
            notes="Retrieval benchmarks show consistent context preservation.",
        ),
        evaluation_tags=["late-chunking", "embeddings", "jina", "retrieval-quality"],
        description="Examines context-aware token embedding chunking vs semantic splitters.",
    ),
    EvaluationCase(
        id="rag_003",
        question="Is Vector DB search sufficient for precise factual QA without reranking?",
        category=EvaluationCategory.RAG_RETRIEVAL,
        difficulty=DifficultyLevel.CONTROVERSIAL,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            likely_disagreement=True,
            requires_primary_sources=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["cosine-similarity-recall", "cross-encoder-necessity"],
            notes="Vendor marketing claims vs rigorous studies proving reranker necessity.",
        ),
        evaluation_tags=["reranking", "cross-encoders", "precision", "vector-search"],
        description="Assesses precision degradation of raw cosine similarity vs cross-encoders.",
    ),
    EvaluationCase(
        id="rag_004",
        question="What is the default vector distance metric used in Qdrant?",
        category=EvaluationCategory.RAG_RETRIEVAL,
        difficulty=DifficultyLevel.EASY,
        characteristics=ResearchCharacteristics(
            requires_primary_sources=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["cosine-distance", "default-metric"],
            notes="Official Qdrant documentation fact.",
        ),
        evaluation_tags=["qdrant", "vector-db", "distance-metrics"],
        description="Basic documentation and configuration query.",
    ),
    # 4. AI Engineering
    EvaluationCase(
        id="eng_001",
        question="DSPy vs Few-Shot: does automatic prompt optimization generalize in production?",
        category=EvaluationCategory.AI_ENGINEERING,
        difficulty=DifficultyLevel.HARD,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            comparative=True,
            likely_disagreement=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["teleprompter-generalization", "drift-fragility"],
            notes="Academic benchmark gains vs brittle production prompt adaptation.",
        ),
        evaluation_tags=["dspy", "prompt-optimization", "teleprompter", "production-ai"],
        description="Synthesizes academic claims with enterprise maintenance feedback.",
    ),
    EvaluationCase(
        id="eng_002",
        question="What are essential telemetry metrics for monitoring LLM drift and latency?",
        category=EvaluationCategory.AI_ENGINEERING,
        difficulty=DifficultyLevel.MODERATE,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            requires_primary_sources=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["ttft", "tokens-per-second", "drift-distance"],
            notes="Industry standard OpenTelemetry GenAI telemetry specifications.",
        ),
        evaluation_tags=["observability", "opentelemetry", "metrics", "llm-drift"],
        description="Industry best practices across OpenTelemetry, Langfuse, and Arize.",
    ),
    EvaluationCase(
        id="eng_003",
        question="Are structured outputs guaranteed by inference engines without retry loops?",
        category=EvaluationCategory.AI_ENGINEERING,
        difficulty=DifficultyLevel.MODERATE,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            requires_primary_sources=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["grammar-constrained-sampling", "json-schema"],
            notes="Grammar-based engines guarantee schema grammar compliance.",
        ),
        evaluation_tags=["structured-outputs", "json-schema", "guidance", "outlines"],
        description="Examines grammar-constrained sampling in vLLM, SGLang, and Outlines.",
    ),
    EvaluationCase(
        id="eng_004",
        question="How does semantic caching reduce LLM inference costs and latency?",
        category=EvaluationCategory.AI_ENGINEERING,
        difficulty=DifficultyLevel.EASY,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=False,
            requires_primary_sources=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["similarity-thresholds", "cache-hits"],
            notes="Standard engineering pattern documentation.",
        ),
        evaluation_tags=["semantic-cache", "redis", "gptcache", "cost-reduction"],
        description="Standard architectural pattern query.",
    ),
    # 5. Models & Reasoning
    EvaluationCase(
        id="models_001",
        question="Test-Time Compute vs Pretraining: what is empirical return on reasoning tokens?",
        category=EvaluationCategory.MODELS_REASONING,
        difficulty=DifficultyLevel.CONTROVERSIAL,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            likely_disagreement=True,
            benchmark_oriented=True,
            requires_recent_evidence=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["scaling-laws", "diminishing-returns"],
            notes="Massive math gains vs open-domain reasoning plateau.",
        ),
        evaluation_tags=["test-time-compute", "scaling-laws", "o1", "r1", "reasoning"],
        description="Synthesizes scaling law revisions from OpenAI o1, DeepSeek-R1, and papers.",
    ),
    EvaluationCase(
        id="models_002",
        question="How does Mixture of Agents (MoA) compare with single frontier model reasoning?",
        category=EvaluationCategory.MODELS_REASONING,
        difficulty=DifficultyLevel.HARD,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            comparative=True,
            benchmark_oriented=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["alpaca-eval-gains", "latency-multiplier"],
            notes="MoA achieves higher benchmark scores but at 3-5x latency.",
        ),
        evaluation_tags=["mixture-of-agents", "moa", "ensemble", "alpaca-eval"],
        description="Examines layered model collaboration benchmarks against GPT-4o.",
    ),
    EvaluationCase(
        id="models_003",
        question="What is the parameter size of the Qwen 2.5 Coder flagship model?",
        category=EvaluationCategory.MODELS_REASONING,
        difficulty=DifficultyLevel.EASY,
        characteristics=ResearchCharacteristics(
            requires_primary_sources=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["32b-parameters", "model-specs"],
            notes="Exact official model specification fact (32B).",
        ),
        evaluation_tags=["qwen", "coding-models", "model-specs"],
        description="Factual model specification lookup.",
    ),
    EvaluationCase(
        id="models_004",
        question="Do reasoning models hallucinate less on mathematical reasoning tasks?",
        category=EvaluationCategory.MODELS_REASONING,
        difficulty=DifficultyLevel.MODERATE,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            benchmark_oriented=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["gsm8k-accuracy", "math-500", "verification"],
            notes="Chain-of-thought verification reduces calculation hallucinations.",
        ),
        evaluation_tags=["hallucination", "math-benchmarks", "gsm8k", "math-500"],
        description="Compares chain-of-thought verification against standard prompt generation.",
    ),
    # 6. Agent Protocols
    EvaluationCase(
        id="proto_001",
        question="A2A vs MCP: How do emerging Agent communication protocols compare?",
        category=EvaluationCategory.AGENT_PROTOCOLS,
        difficulty=DifficultyLevel.HARD,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            comparative=True,
            requires_recent_evidence=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["agent-negotiation-vs-tool-calling", "interoperability"],
            notes="MCP focuses on tool integration; A2A explores peer negotiation.",
        ),
        evaluation_tags=["a2a", "mcp", "agent-protocols", "interoperability"],
        description="Investigates transport, security, and message exchange formats.",
    ),
    EvaluationCase(
        id="proto_002",
        question="What transport mechanisms are supported by the MCP specification?",
        category=EvaluationCategory.AGENT_PROTOCOLS,
        difficulty=DifficultyLevel.EASY,
        characteristics=ResearchCharacteristics(
            requires_primary_sources=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["stdio", "sse"],
            notes="Specification fact: stdio and SSE.",
        ),
        evaluation_tags=["mcp-spec", "stdio", "sse", "transport"],
        description="Specification verification for stdio and Server-Sent Events (SSE).",
    ),
    EvaluationCase(
        id="proto_003",
        question="Is SSE or WebSockets preferred for long-running streaming agent tool execution?",
        category=EvaluationCategory.AGENT_PROTOCOLS,
        difficulty=DifficultyLevel.MODERATE,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            comparative=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["unidirectional-vs-bidirectional", "firewall-traversal"],
            notes="SSE offers HTTP/2 multiplexing; WebSockets allows full duplex signals.",
        ),
        evaluation_tags=["sse", "websockets", "streaming", "agent-ux"],
        description="Tradeoff analysis between unidirectional SSE and full-duplex WebSockets.",
    ),
    EvaluationCase(
        id="proto_004",
        question="How do authorization standards address delegation risks in multi-tool envs?",
        category=EvaluationCategory.AGENT_PROTOCOLS,
        difficulty=DifficultyLevel.HARD,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            requires_primary_sources=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["oauth-scoping", "least-privilege"],
            notes="Consensus around OAuth token exchange and proof-of-possession.",
        ),
        evaluation_tags=["oauth", "delegation", "agent-security", "least-privilege"],
        description="Evaluates cryptographic tokens, scoped keys, and human-in-the-loop controls.",
    ),
    # 7. Evaluation / Benchmarks
    EvaluationCase(
        id="bench_001",
        question="Does SWE-bench Verified address the issue of noisy test assertions?",
        category=EvaluationCategory.EVALUATION_BENCHMARKS,
        difficulty=DifficultyLevel.MODERATE,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            requires_primary_sources=True,
            benchmark_oriented=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["human-annotation-validity", "residual-test-noise"],
            notes="OpenAI review fixed tests, but researchers report task ambiguity.",
        ),
        evaluation_tags=["swe-bench", "benchmark-noise", "openai", "evaluation"],
        description="Evaluates human verification of SWE-bench test suites.",
    ),
    EvaluationCase(
        id="bench_002",
        question="Is LLM-as-a-Judge susceptible to position and length bias in evaluations?",
        category=EvaluationCategory.EVALUATION_BENCHMARKS,
        difficulty=DifficultyLevel.HARD,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            likely_disagreement=True,
            benchmark_oriented=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["position-bias", "verbosity-bias", "self-enhancement"],
            notes="Literature consensus that uncalibrated judges have strong bias.",
        ),
        evaluation_tags=["llm-as-judge", "bias", "mt-bench", "chatarena"],
        description="Analyzes self-enhancement, verbosity, and position bias across judge models.",
    ),
    EvaluationCase(
        id="bench_003",
        question="What benchmark is standard for evaluating agent web-browsing capabilities?",
        category=EvaluationCategory.EVALUATION_BENCHMARKS,
        difficulty=DifficultyLevel.EASY,
        characteristics=ResearchCharacteristics(
            requires_primary_sources=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["webarena", "mind2web"],
            notes="Established literature standard.",
        ),
        evaluation_tags=["webarena", "mind2web", "browsing-benchmarks"],
        description="Identifies standard web navigation benchmarks (WebArena, Mind2Web).",
    ),
    EvaluationCase(
        id="bench_004",
        question="Are synthetic benchmarks good predictors of enterprise RAG performance?",
        category=EvaluationCategory.EVALUATION_BENCHMARKS,
        difficulty=DifficultyLevel.CONTROVERSIAL,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            likely_disagreement=True,
            benchmark_oriented=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["synthetic-question-distribution", "query-drift"],
            notes="Synthetic benchmarks are cheap, but production queries have high noise.",
        ),
        evaluation_tags=["synthetic-data", "rag-benchmarks", "ragas", "real-world-drift"],
        description="Synthesizes academic RAGAS validation against production failure cases.",
    ),
    EvaluationCase(
        id="bench_005",
        question="How does GAIA benchmark evaluate multimodal multi-step tool use?",
        category=EvaluationCategory.EVALUATION_BENCHMARKS,
        difficulty=DifficultyLevel.MODERATE,
        characteristics=ResearchCharacteristics(
            requires_primary_sources=True,
            benchmark_oriented=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=False,
            critical_topics=["three-tier-difficulty", "multimodal-qa"],
            notes="Design description fact from original GAIA benchmark paper.",
        ),
        evaluation_tags=["gaia", "multimodal", "general-assistants"],
        description="Analyzes GAIA level 1-3 task design and evaluation methodology.",
    ),
    EvaluationCase(
        id="bench_006",
        question="Do automated code review benchmarks reflect real-world bug detection accuracy?",
        category=EvaluationCategory.EVALUATION_BENCHMARKS,
        difficulty=DifficultyLevel.HARD,
        characteristics=ResearchCharacteristics(
            requires_multiple_sources=True,
            likely_disagreement=True,
            benchmark_oriented=True,
        ),
        expected_disagreement=ExpectedDisagreement(
            expected=True,
            critical_topics=["synthetic-injected-defects", "architectural-bugs"],
            notes="Benchmarks check local syntax; real reviews check state interactions.",
        ),
        evaluation_tags=["code-review", "defect-detection", "benchmark-validity"],
        description="Assesses synthetic injected bugs vs empirical code review studies.",
    ),
]

# Curated 8-Case Live Pilot Subset (2 Easy, 2 Moderate, 2 Hard, 2 Controversial)
LIVE_PILOT_CASE_IDS = (
    "models_003",  # EASY
    "rag_004",  # EASY
    "local_002",  # MODERATE
    "bench_001",  # MODERATE
    "agents_005",  # HARD
    "proto_001",  # HARD
    "agents_003",  # CONTROVERSIAL
    "models_001",  # CONTROVERSIAL
)


def load_default_dataset() -> EvaluationDataset:
    """Return the built-in versioned evaluation dataset."""
    return EvaluationDataset(
        schema_version="1.1.0",
        created_at="2026-08-28T00:00:00Z",
        dataset_name="autonomous_ai_research_radar_eval_v1",
        cases=DEFAULT_EVALUATION_CASES,
    )


def load_live_pilot_dataset() -> EvaluationDataset:
    """Return the 8-case representative live pilot dataset."""
    pilot_cases = [c for c in DEFAULT_EVALUATION_CASES if c.id in LIVE_PILOT_CASE_IDS]
    return EvaluationDataset(
        schema_version="1.1.0",
        created_at="2026-08-28T00:00:00Z",
        dataset_name="autonomous_ai_research_radar_live_pilot_v1",
        cases=pilot_cases,
    )


def validate_dataset(dataset: EvaluationDataset) -> list[str]:
    """Validate dataset integrity and return list of validation errors (empty if valid)."""
    errors: list[str] = []
    if not dataset.cases:
        errors.append("Dataset contains no evaluation cases.")

    seen_ids: set[str] = set()
    for case in dataset.cases:
        if case.id in seen_ids:
            errors.append(f"Duplicate case id: {case.id}")
        seen_ids.add(case.id)

        if not case.question.strip():
            errors.append(f"Case {case.id} has an empty question.")

        if not case.category:
            errors.append(f"Case {case.id} has an invalid category.")

        if not case.difficulty:
            errors.append(f"Case {case.id} has an invalid difficulty.")

    return errors


def save_dataset_to_file(dataset: EvaluationDataset, target_path: Path) -> None:
    """Save dataset to a JSON file."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(dataset.model_dump(), indent=2, ensure_ascii=False) + "\n")


def load_dataset_from_file(source_path: Path) -> EvaluationDataset:
    """Load dataset from a JSON file."""
    raw = json.loads(source_path.read_text())
    return EvaluationDataset.model_validate(raw)
