"""Deterministic proposition normalization and stance extraction."""

from __future__ import annotations

import re

from research_radar.consensus.models import ClaimStance, NormalizedProposition

_NEGATION_TERMS = {
    "not",
    "no",
    "never",
    "unsupported",
    "unsupported on",
    "fails",
    "failed",
    "failing",
    "degrades",
    "degraded",
    "unable",
    "cannot",
    "can't",
    "won't",
    "doesn't",
    "does not",
    "tidak",
    "bukan",
    "gagal",
    "belum",
    "tidak mendukung",
    "tanpa",
}

_QUALIFICATION_TERMS = {
    "only",
    "limited",
    "partially",
    "small workloads",
    "small-scale",
    "controlled",
    "depends",
    "conditional",
    "may be",
    "might",
    "caveat",
    "experimental",
    "beta",
    "hanya",
    "terbatas",
    "bergantung",
    "skala kecil",
}

_NUMERIC_PATTERN = re.compile(
    r"(\b\d+(?:\.\d+)?\s*(?:gb|mb|tb|%|tokens?/s|tps|ms|x|k|b)(?!\w)|\b\d+(?:\.\d+)?(?!\w))",
    re.IGNORECASE,
)


_QUESTION_TERMS = {
    "could",
    "would",
    "is it possible",
    "can it",
    "will it",
    "apakah",
    "mungkinkah",
    "bisakah",
}

_FUTURE_INTENTION_TERMS = {
    "hope to",
    "hopes to",
    "planning to",
    "plans to",
    "plan to",
    "planned for",
    "roadmap",
    "in future",
    "in the future",
    "target for",
    "will support soon",
    "support soon",
    "akan datang",
    "direncanakan",
    "target rilis",
}


class PropositionNormalizer:
    """Normalizes natural language statements into comparable propositions and stances."""

    @staticmethod
    def normalize_subject(raw_subject: str) -> str:
        """Normalize subject name to canonical form while preserving distinct entity identity."""
        s = raw_subject.strip().lower()
        for prefix in ("the ", "a ", "an ", "project "):
            if s.startswith(prefix):
                s = s[len(prefix) :].strip()
        for suffix in (
            " project",
            " runtime",
            " framework",
            " library",
            " model",
            " system",
            " tool",
        ):
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
        cleaned = re.sub(r"[^\w\s-]", "", s)
        cleaned = re.sub(r"[\s-]+", "_", cleaned).strip("_")
        return cleaned or "general"

    @staticmethod
    def normalize(
        text: str,
        *,
        default_subject: str = "general",
    ) -> tuple[NormalizedProposition, ClaimStance]:
        """Extract normalized proposition and stance from claim text."""
        cleaned = " ".join(text.strip().split())
        lower = cleaned.lower()

        # 1. Stance Detection
        stance = PropositionNormalizer.detect_stance(lower)

        # 2. Extract numeric values if present
        nums = _NUMERIC_PATTERN.findall(lower)
        object_val = nums[0] if nums else None

        # 3. Detect Predicate Category
        subject_raw = default_subject
        predicate = "general_claim"
        qualifier = None

        if any(
            w in lower
            for w in (
                "production",
                "enterprise",
                "mature",
                "deployable",
                "siap produksi",
                "workload",
            )
        ):
            predicate = "production_readiness"
        elif any(w in lower for w in ("vram", "gpu memory", "memory requirement")) or re.search(
            r"\bram\b", lower
        ):
            predicate = "vram_requirement"
        elif any(w in lower for w in ("windows", "linux", "macos", "cuda", "rocm", "metal")):
            predicate = "platform_support"
            for plat in ("windows", "linux", "macos", "cuda", "rocm", "metal"):
                if plat in lower:
                    object_val = plat
                    break
        elif any(
            w in lower
            for w in ("benchmark", "throughput", "latency", "tokens/s", "tps", "faster", "speed")
        ):
            predicate = "performance_benchmark"
        elif any(w in lower for w in ("multimodal", "vision", "audio", "text-only")):
            predicate = "multimodal_support"
        elif any(w in lower for w in ("accuracy", "score", "mmlu", "eval", "akurasi")):
            predicate = "evaluation_accuracy"

        # Check for operational scope and qualifiers
        scopes: list[str] = []
        if "inference" in lower or "inferensi" in lower:
            scopes.append("inference")
        elif "training" in lower or "train" in lower or "pelatihan" in lower:
            scopes.append("training")
        elif "fine-tuning" in lower or "finetuning" in lower:
            scopes.append("finetuning")

        if "minimum" in lower or "minimal" in lower or re.search(r"\bmin\b", lower):
            scopes.append("minimum")
        elif "recommended" in lower or "rekomendasi" in lower or "optimal" in lower:
            scopes.append("recommended")
        elif "peak" in lower or "maksimal" in lower or "maximum" in lower:
            scopes.append("maximum")

        for q in _QUALIFICATION_TERMS:
            if q in lower:
                scopes.append(q)
                break

        for f in _FUTURE_INTENTION_TERMS:
            if f in lower:
                scopes.append(f)
                break

        if scopes:
            qualifier = ":".join(scopes)

        # If subject was default, try extracting from starting words
        words = cleaned.split()
        if len(words) >= 2 and default_subject == "general":
            first_two = " ".join(words[:2])
            if not any(
                k in first_two.lower()
                for k in ("the", "this", "we", "in", "for", "based", "could", "would", "is")
            ):
                subject_raw = first_two

        normalized_subj = PropositionNormalizer.normalize_subject(subject_raw)

        prop = NormalizedProposition(
            subject=normalized_subj,
            predicate=predicate,
            object_val=object_val,
            qualifier=qualifier,
        )
        return prop, stance

    @staticmethod
    def detect_stance(text_lower: str) -> ClaimStance:
        """Deterministic stance detection."""
        # Check if text is a question or purely hypothetical
        if text_lower.endswith("?") or any(text_lower.startswith(q) for q in _QUESTION_TERMS):
            return ClaimStance.NEUTRAL

        # Check for future intention / roadmap
        if any(f in text_lower for f in _FUTURE_INTENTION_TERMS):
            return ClaimStance.QUALIFIES

        has_negation = any(
            re.search(rf"\b{re.escape(term)}\b", text_lower) for term in _NEGATION_TERMS
        )
        has_qualification = any(
            re.search(rf"\b{re.escape(term)}\b", text_lower) for term in _QUALIFICATION_TERMS
        )

        if has_negation:
            return ClaimStance.OPPOSES
        if has_qualification:
            return ClaimStance.QUALIFIES
        return ClaimStance.SUPPORTS
