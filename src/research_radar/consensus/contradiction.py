"""Cross-evidence contradiction detection and temporal/version reasoning."""

from __future__ import annotations

import logging
import re
from uuid import uuid4

from research_radar.consensus.models import (
    ClaimCluster,
    ClaimStance,
    ContradictionRecord,
    ContradictionSeverity,
    ContradictionType,
    EvidenceClaim,
)

logger = logging.getLogger(__name__)

_EXPLICIT_VERSION_PATTERN = re.compile(
    r"\b(?:v|version\s*)(\d+(?:\.\d+)*)\b|\b(\d+\.\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"\b(202[0-9])\b")


_RELEASE_CHANGE_TERMS = {
    "added",
    "introduced",
    "now supported",
    "starting in",
    "as of version",
    "as of v",
    "since version",
    "since v",
    "support added",
    "release notes",
    "changelog",
    "ditambahkan",
    "sejak versi",
    "mulai versi",
}


def _parse_memory_mb(val: str) -> float | None:
    """Parse memory string into megabytes for comparable numerical checks."""
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(gb|mb|tb|g|m|t)?$", val.strip().lower())
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2) or "gb"
    if unit in ("tb", "t"):
        return num * 1024 * 1024
    if unit in ("gb", "g"):
        return num * 1024
    if unit in ("mb", "m"):
        return num
    return None


def _parse_percentage(val: str) -> float | None:
    """Parse percentage or decimal proportion."""
    s = val.strip().lower()
    if s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except ValueError:
            return None
    try:
        f = float(s)
        if 0.0 <= f <= 1.0:
            return f
    except ValueError:
        pass
    return None


def _extract_scope(text: str) -> set[str]:
    """Extract operational scopes (inference, training, finetuning, min, max, rec)."""
    t = text.lower()
    scopes: set[str] = set()
    if "inference" in t or "inferensi" in t:
        scopes.add("inference")
    if "training" in t or "train" in t or "pelatihan" in t:
        scopes.add("training")
    if "fine-tuning" in t or "finetuning" in t:
        scopes.add("finetuning")

    if "minimum" in t or "minimal" in t or re.search(r"\bmin\b", t):
        scopes.add("minimum")
    if "recommended" in t or "rekomendasi" in t or "optimal" in t:
        scopes.add("recommended")
    if "peak" in t or "maksimal" in t or "maximum" in t:
        scopes.add("maximum")

    return scopes


class ContradictionDetector:
    """Detects cross-evidence contradictions and resolves temporal conflicts."""

    def detect_contradictions(
        self,
        clusters: list[ClaimCluster],
    ) -> list[ContradictionRecord]:
        """Detect all pairwise contradictions across claim clusters."""
        all_contradictions: list[ContradictionRecord] = []

        for cluster in clusters:
            claims = cluster.claims
            if len(claims) < 2:
                continue

            for i in range(len(claims)):
                for j in range(i + 1, len(claims)):
                    c_record = self._compare_claims(claims[i], claims[j])
                    if c_record:
                        all_contradictions.append(c_record)

        logger.info(
            "Completed contradiction detection",
            extra={
                "event": "contradiction_detection_completed",
                "contradictions_found": len(all_contradictions),
                "unresolved_count": sum(1 for c in all_contradictions if not c.resolved_by_version),
            },
        )
        return all_contradictions

    def _compare_claims(
        self,
        a: EvidenceClaim,
        b: EvidenceClaim,
    ) -> ContradictionRecord | None:
        """Compare two claims for potential direct, numeric, version, or qualification conflicts."""
        # 0. Check Subject Compatibility: claims about distinct subjects cannot contradict
        subj_a = a.proposition.subject.lower().strip()
        subj_b = b.proposition.subject.lower().strip()
        if subj_a and subj_b and subj_a != subj_b and subj_a != "general" and subj_b != "general":
            return None

        # 0b. Check Platform Compatibility: different platforms (windows vs linux) do not contradict
        if (
            a.proposition.predicate == "platform_support"
            and b.proposition.predicate == "platform_support"
        ):
            obj_a = (a.proposition.object_val or "").lower().strip()
            obj_b = (b.proposition.object_val or "").lower().strip()
            if obj_a and obj_b and obj_a != obj_b:
                return None

        # 1. Check if one claim is explicitly newer (temporal / version evolution)
        is_version_diff, newer_claim, older_claim = self._check_version_evolution(a, b)

        # 2. Direct Negation: SUPPORTS vs OPPOSES
        if (a.stance == ClaimStance.SUPPORTS and b.stance == ClaimStance.OPPOSES) or (
            a.stance == ClaimStance.OPPOSES and b.stance == ClaimStance.SUPPORTS
        ):
            if is_version_diff and newer_claim and older_claim:
                old_ver = older_claim.version_tag or older_claim.published_at or "rilis lama"
                new_ver = newer_claim.version_tag or newer_claim.published_at or "rilis baru"
                return ContradictionRecord(
                    contradiction_id=uuid4().hex,
                    proposition=a.proposition,
                    claim_a=a,
                    claim_b=b,
                    severity=ContradictionSeverity.LOW,
                    contradiction_type=ContradictionType.VERSION_CONFLICT,
                    explanation=(
                        f"Perbedaan status antara sumber lama ({old_ver}) "
                        f"dan rilis baru ({new_ver})."
                    ),
                    resolved_by_version=True,
                    resolution_notes=(
                        f"Diselesaikan oleh pembaruan versi/rilis: {newer_claim.text[:100]}"
                    ),
                )
            else:
                severity = (
                    ContradictionSeverity.HIGH
                    if a.proposition.predicate
                    in ("production_readiness", "platform_support", "vram_requirement")
                    else ContradictionSeverity.MEDIUM
                )
                return ContradictionRecord(
                    contradiction_id=uuid4().hex,
                    proposition=a.proposition,
                    claim_a=a,
                    claim_b=b,
                    severity=severity,
                    contradiction_type=ContradictionType.DIRECT_NEGATION,
                    explanation=f"Pertentangan langsung: '{a.text[:80]}' vs '{b.text[:80]}'.",
                    resolved_by_version=False,
                )

        # 3. Numeric Conflict: same predicate but different values (e.g. 8 GB vs 16 GB)
        val_a = a.proposition.object_val
        val_b = b.proposition.object_val
        if val_a and val_b and a.proposition.predicate == b.proposition.predicate:
            # Check unit equivalence
            mb_a = _parse_memory_mb(val_a)
            mb_b = _parse_memory_mb(val_b)
            if mb_a is not None and mb_b is not None:
                if abs(mb_a - mb_b) < 1.0:
                    # Semantically equivalent (e.g. 8 GB == 8192 MB) -> No conflict
                    return None
            else:
                pct_a = _parse_percentage(val_a)
                pct_b = _parse_percentage(val_b)
                if pct_a is not None and pct_b is not None:
                    if abs(pct_a - pct_b) < 0.001:
                        # Semantically equivalent (e.g. 20% == 0.20) -> No conflict
                        return None
                elif val_a.lower().strip() == val_b.lower().strip():
                    return None

            # Check Scope Disjointness (e.g. inference vs training, minimum vs recommended)
            scope_a = _extract_scope(a.text)
            scope_b = _extract_scope(b.text)
            # If one is explicitly inference and other is training, no conflict
            if ("inference" in scope_a and "training" in scope_b) or (
                "training" in scope_a and "inference" in scope_b
            ):
                return None
            # If one is minimum and other is recommended/peak, no conflict
            if ("minimum" in scope_a and "recommended" in scope_b) or (
                "recommended" in scope_a and "minimum" in scope_b
            ):
                return None

            if is_version_diff:
                return ContradictionRecord(
                    contradiction_id=uuid4().hex,
                    proposition=a.proposition,
                    claim_a=a,
                    claim_b=b,
                    severity=ContradictionSeverity.LOW,
                    contradiction_type=ContradictionType.NUMERIC_CONFLICT,
                    explanation=(f"Perbedaan nilai ({val_a} vs {val_b}) dijelaskan evolusi versi."),
                    resolved_by_version=True,
                    resolution_notes="Diselesaikan oleh versi rilis terbaru.",
                )
            else:
                return ContradictionRecord(
                    contradiction_id=uuid4().hex,
                    proposition=a.proposition,
                    claim_a=a,
                    claim_b=b,
                    severity=(
                        ContradictionSeverity.HIGH
                        if "requirement" in a.proposition.predicate
                        else ContradictionSeverity.MEDIUM
                    ),
                    contradiction_type=ContradictionType.NUMERIC_CONFLICT,
                    explanation=(
                        f"Konflik numerik pada {a.proposition.predicate}: {val_a} vs {val_b}."
                    ),
                    resolved_by_version=False,
                )

        # 4. Qualification Conflict: SUPPORTS vs QUALIFIES
        if (a.stance == ClaimStance.SUPPORTS and b.stance == ClaimStance.QUALIFIES) or (
            a.stance == ClaimStance.QUALIFIES and b.stance == ClaimStance.SUPPORTS
        ):
            qual_claim = a if a.stance == ClaimStance.QUALIFIES else b
            return ContradictionRecord(
                contradiction_id=uuid4().hex,
                proposition=a.proposition,
                claim_a=a,
                claim_b=b,
                severity=ContradictionSeverity.MEDIUM,
                contradiction_type=ContradictionType.QUALIFICATION,
                explanation=(
                    f"Klaim berkualifikasi: klaim penuh dibatasi oleh '{qual_claim.text[:90]}'."
                ),
                resolved_by_version=False,
            )

        return None

    def _check_version_evolution(
        self,
        a: EvidenceClaim,
        b: EvidenceClaim,
    ) -> tuple[bool, EvidenceClaim | None, EvidenceClaim | None]:
        """Check if one claim represents a genuine version evolution resolving the previous state.

        Version evolution requires:
        1. Explicit version ordering (e.g. v2 > v1, 2.0 > 1.0) OR
        2. Explicit temporal ordering WITH release/change language in the newer claim.
        Does NOT automatically resolve raw benchmark measurements across different dates.
        """
        # Extract explicit versions
        tag_a = a.version_tag
        if not tag_a:
            matches_a = _EXPLICIT_VERSION_PATTERN.findall(a.text)
            if matches_a:
                tag_a = matches_a[0][0] or matches_a[0][1]

        tag_b = b.version_tag
        if not tag_b:
            matches_b = _EXPLICIT_VERSION_PATTERN.findall(b.text)
            if matches_b:
                tag_b = matches_b[0][0] or matches_b[0][1]

        if tag_a and tag_b and tag_a.lower().strip() != tag_b.lower().strip():
            try:
                va = float(tag_a.lower().lstrip("v"))
                vb = float(tag_b.lower().lstrip("v"))
                if va > vb:
                    return True, a, b
                elif vb > va:
                    return True, b, a
            except ValueError:
                pass

        # Check temporal ordering + explicit release/change language
        newer: EvidenceClaim | None = None
        older: EvidenceClaim | None = None

        if a.published_at and b.published_at and a.published_at != b.published_at:
            if a.published_at > b.published_at:
                newer, older = a, b
            else:
                newer, older = b, a
        else:
            years_a = _YEAR_PATTERN.findall(a.text)
            years_b = _YEAR_PATTERN.findall(b.text)
            if years_a and years_b and years_a[0] != years_b[0]:
                try:
                    ya = int(years_a[0])
                    yb = int(years_b[0])
                    if ya > yb:
                        newer, older = a, b
                    elif yb > ya:
                        newer, older = b, a
                except ValueError:
                    pass

        if newer and older:
            newer_lower = newer.text.lower()
            has_change_term = any(term in newer_lower for term in _RELEASE_CHANGE_TERMS)
            has_version = bool(newer.version_tag) or bool(
                _EXPLICIT_VERSION_PATTERN.findall(newer.text)
            )
            if has_change_term or (
                has_version
                and newer.proposition.predicate in ("platform_support", "multimodal_support")
            ):
                return True, newer, older

        return False, None, None
