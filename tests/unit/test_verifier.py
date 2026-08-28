"""Comprehensive tests for Claim Verification and Fact vs Interpretation handling."""

from __future__ import annotations

from research_radar.evidence.models import (
    ClaimType,
    StructuredClaim,
    VerificationStatus,
)
from research_radar.research.verifier import ClaimVerifier


def test_verifier_supported_fact_claim() -> None:
    verifier = ClaimVerifier()
    claims = [
        StructuredClaim(
            text="Gemma 3 supports multimodal reasoning",
            claim_type=ClaimType.FACT,
            evidence_ids=["arxiv-1", "github-1"],
            confidence=0.9,
        )
    ]
    context = {"arxiv-1", "github-1", "news-1"}
    report = verifier.verify(claims, context)

    assert report.total_claims == 1
    assert report.supported_claims == 1
    assert report.unsupported_claims == 0
    assert report.all_supported is True
    assert report.citation_coverage == 1.0


def test_verifier_unsupported_fact_claim_without_evidence() -> None:
    verifier = ClaimVerifier()
    claims = [
        StructuredClaim(
            text="Model achieves 99.9% accuracy on human tests",
            claim_type=ClaimType.FACT,
            evidence_ids=[],
            confidence=0.85,
        )
    ]
    context = {"arxiv-1"}
    report = verifier.verify(claims, context)

    assert report.total_claims == 1
    assert report.unsupported_claims == 1
    assert report.supported_claims == 0
    assert report.all_supported is False
    assert report.citation_coverage == 0.0


def test_verifier_unsupported_fact_with_invalid_evidence_ids() -> None:
    verifier = ClaimVerifier()
    claims = [
        StructuredClaim(
            text="New framework uses Rust backend",
            claim_type=ClaimType.FACT,
            evidence_ids=["non-existent-id"],
            confidence=0.7,
        )
    ]
    context = {"arxiv-1", "github-1"}
    report = verifier.verify(claims, context)

    assert report.unsupported_claims == 1
    assert report.supported_claims == 0


def test_verifier_interpretation_claim_graceful_handling() -> None:
    verifier = ClaimVerifier()
    claims = [
        StructuredClaim(
            text="This architecture might reduce server latency by 20%",
            claim_type=ClaimType.INTERPRETATION,
            evidence_ids=[],  # Interpretation without direct reference is partially supported
            confidence=0.5,
        ),
        StructuredClaim(
            text="Adoption could accelerate in enterprise deployments",
            claim_type=ClaimType.INTERPRETATION,
            evidence_ids=["github-1"],
            confidence=0.6,
        ),
    ]
    context = {"github-1"}
    report = verifier.verify(claims, context)

    assert report.total_claims == 2
    assert report.supported_claims == 1  # The one with valid context evidence
    assert report.partially_supported_claims == 1  # The one without evidence
    assert report.unsupported_claims == 0  # Interpretations are not flagged as hard unsupported


def test_downgrade_unsupported_factual_claims() -> None:
    verifier = ClaimVerifier()
    claims = [
        StructuredClaim(
            text="Benchmark beats SOTA by 50%",
            claim_type=ClaimType.FACT,
            evidence_ids=[],
            confidence=0.9,
        ),
        StructuredClaim(
            text="Repository has MIT license",
            claim_type=ClaimType.FACT,
            evidence_ids=["github-1"],
            confidence=0.95,
        ),
    ]
    context = {"github-1"}
    report = verifier.verify(claims, context)

    downgraded = verifier.downgrade_unsupported_claims(claims, report)

    # First claim (unsupported fact) should be downgraded to interpretation
    assert downgraded[0].claim_type == ClaimType.INTERPRETATION
    assert downgraded[0].confidence <= 0.5
    assert downgraded[0].verification_status == VerificationStatus.UNSUPPORTED

    # Second claim (supported fact) remains a FACT
    assert downgraded[1].claim_type == ClaimType.FACT
    assert downgraded[1].verification_status == VerificationStatus.SUPPORTED


def test_adversarial_case_a_evidence_exists_but_content_does_not_support_claim() -> None:
    verifier = ClaimVerifier()
    # Claim asserts vLLM is used for inference
    claim = StructuredClaim(
        text="Repository uses vLLM for high-throughput inference",
        claim_type=ClaimType.FACT,
        evidence_ids=["github-1"],
        confidence=0.9,
    )
    # Evidence text only talks about basic Python utilities without any mention of vLLM
    content_map = {
        "github-1": "A Python repository for general AI utility functions and file helpers."
    }
    report = verifier.verify([claim], {"github-1"}, content_map)

    # Must be UNSUPPORTED despite evidence_id existing!
    assert report.total_claims == 1
    assert report.unsupported_claims == 1
    assert report.supported_claims == 0
    assert report.results[0].status == VerificationStatus.UNSUPPORTED


def test_adversarial_case_b_direct_support() -> None:
    verifier = ClaimVerifier()
    claim = StructuredClaim(
        text="Repository uses vLLM for high-throughput inference",
        claim_type=ClaimType.FACT,
        evidence_ids=["github-1"],
        confidence=0.9,
    )
    content_map = {
        "github-1": "This system integrates vLLM backend for high-throughput GPU model inference."
    }
    report = verifier.verify([claim], {"github-1"}, content_map)

    assert report.supported_claims == 1
    assert report.unsupported_claims == 0
    assert report.results[0].status == VerificationStatus.SUPPORTED


def test_adversarial_case_c_partial_support() -> None:
    verifier = ClaimVerifier()
    # Claim makes two statements: distributed training and FP8 quantization
    claim = StructuredClaim(
        text="Framework supports distributed multi-node training and FP8 quantization",
        claim_type=ClaimType.FACT,
        evidence_ids=["github-1"],
        confidence=0.8,
    )
    # Evidence only mentions distributed training, nothing about FP8 quantization
    content_map = {"github-1": "Features: distributed multi-node cluster training with PyTorch."}
    report = verifier.verify([claim], {"github-1"}, content_map)

    assert report.partially_supported_claims == 1
    assert report.results[0].status == VerificationStatus.PARTIALLY_SUPPORTED


def test_adversarial_case_e_interpretation_remains_labeled_as_inference() -> None:
    verifier = ClaimVerifier()
    claim = StructuredClaim(
        text="Inferensi: Adopsi framework ini dapat meningkatkan produktivitas tim riset",
        claim_type=ClaimType.INTERPRETATION,
        evidence_ids=["github-1"],
        confidence=0.6,
    )
    content_map = {"github-1": "A Python framework for agent orchestration."}
    report = verifier.verify([claim], {"github-1"}, content_map)

    assert report.supported_claims == 1
    assert report.results[0].status == VerificationStatus.SUPPORTED
    assert report.results[0].claim.claim_type == ClaimType.INTERPRETATION


def test_conflicting_contradiction_detection() -> None:
    verifier = ClaimVerifier()
    # Claim asserts support for CUDA
    claim = StructuredClaim(
        text="Library supports CUDA acceleration on Nvidia GPUs",
        claim_type=ClaimType.FACT,
        evidence_ids=["github-1"],
        confidence=0.85,
    )
    # Evidence text explicitly says it does not support CUDA
    content_map = {"github-1": "Notice: This release does not support CUDA hardware acceleration."}
    report = verifier.verify([claim], {"github-1"}, content_map)

    assert report.conflicting_claims == 1
    assert report.results[0].status == VerificationStatus.CONFLICTING
