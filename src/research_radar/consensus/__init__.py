"""P1B Consensus & Contradiction Intelligence package."""

from research_radar.consensus.clustering import ClaimClusterer
from research_radar.consensus.contradiction import ContradictionDetector
from research_radar.consensus.engine import ConsensusEngine
from research_radar.consensus.extractor import ClaimExtractor
from research_radar.consensus.models import (
    ClaimCluster,
    ClaimStance,
    ConsensusAssessment,
    ConsensusLevel,
    ConsensusReport,
    ContradictionRecord,
    ContradictionSeverity,
    ContradictionType,
    EvidenceClaim,
    NormalizedProposition,
)
from research_radar.consensus.normalizer import PropositionNormalizer
from research_radar.consensus.synthesis import ConsensusSynthesizer

__all__ = [
    "ClaimCluster",
    "ClaimClusterer",
    "ClaimExtractor",
    "ClaimStance",
    "ConsensusAssessment",
    "ConsensusEngine",
    "ConsensusLevel",
    "ConsensusReport",
    "ConsensusSynthesizer",
    "ContradictionDetector",
    "ContradictionRecord",
    "ContradictionSeverity",
    "ContradictionType",
    "EvidenceClaim",
    "NormalizedProposition",
    "PropositionNormalizer",
]
