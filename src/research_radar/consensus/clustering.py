"""In-memory proposition clustering for research claims."""

from __future__ import annotations

import logging
from collections import defaultdict
from uuid import uuid4

from research_radar.consensus.models import ClaimCluster, EvidenceClaim, NormalizedProposition

logger = logging.getLogger(__name__)


class ClaimClusterer:
    """Groups comparable claims by normalized proposition identity and topic."""

    @staticmethod
    def cluster_claims(claims: list[EvidenceClaim]) -> list[ClaimCluster]:
        """Cluster claims into proposition groups with compatible subject, predicate, and scope."""
        if not claims:
            return []

        # Group by normalized subject, predicate, and relevant object dimension
        grouped: dict[str, list[EvidenceClaim]] = defaultdict(list)
        representative_prop: dict[str, NormalizedProposition] = {}

        for claim in claims:
            prop = claim.proposition
            subj_key = prop.subject.lower().strip() or "general"
            pred_key = prop.predicate.lower().strip() or "general_claim"

            # Separate platforms (e.g. windows vs linux) into distinct proposition clusters
            if pred_key == "platform_support" and prop.object_val:
                obj_key = prop.object_val.lower().strip()
                group_key = f"{subj_key}::{pred_key}::{obj_key}"
            elif pred_key in ("general_claim", ""):
                group_key = f"{subj_key}::general"
            else:
                group_key = f"{subj_key}::{pred_key}"

            grouped[group_key].append(claim)
            if group_key not in representative_prop:
                representative_prop[group_key] = prop

        clusters: list[ClaimCluster] = []
        for group_key, group_claims in grouped.items():
            cluster = ClaimCluster(
                cluster_id=uuid4().hex,
                canonical_topic=group_key,
                proposition=representative_prop[group_key],
                claims=group_claims,
            )
            clusters.append(cluster)

        logger.info(
            "Clustered evidence claims",
            extra={
                "event": "claim_clustering_completed",
                "total_claims": len(claims),
                "cluster_count": len(clusters),
            },
        )
        return clusters
