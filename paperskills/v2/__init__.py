"""PaperToSkill 2.0: scientific + operational evidence discovery.

The v2 package is intentionally separate from ``paperskills.library`` so the
current paper-only workflow can keep running unchanged while we build the
technical-documentation grounding layer.
"""

from paperskills.v2.orchestrator import PaperToSkillV2Builder
from paperskills.v2.models import (
    CandidatePackage,
    EvidenceBundle,
    EvidenceSource,
    ExtractedOperation,
    TaskIntent,
)

__all__ = [
    "CandidatePackage",
    "EvidenceBundle",
    "EvidenceSource",
    "ExtractedOperation",
    "PaperToSkillV2Builder",
    "TaskIntent",
]

