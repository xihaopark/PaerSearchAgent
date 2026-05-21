"""Data models for PaperToSkill 2.0 evidence planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskIntent:
    """Structured view of a scientific programming task."""

    task_id: str = ""
    domain: str = ""
    analysis_intent: str = ""
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    operation_keywords: List[str] = field(default_factory=list)
    package_hints: List[str] = field(default_factory=list)
    object_hints: List[str] = field(default_factory=list)
    risk_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidatePackage:
    """A routed package or method family candidate."""

    package: str
    ecosystem: str = "Bioconductor"
    reason: str = ""
    functions: List[str] = field(default_factory=list)
    object_classes: List[str] = field(default_factory=list)
    query_hints: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceSource:
    """A source that can support skill construction."""

    source_type: str
    title: str
    url: str = ""
    package: str = ""
    evidence_role: str = ""
    query: str = ""
    useful_sections: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    source_access_level: str = "planned"
    fetch_status: str = ""
    local_path: str = ""
    excerpt: str = ""
    content_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedOperation:
    """Executable operation distilled from evidence."""

    step: str
    function_or_object: str = ""
    required_arguments: List[str] = field(default_factory=list)
    input_mapping: str = ""
    output_mapping: str = ""
    risks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceBundle:
    """Combined scientific and operational evidence plan."""

    task_intent: TaskIntent
    candidate_packages: List[CandidatePackage] = field(default_factory=list)
    scientific_sources: List[EvidenceSource] = field(default_factory=list)
    technical_sources: List[EvidenceSource] = field(default_factory=list)
    debug_sources: List[EvidenceSource] = field(default_factory=list)
    extracted_operations: List[ExtractedOperation] = field(default_factory=list)
    skill_payload: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_intent": self.task_intent.to_dict(),
            "candidate_packages": [p.to_dict() for p in self.candidate_packages],
            "scientific_sources": [s.to_dict() for s in self.scientific_sources],
            "technical_sources": [s.to_dict() for s in self.technical_sources],
            "debug_sources": [s.to_dict() for s in self.debug_sources],
            "extracted_operations": [op.to_dict() for op in self.extracted_operations],
            "skill_payload": self.skill_payload,
            "notes": list(self.notes),
        }


@dataclass
class BuildResult:
    """PaperToSkill 2.0 build artifact."""

    bundle: EvidenceBundle
    skill_markdown: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle": self.bundle.to_dict(),
            "skill_markdown": self.skill_markdown,
            "metadata": self.metadata or {},
        }
