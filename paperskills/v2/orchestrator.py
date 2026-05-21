"""PaperToSkill 2.0 orchestration layer."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from paperskills.v2.domain_router import DomainRouter
from paperskills.v2.models import BuildResult, EvidenceBundle, EvidenceSource, ExtractedOperation
from paperskills.v2.skill_renderer import render_skill_markdown
from paperskills.v2.technical_docs import TechnicalDocFetcher, TechnicalDocPlanner


class PaperToSkillV2Builder:
    """Build a v2 evidence plan and operational skill draft from task text.

    This first version is a deterministic scaffold. It gives the runner and
    agent a structured target for technical documentation discovery before we
    plug in live fetching and LLM extraction.
    """

    def __init__(
        self,
        router: Optional[DomainRouter] = None,
        doc_planner: Optional[TechnicalDocPlanner] = None,
    ) -> None:
        self.router = router or DomainRouter()
        self.doc_planner = doc_planner or TechnicalDocPlanner()

    def build(
        self,
        task_text: str,
        task_id: str = "",
        error_messages: Optional[Iterable[str]] = None,
        fetch_technical_docs: bool = False,
        cache_dir: Optional[Path] = None,
    ) -> BuildResult:
        intent = self.router.parse_intent(task_text, task_id=task_id)
        packages = self.router.route(task_text)
        technical_sources = self.doc_planner.plan_many(packages, intent=intent)
        if fetch_technical_docs:
            if cache_dir is None:
                raise ValueError("cache_dir is required when fetch_technical_docs=True")
            technical_sources = TechnicalDocFetcher(cache_dir).fetch_many(technical_sources)
        scientific_sources = self._plan_scientific_sources(intent, packages)
        debug_sources = self._plan_debug_sources(packages, error_messages or [])
        operations = self._seed_operations(packages)

        bundle = EvidenceBundle(
            task_intent=intent,
            candidate_packages=packages,
            scientific_sources=scientific_sources,
            technical_sources=technical_sources,
            debug_sources=debug_sources,
            extracted_operations=operations,
            skill_payload=self._skill_payload(packages),
            notes=[
                "PaperToSkill 2.0 separates scientific evidence from operational documentation.",
                "Use papers for method context and technical docs for package APIs, objects, parameters, and examples.",
            ],
        )
        return BuildResult(
            bundle=bundle,
            skill_markdown=render_skill_markdown(bundle),
            metadata={"builder": "PaperToSkillV2Builder", "version": "0.1"},
        )

    def _plan_scientific_sources(self, intent, packages) -> List[EvidenceSource]:
        package_terms = " ".join(package.package for package in packages[:3])
        query = f"{intent.analysis_intent} {package_terms} method software paper".strip()
        if not query:
            query = "bioinformatics method software paper"
        return [
            EvidenceSource(
                source_type="method/software paper search",
                title="Scientific method grounding search",
                evidence_role="conceptual grounding",
                query=query,
                useful_sections=["method rationale", "algorithm", "recommended use cases"],
            )
        ]

    def _plan_debug_sources(self, packages, error_messages: Iterable[str]) -> List[EvidenceSource]:
        debug_sources: List[EvidenceSource] = []
        for message in error_messages:
            for package in packages[:3]:
                debug_sources.extend(self.doc_planner.plan_debug_sources(package, message))
        return debug_sources

    def _seed_operations(self, packages) -> List[ExtractedOperation]:
        if not packages:
            return []
        primary = packages[0]
        operations: List[ExtractedOperation] = []
        if primary.package == "methylKit":
            operations = [
                ExtractedOperation(
                    step="Import methylation coverage files into methylKit objects.",
                    function_or_object="methRead",
                    required_arguments=["location", "sample.id", "assembly", "treatment", "pipeline"],
                    input_mapping="Bismark coverage files -> methylRawList",
                    output_mapping="saveRDS(methylRawList, ...)",
                    risks=["Multiple-file dispatch may require list inputs and treatment length matching sample count."],
                ),
                ExtractedOperation(
                    step="Apply coverage filtering and normalization when requested.",
                    function_or_object="filterByCoverage / normalizeCoverage",
                    required_arguments=["lo.count", "hi.perc"],
                    risks=["Coverage thresholds are workflow parameters; do not invent them when hidden."],
                ),
                ExtractedOperation(
                    step="Merge common CpG sites across samples.",
                    function_or_object="unite",
                    input_mapping="methylRawList -> methylBase",
                    output_mapping="getData(methylBase) or saveRDS(methylBase, ...)",
                    risks=["Object class and common-site semantics matter for evaluation."],
                ),
            ]
        elif primary.package in {"DESeq2", "edgeR", "limma"}:
            operations = [
                ExtractedOperation(
                    step="Construct the package-specific expression object from counts and sample metadata.",
                    function_or_object="DESeqDataSetFromMatrix / DGEList / voom",
                    required_arguments=["count matrix", "sample metadata", "design"],
                    risks=["Contrast direction and thresholds may be latent workflow constants."],
                ),
                ExtractedOperation(
                    step="Fit model and extract results with documented API.",
                    function_or_object="DESeq/results or edgeR/limma model functions",
                    output_mapping="coerce model result to task table",
                ),
            ]
        elif primary.package == "GenomicRanges":
            operations = [
                ExtractedOperation(
                    step="Import intervals as GRanges and harmonize coordinate conventions.",
                    function_or_object="rtracklayer::import / GenomicRanges::GRanges",
                    risks=["BED is 0-based while GRanges is 1-based; seqlevels must match."],
                ),
                ExtractedOperation(
                    step="Perform nearest/overlap operation with GenomicRanges.",
                    function_or_object="nearest / distanceToNearest / findOverlaps",
                    output_mapping="export annotated table or BED-like rows",
                ),
            ]
        elif primary.package == "clusterProfiler":
            operations = [
                ExtractedOperation(
                    step="Map gene identifiers to the required key type.",
                    function_or_object="bitr / OrgDb / keyType",
                    risks=["Wrong gene ID type is a common cause of empty enrichment results."],
                ),
                ExtractedOperation(
                    step="Run enrichment and export the result object.",
                    function_or_object="enrichGO / enrichKEGG / compareCluster",
                    required_arguments=["gene", "OrgDb", "keyType", "ont", "pAdjustMethod"],
                    output_mapping="as.data.frame(enrichResult)",
                ),
            ]
        return operations

    def _skill_payload(self, packages) -> dict:
        return {
            "packages": [package.package for package in packages],
            "object_classes": list(dict.fromkeys(cls for package in packages for cls in package.object_classes)),
            "functions": list(dict.fromkeys(fn for package in packages for fn in package.functions)),
            "evidence_roles": ["scientific_method", "technical_workflow", "function_signature", "debug_repair"],
        }
