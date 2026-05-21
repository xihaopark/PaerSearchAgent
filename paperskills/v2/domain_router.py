"""Domain-aware routing from task text to R/Bioconductor technical space."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List

from paperskills.v2.models import CandidatePackage, TaskIntent


@dataclass(frozen=True)
class RouteRule:
    name: str
    patterns: List[str]
    domain: str
    analysis_intent: str
    packages: List[CandidatePackage] = field(default_factory=list)
    input_hints: List[str] = field(default_factory=list)
    output_hints: List[str] = field(default_factory=list)
    operation_keywords: List[str] = field(default_factory=list)
    object_hints: List[str] = field(default_factory=list)
    risk_notes: List[str] = field(default_factory=list)

    def matches(self, text: str) -> bool:
        return any(re.search(pattern, text, re.I) for pattern in self.patterns)


def _pkg(
    name: str,
    reason: str,
    functions: Iterable[str],
    object_classes: Iterable[str] = (),
    query_hints: Iterable[str] = (),
    confidence: float = 0.7,
) -> CandidatePackage:
    return CandidatePackage(
        package=name,
        reason=reason,
        functions=list(functions),
        object_classes=list(object_classes),
        query_hints=list(query_hints),
        confidence=confidence,
    )


ROUTE_RULES: List[RouteRule] = [
    RouteRule(
        name="methylkit",
        patterns=[r"methyl", r"bismark", r"cpg", r"bisulfite"],
        domain="methylation",
        analysis_intent="DNA methylation processing",
        packages=[
            _pkg(
                "methylKit",
                "Canonical Bioconductor package for Bismark coverage import, filtering, normalization, and CpG merging.",
                ["methRead", "filterByCoverage", "normalizeCoverage", "unite", "getData", "percMethylation"],
                ["methylRaw", "methylRawList", "methylBase", "methylBaseDB"],
                ["methylKit methRead bismarkCoverage vignette", "methylKit unite methylRawList common CpG sites"],
                0.95,
            ),
            _pkg(
                "bsseq",
                "Alternative Bioconductor methylation object ecosystem for bisulfite sequencing.",
                ["BSseq", "read.bismark", "getCoverage", "getMeth"],
                ["BSseq", "GRanges"],
                ["bsseq Bioconductor vignette bisulfite methylation"],
                0.45,
            ),
        ],
        input_hints=["Bismark coverage", "methylated/unmethylated counts", "CpG loci"],
        output_hints=["methylRawList", "methylBase", "methylation table"],
        operation_keywords=["import", "coverage filter", "normalize", "unite", "percent methylation"],
        object_hints=["methylRawList", "methylBase"],
        risk_notes=["methylKit has S4 method dispatch; list vs character arguments matter."],
    ),
    RouteRule(
        name="differential_expression",
        patterns=[r"differential expression", r"\bdeseq2\b", r"\bedger\b", r"\blimma\b", r"\bvoom\b", r"count matrix"],
        domain="rna",
        analysis_intent="RNA-seq differential expression",
        packages=[
            _pkg(
                "DESeq2",
                "Canonical negative-binomial RNA-seq differential expression workflow.",
                ["DESeqDataSetFromMatrix", "DESeq", "results", "lfcShrink", "counts"],
                ["DESeqDataSet", "DESeqResults"],
                ["DESeq2 vignette results contrast lfcShrink"],
                0.85,
            ),
            _pkg(
                "edgeR",
                "Canonical count normalization and GLM/exact-test workflow for RNA-seq.",
                ["DGEList", "calcNormFactors", "filterByExpr", "estimateDisp", "glmQLFit", "glmQLFTest"],
                ["DGEList", "DGEGLM"],
                ["edgeR users guide TMM filterByExpr glmQLFTest"],
                0.75,
            ),
            _pkg(
                "limma",
                "Canonical voom/linear-model workflow for RNA-seq and expression matrices.",
                ["voom", "lmFit", "makeContrasts", "contrasts.fit", "eBayes", "topTable"],
                ["EList", "MArrayLM"],
                ["limma voom users guide topTable"],
                0.75,
            ),
        ],
        input_hints=["count matrix", "sample metadata", "design matrix"],
        output_hints=["results table", "normalized counts", "contrast table"],
        operation_keywords=["normalize", "design", "contrast", "test", "adjust p-values"],
        object_hints=["DESeqDataSet", "DGEList", "EList", "MArrayLM"],
        risk_notes=["Prompt often under-specifies contrasts and thresholds; separate method errors from latent workflow constants."],
    ),
    RouteRule(
        name="genomic_ranges",
        patterns=[r"\bbed\b", r"genomic range", r"overlap", r"nearest gene", r"annotat(e|ion)", r"peak"],
        domain="genomic_intervals",
        analysis_intent="Genomic interval annotation or overlap",
        packages=[
            _pkg(
                "GenomicRanges",
                "Core Bioconductor package for interval objects, overlaps, nearest features, and genomic joins.",
                ["GRanges", "findOverlaps", "nearest", "distanceToNearest", "seqlevelsStyle"],
                ["GRanges", "Hits"],
                ["GenomicRanges nearest findOverlaps vignette"],
                0.85,
            ),
            _pkg(
                "rtracklayer",
                "Import/export package for BED/GFF/BigWig and other genomic file formats.",
                ["import", "export"],
                ["GRanges"],
                ["rtracklayer import BED Bioconductor"],
                0.7,
            ),
            _pkg(
                "ChIPseeker",
                "Bioconductor package for ChIP-seq peak annotation and nearest gene style summaries.",
                ["annotatePeak", "as.GRanges"],
                ["csAnno", "GRanges"],
                ["ChIPseeker annotatePeak TxDb vignette"],
                0.65,
            ),
        ],
        input_hints=["BED", "narrowPeak", "gene annotation"],
        output_hints=["annotated BED", "peak summary", "nearest gene table"],
        operation_keywords=["import intervals", "harmonize seqlevels", "nearest", "overlap", "export"],
        object_hints=["GRanges", "TxDb", "Hits"],
        risk_notes=["Most errors are object/seqlevel/coordinate-system issues, not paper-method issues."],
    ),
    RouteRule(
        name="enrichment",
        patterns=[r"enrichment", r"\bGO\b", r"\bKEGG\b", r"pathway", r"gene set", r"comparecluster"],
        domain="functional_enrichment",
        analysis_intent="Gene set or pathway enrichment",
        packages=[
            _pkg(
                "clusterProfiler",
                "Canonical Bioconductor package for enrichGO/enrichKEGG/GSEA and compareCluster workflows.",
                ["enrichGO", "enrichKEGG", "gseGO", "compareCluster", "bitr", "setReadable"],
                ["enrichResult", "gseaResult", "compareClusterResult"],
                ["clusterProfiler enrichGO OrgDb keyType vignette"],
                0.95,
            ),
            _pkg(
                "fgsea",
                "Fast preranked gene set enrichment workflow.",
                ["fgsea", "fgseaMultilevel"],
                ["data.frame"],
                ["fgsea Bioconductor preranked gene set enrichment"],
                0.6,
            ),
        ],
        input_hints=["gene list", "ranked statistics", "organism annotation"],
        output_hints=["enrichment result table", "pathway table"],
        operation_keywords=["gene ID mapping", "OrgDb", "keyType", "ontology", "export result"],
        object_hints=["enrichResult", "compareClusterResult"],
        risk_notes=["Gene identifier type and organism database are frequent operational blockers."],
    ),
    RouteRule(
        name="single_cell",
        patterns=[r"single.cell", r"\bscrna\b", r"scRNA", r"SingleCellExperiment", r"Seurat", r"cell"],
        domain="single_cell",
        analysis_intent="Single-cell expression processing",
        packages=[
            _pkg(
                "SingleCellExperiment",
                "Core Bioconductor object class for single-cell assays and metadata.",
                ["SingleCellExperiment", "assay", "colData", "rowData", "reducedDim"],
                ["SingleCellExperiment"],
                ["SingleCellExperiment vignette assay colData"],
                0.75,
            ),
            _pkg(
                "scater",
                "Bioconductor package for single-cell QC and visualization.",
                ["perCellQCMetrics", "addPerCellQC", "plotColData"],
                ["SingleCellExperiment"],
                ["scater perCellQCMetrics vignette"],
                0.7,
            ),
            _pkg(
                "scran",
                "Bioconductor package for normalization and downstream single-cell methods.",
                ["computeSumFactors", "modelGeneVar", "findMarkers"],
                ["SingleCellExperiment"],
                ["scran computeSumFactors vignette"],
                0.65,
            ),
        ],
        input_hints=["count matrix", "cell metadata", "SingleCellExperiment"],
        output_hints=["QC table", "normalized counts", "report"],
        operation_keywords=["assay access", "QC metrics", "normalization"],
        object_hints=["SingleCellExperiment"],
        risk_notes=["Object slots and assay names are a common source of execution failures."],
    ),
]


class DomainRouter:
    """Map sparse task text into candidate Bioconductor packages."""

    def parse_intent(self, task_text: str, task_id: str = "") -> TaskIntent:
        text = task_text or ""
        matched = [rule for rule in ROUTE_RULES if rule.matches(text)]
        if not matched:
            return TaskIntent(
                task_id=task_id,
                domain="unknown",
                analysis_intent="unknown",
                risk_notes=["No domain route matched; use broad paper and documentation search."],
            )

        primary = matched[0]
        intent = TaskIntent(
            task_id=task_id,
            domain=primary.domain,
            analysis_intent=primary.analysis_intent,
            input_types=list(dict.fromkeys(x for rule in matched for x in rule.input_hints)),
            output_types=list(dict.fromkeys(x for rule in matched for x in rule.output_hints)),
            operation_keywords=list(dict.fromkeys(x for rule in matched for x in rule.operation_keywords)),
            package_hints=list(dict.fromkeys(pkg.package for rule in matched for pkg in rule.packages)),
            object_hints=list(dict.fromkeys(x for rule in matched for x in rule.object_hints)),
            risk_notes=list(dict.fromkeys(x for rule in matched for x in rule.risk_notes)),
        )
        return intent

    def route(self, task_text: str) -> List[CandidatePackage]:
        packages: List[CandidatePackage] = []
        seen = set()
        for rule in ROUTE_RULES:
            if not rule.matches(task_text or ""):
                continue
            for pkg in rule.packages:
                if pkg.package in seen:
                    continue
                seen.add(pkg.package)
                packages.append(pkg)
        return packages

