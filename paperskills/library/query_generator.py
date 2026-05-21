#!/usr/bin/env python3
"""Query Generator Module - Generate optimized search queries for paper retrieval.

This module analyzes task metadata and generates effective search queries
for PubMed, Europe PMC, and bioRxiv.

Usage:
    generator = QueryGenerator()
    query = generator.generate_pubmed_query({
        "family": "rna",
        "analysis_type": "differential_expression",
        "tool_hint": "DESeq2",
        "key_method": "apeglm"
    })
    # Returns: "DESeq2 apeglm differential expression RNA-seq hasabstract[text]"
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


FAMILY_ALIASES = {
    "spatial": "spatial_transcriptomics",
    "dna_methylation": "methylation",
    "dna-methylation": "methylation",
    "single_cell": "scrna",
    "single-cell": "scrna",
}

ANALYSIS_ALIASES = {
    "quality_control": "qc_metrics",
    "qc": "qc_metrics",
    "data_transformation": "methylation_analysis",
    "tabular_conversion": "methylation_analysis",
    "pseudotime": "trajectory",
    "trajectory_de": "trajectory",
    "regulatory_module": "regulatory_modules",
}


def normalize_label(value: str, aliases: Dict[str, str]) -> str:
    key = (value or "").strip().lower().replace(" ", "_")
    return aliases.get(key, value)


@dataclass
class TaskContext:
    """Context information about a task for query generation."""
    family: str = ""  # rna, methylation, chipseq, etc.
    analysis_type: str = ""  # differential_expression, peak_calling, etc.
    data_type: str = ""  # rna_seq_counts, methylation_data, etc.
    tool_hint: str = ""  # DESeq2, limma, MACS2, etc.
    key_method: str = ""  # apeglm, duplicateCorrelation, etc.
    stage: str = ""  # early, mid, late
    description: str = ""
    retrieval_profile: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "TaskContext":
        return TaskContext(
            family=normalize_label(self.family, FAMILY_ALIASES),
            analysis_type=normalize_label(self.analysis_type, ANALYSIS_ALIASES),
            data_type=self.data_type,
            tool_hint=self.tool_hint,
            key_method=self.key_method,
            stage=self.stage,
            description=self.description,
            retrieval_profile=dict(self.retrieval_profile or {}),
        )
    
    @classmethod
    def from_registry(cls, task_metadata: Dict[str, Any]) -> "TaskContext":
        """Create TaskContext from task registry metadata."""
        return cls(
            family=task_metadata.get("family", ""),
            analysis_type=task_metadata.get("analysis_type", ""),
            data_type=task_metadata.get("data_type", ""),
            tool_hint=task_metadata.get("tool_hint", ""),
            key_method=task_metadata.get("key_method", ""),
            stage=task_metadata.get("stage", ""),
            description=task_metadata.get("description", ""),
            retrieval_profile=task_metadata.get("retrieval_profile", {}) or {},
        )
    
    @classmethod
    def from_objective(cls, objective_text: str) -> "TaskContext":
        """Extract context from OBJECTIVE.md text."""
        context = cls()
        
        # Extract family from text
        family_keywords = {
            "rna": ["rna", "rnaseq", "rna-seq", "differential expression"],
            "methylation": ["methyl", "methylation", "cpg", "bs-seq"],
            "chipseq": ["chip-seq", "chipseq", "chip", "peak calling"],
            "scrna": ["single cell", "sc-rna", "seurat", "scrna"],
            "variant": ["variant", "snp", "mutation", "vcf"],
        }
        
        text_lower = objective_text.lower()
        for family, keywords in family_keywords.items():
            if any(kw in text_lower for kw in keywords):
                context.family = family
                break
        
        # Extract tool hints
        tool_patterns = [
            (r"DESeq2", "DESeq2"),
            (r"limma", "limma"),
            (r"edgeR", "edgeR"),
            (r"methylKit", "methylKit"),
            (r"MACS2", "MACS2"),
            (r"Seurat", "Seurat"),
            (r"ComBat[\-_]?seq", "ComBat-seq"),
            (r"sctransform", "SCTransform"),
            (r"apeglm", "apeglm"),
            (r"duplicateCorrelation", "duplicateCorrelation"),
            (r"voom", "voom"),
            (r"lfcShrink", "lfcShrink"),
        ]
        
        for pattern, tool in tool_patterns:
            if re.search(pattern, objective_text, re.IGNORECASE):
                if not context.tool_hint:
                    context.tool_hint = tool
                else:
                    context.key_method = tool
        
        return context


class QueryGenerator:
    """Generate optimized search queries for academic databases."""
    
    # Synonym expansion for common terms
    SYNONYMS = {
        "rna-seq": ["RNA-seq", "RNA sequencing", "transcriptomics"],
        "differential expression": ["differential expression", "DE analysis", "gene expression"],
        "methylation": ["methylation", "DNA methylation", "epigenetics"],
        "peak calling": ["peak calling", "ChIP-seq analysis"],
        "single cell": ["single cell", "scRNA-seq", "single cell RNA-seq"],
    }
    
    # PubMed field tags
    PUBMED_FIELDS = {
        "title": "[Title]",
        "abstract": "[Abstract]",
        "text": "[Text Word]",
        "mesh": "[MeSH Terms]",
    }
    
    def __init__(self):
        self.expanded_queries_cache = {}
    
    def generate_pubmed_query(self, context: TaskContext, 
                            require_fulltext: bool = True,
                            open_access_only: bool = False) -> str:
        """Generate optimized PubMed search query.
        
        Args:
            context: Task context information
            require_fulltext: Whether to require full text availability
            open_access_only: Whether to restrict to open access papers
            
        Returns:
            Optimized PubMed query string
        """
        context = context.normalized()
        components = []
        
        # 1. Tool/method hint (highest priority - in title/abstract)
        if context.tool_hint:
            tool_term = self._expand_term(context.tool_hint)
            components.append(f"({tool_term})[Title/Abstract]")
        
        # 2. Key method (specific technique)
        if context.key_method:
            method_term = self._expand_term(context.key_method)
            components.append(f"({method_term})[Title/Abstract]")
        
        # 3. Analysis type (general approach)
        if context.analysis_type:
            analysis_terms = self._get_analysis_terms(context.analysis_type)
            components.append(f"({analysis_terms})[Title/Abstract]")
        
        # 4. Family/field context
        if context.family:
            family_terms = self._get_family_terms(context.family)
            if family_terms:
                components.append(f"({family_terms})[Title/Abstract]")
        
        # 5. Data type
        if context.data_type:
            data_terms = self._get_data_type_terms(context.data_type)
            if data_terms:
                components.append(f"({data_terms})[Title/Abstract]")
        
        # Combine with AND
        query = " AND ".join(components) if components else ""
        
        # Add filters
        filters = []
        
        # Require abstract
        filters.append("(hasabstract[text])")
        
        # Full text preference
        if require_fulltext:
            filters.append("(hasfulltext[text] OR haspmc[text])")
        
        # Open access filter
        if open_access_only:
            filters.append("(free full text[sb] OR open access[filter])")
        
        if filters:
            query = f"{query} AND " + " AND ".join(filters) if query else " AND ".join(filters)
        
        return query
    
    def generate_europepmc_query(self, context: TaskContext,
                                open_access_only: bool = True) -> str:
        """Generate Europe PMC search query.
        
        Europe PMC uses different syntax than PubMed.
        
        Args:
            context: Task context
            open_access_only: Restrict to open access
            
        Returns:
            Europe PMC query string
        """
        components = []
        
        # Tool/method
        if context.tool_hint:
            components.append(f'"{context.tool_hint}"')
        
        # Key method
        if context.key_method:
            components.append(f'"{context.key_method}"')
        
        # Analysis type
        if context.analysis_type:
            components.append(context.analysis_type.replace("_", " "))
        
        # Family
        if context.family:
            family_term = self._get_family_search_term(context.family)
            if family_term:
                components.append(f'"{family_term}"')
        
        query = " AND ".join(components) if components else "*"
        
        # Open access filter for Europe PMC
        if open_access_only:
            query += " OPEN_ACCESS:y"
        
        return query
    
    def generate_biorxiv_query(self, context: TaskContext) -> str:
        """Generate bioRxiv search query.
        
        bioRxiv uses simple text search.
        
        Args:
            context: Task context
            
        Returns:
            bioRxiv query string
        """
        components = []
        
        if context.tool_hint:
            components.append(context.tool_hint)
        
        if context.key_method:
            components.append(context.key_method)
        
        if context.family:
            components.append(context.family)
        
        return " ".join(components) if components else ""
    
    def generate_all_queries(self, context: TaskContext) -> Dict[str, str]:
        """Generate queries for all supported sources.
        
        Args:
            context: Task context
            
        Returns:
            Dictionary mapping source name to query string
        """
        return {
            "pubmed": self.generate_pubmed_query(context),
            "europepmc": self.generate_europepmc_query(context),
            "biorxiv": self.generate_biorxiv_query(context),
        }
    
    def _expand_term(self, term: str) -> str:
        """Expand a term with synonyms for better recall.
        
        Args:
            term: Original term
            
        Returns:
            Expanded term with OR logic
        """
        # Check cache
        if term in self.expanded_queries_cache:
            return self.expanded_queries_cache[term]
        
        # Look up synonyms
        term_lower = term.lower()
        for key, synonyms in self.SYNONYMS.items():
            if term_lower == key or term_lower in key:
                expanded = " OR ".join(f'"{s}"' for s in synonyms)
                self.expanded_queries_cache[term] = expanded
                return expanded
        
        # No expansion needed
        self.expanded_queries_cache[term] = f'"{term}"'
        return f'"{term}"'
    
    def _get_analysis_terms(self, analysis_type: str) -> str:
        """Get search terms for analysis type."""
        analysis_terms = {
            "differential_expression": "differential expression OR DE analysis OR gene expression analysis",
            "peak_calling": "peak calling OR ChIP-seq analysis",
            "normalization": "normalization OR scaling OR standardization",
            "batch_correction": "batch correction OR batch effect OR ComBat",
            "clustering": "clustering OR unsupervised learning",
            "dimensionality_reduction": "dimensionality reduction OR PCA OR t-SNE",
            "pathway_enrichment": "pathway enrichment OR GO analysis OR GSEA",
            "variant_calling": "variant calling OR SNP calling",
            "methylation_analysis": "methylation analysis OR DNA methylation",
            "qc_metrics": "quality control OR QC metrics OR benchmarking OR reproducibility OR sensitivity",
            "trajectory": "trajectory OR pseudotime OR branching OR lineage",
            "regulatory_modules": "regulatory module OR gene regulatory network OR transcription factor",
        }
        
        return analysis_terms.get(analysis_type, analysis_type.replace("_", " "))
    
    def _get_family_terms(self, family: str) -> str:
        """Get search terms for analysis family."""
        family_terms = {
            "rna": "RNA-seq OR RNA sequencing OR transcriptomics",
            "methylation": "methylation OR DNA methylation OR bisulfite",
            "chipseq": "ChIP-seq OR ChIP OR chromatin",
            "scrna": "single cell OR scRNA-seq OR single cell RNA",
            "variant": "variant OR SNP OR mutation",
            "epigenomics": "epigenomics OR chromatin",
            "proteomics": "proteomics OR mass spectrometry",
            "spatial_transcriptomics": "spatial transcriptomics OR imaging-based transcriptomics OR MERFISH OR seqFISH",
            "regulatory_networks": "gene regulatory network OR transcription factor OR regulon",
        }
        
        return family_terms.get(family, family)
    
    def _get_family_search_term(self, family: str) -> str:
        """Get primary search term for family."""
        primary = {
            "rna": "RNA-seq",
            "methylation": "methylation",
            "chipseq": "ChIP-seq",
            "scrna": "single-cell",
            "variant": "variant",
            "spatial_transcriptomics": "spatial transcriptomics",
            "regulatory_networks": "gene regulatory network",
        }
        return primary.get(family, family)
    
    def _get_data_type_terms(self, data_type: str) -> str:
        """Get search terms for data type."""
        data_terms = {
            "rna_seq_counts": "RNA-seq OR count data",
            "scrna_expression": "single-cell RNA-seq",
            "chip_seq_reads": "ChIP-seq",
            "methylation_data": "methylation data OR bisulfite sequencing",
            "genomic_variants": "genomic variants OR SNP data",
        }
        
        return data_terms.get(data_type, data_type.replace("_", " "))


class QueryOptimizer:
    """Optimize queries for better retrieval performance."""
    
    def __init__(self):
        self.generator = QueryGenerator()
    
    def optimize_for_recall(self, context: TaskContext) -> Dict[str, str]:
        """Generate queries optimized for high recall.
        
        Uses broader terms and fewer constraints.
        
        Returns:
            Dictionary of optimized queries
        """
        # Broader context
        broad_context = TaskContext(
            family=context.family,
            analysis_type=context.analysis_type,
            tool_hint=context.tool_hint,
        )
        
        return self.generator.generate_all_queries(broad_context)
    
    def optimize_for_precision(self, context: TaskContext) -> Dict[str, str]:
        """Generate queries optimized for high precision.
        
        Uses specific terms and stricter filters.
        
        Returns:
            Dictionary of optimized queries
        """
        return {
            "pubmed": self.generator.generate_pubmed_query(
                context, require_fulltext=True, open_access_only=True
            ),
            "europepmc": self.generator.generate_europepmc_query(
                context, open_access_only=True
            ),
            "biorxiv": self.generator.generate_biorxiv_query(context),
        }
    
    def generate_multi_strategy_queries(self, context: TaskContext) -> List[Dict[str, str]]:
        """Generate multiple query strategies for comprehensive search.
        
        Returns:
            List of query sets with different strategies
        """
        strategies = []
        
        # Strategy 1: Tool-focused
        if context.tool_hint:
            tool_context = TaskContext(tool_hint=context.tool_hint)
            strategies.append({
                "name": "tool_focused",
                "queries": self.generator.generate_all_queries(tool_context)
            })
        
        # Strategy 2: Method-focused
        if context.key_method:
            method_context = TaskContext(key_method=context.key_method)
            strategies.append({
                "name": "method_focused",
                "queries": self.generator.generate_all_queries(method_context)
            })
        
        # Strategy 3: Full context
        strategies.append({
            "name": "full_context",
            "queries": self.generator.generate_all_queries(context)
        })
        
        # Strategy 4: Recall-optimized
        strategies.append({
            "name": "recall_optimized",
            "queries": self.optimize_for_recall(context)
        })
        
        return strategies


def generate_search_query(task_metadata: Dict[str, Any], 
                         source: str = "pubmed") -> str:
    """Convenience function to generate search query.
    
    Args:
        task_metadata: Task metadata dictionary
        source: Target source ("pubmed", "europepmc", "biorxiv")
        
    Returns:
        Search query string
    """
    generator = QueryGenerator()
    
    if isinstance(task_metadata, dict):
        context = TaskContext.from_registry(task_metadata)
    else:
        context = task_metadata
    
    if source == "pubmed":
        return generator.generate_pubmed_query(context)
    elif source == "europepmc":
        return generator.generate_europepmc_query(context)
    elif source == "biorxiv":
        return generator.generate_biorxiv_query(context)
    else:
        raise ValueError(f"Unknown source: {source}")


def get_example_queries() -> Dict[str, str]:
    """Get example queries for common tasks.
    
    Returns:
        Dictionary of example queries by task type
    """
    generator = QueryGenerator()
    
    examples = {}
    
    # Example 1: DESeq2 with apeglm
    examples["deseq2_apeglm"] = generator.generate_pubmed_query(
        TaskContext(
            tool_hint="DESeq2",
            key_method="apeglm",
            family="rna",
            analysis_type="differential_expression"
        )
    )
    
    # Example 2: limma duplicateCorrelation
    examples["limma_dupcor"] = generator.generate_pubmed_query(
        TaskContext(
            tool_hint="limma",
            key_method="duplicateCorrelation",
            family="rna",
            analysis_type="differential_expression"
        )
    )
    
    # Example 3: ComBat-seq batch correction
    examples["combat_seq"] = generator.generate_pubmed_query(
        TaskContext(
            tool_hint="ComBat-seq",
            family="rna",
            analysis_type="batch_correction"
        )
    )
    
    return examples


if __name__ == "__main__":
    # Example usage
    print("Testing Query Generator...\n")
    
    generator = QueryGenerator()
    
    # Test case 1: DESeq2 apeglm
    context1 = TaskContext(
        tool_hint="DESeq2",
        key_method="apeglm",
        family="rna",
        analysis_type="differential_expression"
    )
    
    queries1 = generator.generate_all_queries(context1)
    print("Query 1: DESeq2 + apeglm")
    for source, query in queries1.items():
        print(f"  {source}: {query}")
    print()
    
    # Test case 2: limma duplicateCorrelation
    context2 = TaskContext(
        tool_hint="limma",
        key_method="duplicateCorrelation",
        family="rna",
        analysis_type="differential_expression"
    )
    
    queries2 = generator.generate_all_queries(context2)
    print("Query 2: limma + duplicateCorrelation")
    for source, query in queries2.items():
        print(f"  {source}: {query}")
    print()
    
    # Test case 3: ComBat-seq
    context3 = TaskContext(
        tool_hint="ComBat-seq",
        family="rna",
        analysis_type="batch_correction"
    )
    
    queries3 = generator.generate_all_queries(context3)
    print("Query 3: ComBat-seq batch correction")
    for source, query in queries3.items():
        print(f"  {source}: {query}")
