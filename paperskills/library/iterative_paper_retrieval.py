#!/usr/bin/env python3
"""
Iterative Paper Retrieval System with Paper Pool and Adaptive Scoring

Core workflow:
1. Initialize paper pool (empty)
2. Multi-round search:
   - Search with current query
   - Add candidates to pool with relevance scores
   - Analyze pool content
   - Refine search strategy based on gaps
3. Score papers in pool using task context + abstracts
4. Rank and select top-K papers
5. Fetch full content for top-K papers only

Benefits:
- Avoids fetching low-quality papers (saves API calls)
- Iteratively improves search strategy
- Makes informed decision about which papers to read deeply
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from paperskills.library.paper_search import PaperSearchAggregator, PaperMetadata
from paperskills.library.query_generator import QueryGenerator, TaskContext


def _norm_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _norm_title(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())).strip()


def _title_token_match(a: str, b: str) -> bool:
    ta = {t for t in _norm_title(a).split() if len(t) >= 4}
    tb = {t for t in _norm_title(b).split() if len(t) >= 4}
    if not ta or not tb:
        return False
    if _norm_title(a) in _norm_title(b) or _norm_title(b) in _norm_title(a):
        return True
    overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
    return overlap >= 0.5 and bool({"scmasigpro", "spatialqm", "cregulon"} & (ta | tb))


@dataclass
class PoolEntry:
    """A paper candidate in the retrieval pool."""
    paper: PaperMetadata
    round_added: int
    search_query: str
    relevance_score: float = 0.0  # 0-1 score based on task match
    method_package_score: float = 0.0  # method/software/package match
    skill_extractability_score: float = 0.0  # object/function/workflow signal
    source_scope_score: float = 0.0  # source type and source-task scope fit
    negative_source_penalty: float = 0.0  # application/review penalty
    source_type_guess: str = "unknown"
    abstract_quality_score: float = 0.0  # 0-1 based on abstract informativeness
    final_score: float = 0.0  # Combined score for ranking
    fetch_decision: bool = False  # Whether to fetch full text
    fetched_content: Optional[str] = None
    methods_extracted: Optional[str] = None
    generated_skill: Optional[Dict[str, Any]] = None
    generated_skill_md: Optional[str] = None
    skill_tags: List[str] = field(default_factory=list)
    skill_path: Optional[str] = None
    skill_validation: Optional[Dict[str, Any]] = None
    low_confidence_selection: bool = False
    extractor: str = "heuristic"
    extractor_metadata: Dict[str, Any] = field(default_factory=dict)
    source_bundle: List[Dict[str, Any]] = field(default_factory=list)
    source_access_level: str = ""
    source_url: str = ""
    source_local_path: str = ""
    formal_source_valid: bool = False
    formal_source_strength: str = ""
    formal_skill_valid: bool = False
    source_attempts: List[Dict[str, Any]] = field(default_factory=list)
    source_portfolio: List[Dict[str, Any]] = field(default_factory=list)
    auxiliary_sources: List[Dict[str, Any]] = field(default_factory=list)
    docs_source_valid: bool = False
    paper_fulltext_skill: bool = False
    docs_supported_skill: bool = False
    abstract_only_skill: bool = False
    failure_category: str = ""


FAMILY_ALIASES = {
    "spatial": "spatial_transcriptomics",
    "spatial_transcriptomics": "spatial_transcriptomics",
    "spatial-transcriptomics": "spatial_transcriptomics",
    "dna_methylation": "methylation",
    "dna-methylation": "methylation",
    "methylation": "methylation",
    "single-cell": "scrna",
    "single_cell": "scrna",
    "single-cell_rna": "scrna",
    "single_cell_rna": "scrna",
    "regulatory_networks": "regulatory_networks",
}


ANALYSIS_ALIASES = {
    "quality_control": "qc_metrics",
    "qc": "qc_metrics",
    "quality": "qc_metrics",
    "data_transformation": "methylation_analysis",
    "tabular_conversion": "methylation_analysis",
    "methylation": "methylation_analysis",
    "trajectory_de": "trajectory",
    "pseudotime": "trajectory",
    "regulatory_module": "regulatory_modules",
    "regulatory_modules": "regulatory_modules",
}


def normalize_task_context(context: TaskContext) -> TaskContext:
    """Normalize agent-supplied retrieval labels to the retriever taxonomy."""
    family_key = (context.family or "").strip().lower().replace(" ", "_")
    analysis_key = (context.analysis_type or "").strip().lower().replace(" ", "_")
    return TaskContext(
        family=FAMILY_ALIASES.get(family_key, context.family),
        analysis_type=ANALYSIS_ALIASES.get(analysis_key, context.analysis_type),
        data_type=context.data_type,
        tool_hint=context.tool_hint,
        key_method=context.key_method,
        stage=context.stage,
        description=context.description,
        retrieval_profile=dict(context.retrieval_profile or {}),
    )


@dataclass
class RetrievalRound:
    """Record of one search round."""
    round_num: int
    query: str
    strategy: str  # Why this query was chosen
    results_count: int
    new_papers_added: int
    pool_size_after: int


@dataclass
class PaperSearchResearchPlan:
    """DeepResearch-style plan for method-source discovery."""
    method_family: str = ""
    target_source_types: List[str] = field(default_factory=list)
    must_cover: List[str] = field(default_factory=list)
    negative_source_types: List[str] = field(default_factory=list)
    fallback_route: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_family": self.method_family,
            "target_source_types": self.target_source_types,
            "must_cover": self.must_cover,
            "negative_source_types": self.negative_source_types,
            "fallback_route": self.fallback_route,
        }


@dataclass
class ResearchDecision:
    """One search/read/judge decision in the research loop."""
    round: int
    goal: str
    query: str = ""
    source_channel: str = "pubmed"
    result_summary: str = ""
    decision: str = "retry"
    gap: str = ""
    next_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round,
            "goal": self.goal,
            "query": self.query,
            "source_channel": self.source_channel,
            "result_summary": self.result_summary,
            "decision": self.decision,
            "gap": self.gap,
            "next_action": self.next_action,
        }


class IterativePaperRetriever:
    """
    Multi-round paper retrieval with pool-based candidate management.
    
    Usage:
        retriever = IterativePaperRetriever(max_rounds=5, top_k=3)
        task_context = TaskContext(family="rna", analysis_type="differential_expression", ...)
        
        # Run iterative retrieval
        top_papers = await retriever.retrieve(task_context)
        
        # top_papers contains full content for the best 3 papers
    """
    
    def __init__(
        self,
        max_rounds: int = 5,
        top_k: int = 3,
        pool_size_limit: int = 20,
        min_relevance_threshold: float = 0.3,
        skill_output_dir: Optional[Path] = None,
        paper_skill_extractor: str = "heuristic",
    ):
        self.max_rounds = max_rounds
        self.top_k = top_k
        self.pool_size_limit = pool_size_limit
        self.min_relevance_threshold = min_relevance_threshold
        self.skill_output_dir = skill_output_dir or Path(".cache/paper_iterative_skills")
        
        self.searcher = PaperSearchAggregator()
        # Use PubMed only - Europe PMC and bioRxiv are unreliable
        self.use_pubmed_only = True
        self.query_gen = QueryGenerator()
        self.paper_skill_extractor = paper_skill_extractor
        
        # State
        self.pool: Dict[str, PoolEntry] = {}  # pmid -> entry
        self.rounds: List[RetrievalRound] = []
        self.research_decisions: List[ResearchDecision] = []
        self.research_plan: PaperSearchResearchPlan = PaperSearchResearchPlan()
        self.task_context: Optional[TaskContext] = None
        
    async def retrieve(self, task_context: TaskContext) -> List[PoolEntry]:
        """
        Main entry point: run iterative retrieval and return top-K papers with full content.
        
        Returns:
            List of PoolEntry with fetched_content populated for top-K papers
        """
        self.task_context = normalize_task_context(task_context)
        self.pool = {}
        self.rounds = []
        self.research_decisions = []
        self.research_plan = self._build_research_plan()
        
        # Phase 1: Multi-round search to populate pool
        await self._populate_pool()
        
        # Phase 2: Score all papers in pool
        self._score_pool()
        
        # Phase 3: Select top-K and fetch full content
        top_entries = self._select_and_fetch_top_k()
        
        return top_entries
    
    async def _populate_pool(self):
        """Run multiple search rounds to build candidate pool."""
        
        # Generate initial query set based on task context
        base_queries = self._generate_method_aware_queries()
        
        # Track which queries we've used
        used_queries = set()
        
        for round_num in range(1, self.max_rounds + 1):
            # Select best query for this round
            query, strategy = self._select_query_for_round(
                round_num, base_queries, used_queries
            )
            used_queries.add(query)
            
            # Execute search (PubMed only for reliability)
            try:
                from paperskills.library.paper_search import PubMedSearcher
                pm_searcher = PubMedSearcher()
                pm_results = pm_searcher.search(query, max_results=5)
                papers = pm_results.get("papers", []) if isinstance(pm_results, dict) else []
            except Exception as e:
                print(f"Search error in round {round_num}: {e}")
                papers = []
            
            # Add new papers to pool
            new_count = 0
            for paper_dict in papers:
                # Handle both dict and PaperMetadata objects
                if hasattr(paper_dict, 'pmid'):
                    pmid = paper_dict.pmid
                else:
                    pmid = paper_dict.get("pmid")
                if not pmid or pmid in self.pool:
                    continue
                
                # Create metadata object (handle both dict and existing PaperMetadata)
                if isinstance(paper_dict, PaperMetadata):
                    paper = paper_dict
                else:
                    paper = PaperMetadata(**paper_dict)
                
                entry = PoolEntry(
                    paper=paper,
                    round_added=round_num,
                    search_query=query,
                )
                self.pool[pmid] = entry
                new_count += 1
            
            # Record this round
            self.rounds.append(RetrievalRound(
                round_num=round_num,
                query=query,
                strategy=strategy,
                results_count=len(papers),
                new_papers_added=new_count,
                pool_size_after=len(self.pool),
            ))
            self.research_decisions.append(self._judge_search_round(
                round_num=round_num,
                query=query,
                strategy=strategy,
                papers=papers,
                new_count=new_count,
            ))
            
            # Check if we should continue
            if len(self.pool) >= self.pool_size_limit:
                print(f"Pool size limit reached ({self.pool_size_limit}), stopping search")
                break
            
            if round_num >= 2 and new_count == 0 and self._pool_has_candidate_source():
                print(f"No new papers in round {round_num}, search exhausted")
                break
            
            time.sleep(0.5)  # Rate limiting

        if not self.pool and self._can_use_docs_fallback():
            self._add_documentation_candidate("no_candidate_found")
        elif self.pool and self._can_use_docs_fallback() and not any(self._passes_candidate_gate(e, allow_weak=True) for e in self.pool.values()):
            self._add_documentation_candidate("only_wrapper_sources_found")

    def _judge_search_round(
        self,
        *,
        round_num: int,
        query: str,
        strategy: str,
        papers: List[Any],
        new_count: int,
    ) -> ResearchDecision:
        """Summarize what the round taught the research loop."""
        if not papers:
            return ResearchDecision(
                round=round_num,
                goal=strategy,
                query=query,
                result_summary="0 candidates returned",
                decision="switch_channel" if self._can_use_docs_fallback() else "retry",
                gap="no_candidate_found",
                next_action="search_bioconductor_docs" if self._can_use_docs_fallback() else "try_next_query",
            )
        temp_entries = []
        for item in papers:
            paper = item if isinstance(item, PaperMetadata) else PaperMetadata(**item)
            method_score, source_type = self._compute_method_package_score(paper)
            _, refined = self._compute_source_scope_score(paper, source_type)
            temp_entries.append((paper, method_score, refined))
        if all(refined == "wrapper_or_downstream_pipeline" for _, _, refined in temp_entries):
            return ResearchDecision(
                round=round_num,
                goal=strategy,
                query=query,
                result_summary=f"{len(papers)} candidates, all wrapper/downstream-like",
                decision="reject",
                gap="only_wrapper_sources_found",
                next_action="search_canonical_method_source",
            )
        if any(refined == "canonical_method_or_software" for _, _, refined in temp_entries):
            return ResearchDecision(
                round=round_num,
                goal=strategy,
                query=query,
                result_summary=f"{len(papers)} candidates; canonical source candidate present; new={new_count}",
                decision="accept",
                gap="",
                next_action="score_and_fetch_fulltext",
            )
        return ResearchDecision(
            round=round_num,
            goal=strategy,
            query=query,
            result_summary=f"{len(papers)} candidates; no canonical source yet; new={new_count}",
            decision="retry",
            gap="no_canonical_source",
            next_action="try_next_method_intent",
        )

    def _pool_has_candidate_source(self) -> bool:
        if not self.pool:
            return False
        for entry in self.pool.values():
            method_score, source_type = self._compute_method_package_score(entry.paper)
            _, refined = self._compute_source_scope_score(entry.paper, source_type)
            if refined in {"canonical_method_or_software", "documentation_or_vignette", "protocol_or_workflow"} and method_score > 0:
                return True
        return False

    def _can_use_docs_fallback(self) -> bool:
        profile = self._retrieval_profile()
        package = str(profile.get("package") or "").strip()
        ecosystem = str(profile.get("ecosystem") or "").lower()
        return bool(package and ("bioconductor" in ecosystem or ecosystem == "bioc"))

    def _add_documentation_candidate(self, gap: str) -> None:
        """Add an official-docs candidate when paper search cannot supply one."""
        profile = self._retrieval_profile()
        package = str(profile.get("package") or "").strip()
        if not package:
            return
        pmid = f"docs:{package}"
        if pmid in self.pool:
            return
        paper = PaperMetadata(
            pmid=pmid,
            title=f"{package} official Bioconductor documentation",
            abstract=(
                f"Official Bioconductor package page, vignette, manual, and function documentation for {package}. "
                "This candidate is an implementation evidence fallback when PubMed canonical paper retrieval is insufficient."
            ),
            journal="Bioconductor",
            source="bioconductor_docs",
            is_open_access=True,
            has_fulltext=True,
        )
        entry = PoolEntry(
            paper=paper,
            round_added=len(self.rounds) + 1,
            search_query=f"{package} Bioconductor documentation fallback",
            relevance_score=0.3,
            method_package_score=0.3,
            skill_extractability_score=0.25,
            source_scope_score=0.2,
            source_type_guess="documentation_or_vignette",
            abstract_quality_score=0.1,
            final_score=0.85,
            failure_category=gap,
        )
        self.pool[pmid] = entry
        self.research_decisions.append(ResearchDecision(
            round=len(self.rounds) + 1,
            goal="search_official_package_docs",
            query=entry.search_query,
            source_channel="bioconductor_docs",
            result_summary="Added official documentation fallback candidate",
            decision="switch_channel",
            gap=gap,
            next_action="extract_docs_supported_skill",
        ))

    def _retrieval_profile(self) -> Dict[str, Any]:
        if not self.task_context:
            return {}
        return self.task_context.retrieval_profile or {}

    def _build_research_plan(self) -> PaperSearchResearchPlan:
        """Create a compact method-source research plan before searching."""
        profile = self._retrieval_profile()
        package = str(profile.get("package") or (self.task_context.tool_hint if self.task_context else "")).strip()
        ecosystem = str(profile.get("ecosystem") or "").strip().lower()
        must_cover = self._as_list(profile.get("data_objects"))
        must_cover.extend(self._as_list(profile.get("core_functions")))
        must_cover.extend(self._as_list(profile.get("expected_skill_tags")))
        target_source_types = ["canonical_method_paper", "protocol_workflow"]
        fallback_route = ["pubmed_canonical_method_paper", "pmc_or_publisher_fulltext"]
        if package and ("bioconductor" in ecosystem or ecosystem == "bioc"):
            target_source_types.extend(["official_package_doc", "vignette_manual", "function_doc"])
            fallback_route.extend(["bioconductor_package_page", "bioconductor_vignette_manual", "rdrr_function_docs"])
        return PaperSearchResearchPlan(
            method_family=package or (self.task_context.family if self.task_context else ""),
            target_source_types=target_source_types,
            must_cover=self._dedupe_terms(must_cover),
            negative_source_types=self._as_list(profile.get("negative_source_types")) or [
                "wrapper_gui_pipeline",
                "application_paper",
                "broad_review",
            ],
            fallback_route=fallback_route,
        )

    def _as_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return [str(value).strip()] if str(value).strip() else []

    def _profile_terms(self, *keys: str) -> List[str]:
        profile = self._retrieval_profile()
        terms: List[str] = []
        for key in keys:
            terms.extend(self._as_list(profile.get(key)))
        seen = set()
        unique = []
        for term in terms:
            norm = term.lower()
            if norm not in seen:
                seen.add(norm)
                unique.append(term)
        return unique

    def _dedupe_terms(self, terms: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for term in terms:
            norm = str(term).strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                out.append(str(term).strip())
        return out

    def _generate_method_aware_queries(self) -> Dict[str, str]:
        """Generate a multi-intent query set centered on skill sources."""
        generic = self.query_gen.generate_all_queries(self.task_context)
        profile = self._retrieval_profile()
        if not profile:
            return generic

        package = str(profile.get("package") or "").strip()
        ecosystem = str(profile.get("ecosystem") or "").strip()
        language = str(profile.get("language") or "").strip()
        family = str(profile.get("analysis_family") or self._get_family_terms()).strip()
        data_objects = self._profile_terms("data_objects")
        core_functions = self._profile_terms("core_functions")
        method_terms = self._profile_terms("method_terms")
        search_terms = self._profile_terms("search_terms")

        queries: Dict[str, str] = {}
        if search_terms:
            queries["profile_search_terms"] = " OR ".join(f'("{term}"[Title/Abstract])' for term in search_terms[:3])
        if package:
            canonical_parts = [package, language, "package", family]
            queries["canonical_package"] = " ".join(p for p in canonical_parts if p) + " method software"
        if package and ecosystem:
            docs_terms = [package, ecosystem, *data_objects[:3], "manual vignette workflow"]
            queries["package_documentation"] = " ".join(t for t in docs_terms if t)
        if package and (data_objects or core_functions):
            obj_terms = [package, *data_objects[:4], *core_functions[:4], "data frame object function"]
            queries["function_object"] = " ".join(t for t in obj_terms if t)
        if method_terms:
            method_parts = [*method_terms[:3], language, "package method workflow"]
            queries["method_domain"] = " ".join(t for t in method_parts if t)

        for key, query in generic.items():
            queries.setdefault(key, query)
        return queries
    
    def _select_query_for_round(
        self, 
        round_num: int, 
        base_queries: Dict[str, str],
        used_queries: set
    ) -> Tuple[str, str]:
        """
        Select the best query for this round based on strategy.
        
        Returns: (query, strategy_description)
        """
        # Round 1: Use task-optimized query
        if round_num == 1:
            if self._retrieval_profile():
                for key in ("canonical_package", "profile_search_terms", "package_documentation", "function_object"):
                    query = base_queries.get(key)
                    if query:
                        return query, f"method_aware_{key}"
            query = base_queries.get("pubmed", list(base_queries.values())[0])
            return query, "initial_task_optimized"

        # Method-aware profiles should exhaust package/object/function intents
        # before falling back to broad topical search.
        if self._retrieval_profile():
            for key in (
                "canonical_package",
                "package_documentation",
                "function_object",
                "method_domain",
                "profile_search_terms",
                "pubmed",
                "europepmc",
                "biorxiv",
            ):
                query = base_queries.get(key)
                if query and query not in used_queries:
                    return query, f"method_aware_{key}"
        
        # Round 2: Try broader search without tool filter
        if round_num == 2:
            # Remove specific tool constraint, keep family
            family_terms = self._get_family_terms()
            query = f"{family_terms} differential expression analysis"
            if query not in used_queries:
                return query, "broadened_family_focus"
        
        # Round 3+: Adaptive based on pool content
        if self.pool:
            # Analyze what we have
            journals = {}
            years = {}
            for entry in self.pool.values():
                j = entry.paper.journal
                journals[j] = journals.get(j, 0) + 1
                y = entry.paper.year
                if y:
                    years[y] = years.get(y, 0) + 1
            
            # If we have many recent papers, try older foundational papers
            if max(years.keys(), default=2024) >= 2023:
                query = base_queries.get("pubmed", "")
                query += " foundational method tutorial"
                if query not in used_queries:
                    return query, "seeking_foundation_papers"
            
            # If concentrated in one journal, try other sources
            if len(journals) == 1:
                query = base_queries.get("europepmc", list(base_queries.values())[0])
                if query not in used_queries:
                    return query, "diversifying_sources"
        
        # Default: use any unused query
        for source, query in base_queries.items():
            if query not in used_queries:
                return query, f"unused_{source}"
        
        # Fallback: slight variation of first query
        base = list(base_queries.values())[0]
        query = base + " 2024 2025"
        return query, "recency_boosted"
    
    def _get_family_terms(self) -> str:
        """Get search terms for task family."""
        family = self.task_context.family if self.task_context else ""
        family_terms = {
            "rna": "RNA-seq transcriptomics",
            "spatial_transcriptomics": "spatial transcriptomics",
            "methylation": "DNA methylation",
            "multi_omics": "multi-omics",
            "regulatory_networks": "gene regulatory network transcription factor",
        }
        return family_terms.get(family, family)
    
    def _score_pool(self):
        """Score all papers in pool based on relevance to task."""
        
        for pmid, entry in self.pool.items():
            paper = entry.paper
            
            # Component 1: Title/Abstract relevance (0-0.5)
            relevance = self._compute_relevance_score(paper)
            entry.relevance_score = relevance

            method_score, source_type = self._compute_method_package_score(paper)
            entry.method_package_score = method_score
            entry.source_type_guess = source_type

            extractability = self._compute_skill_extractability_score(paper)
            entry.skill_extractability_score = extractability

            scope_score, refined_source_type = self._compute_source_scope_score(paper, source_type)
            entry.source_scope_score = scope_score
            entry.source_type_guess = refined_source_type

            penalty = self._compute_negative_source_penalty(paper)
            entry.negative_source_penalty = penalty
            
            # Component 2: Abstract quality/informativeness (0-0.3)
            abstract_quality = self._compute_abstract_quality(paper)
            entry.abstract_quality_score = abstract_quality
            
            # Component 3: Source reliability (0-0.2)
            source_bonus = 0.0
            if paper.is_open_access:
                source_bonus += 0.1
            if paper.citation_count and paper.citation_count > 10:
                source_bonus += 0.1
            
            # Combined score. PaperToSkill needs procedural relevance, not
            # just topical relevance, so method/package and extractability
            # can outweigh generic domain match.
            entry.final_score = max(
                0.0,
                relevance + abstract_quality + source_bonus + method_score + extractability + scope_score - penalty,
            )

    def _candidate_text(self, paper: PaperMetadata) -> str:
        publication_type = getattr(paper, "publication_type", "") or ""
        return " ".join([
            paper.title or "",
            paper.abstract or "",
            paper.journal or "",
            str(publication_type),
        ]).lower()

    def _compute_method_package_score(self, paper: PaperMetadata) -> Tuple[float, str]:
        """Score whether candidate is a method/software/package skill source."""
        text = self._candidate_text(paper)
        title = (paper.title or "").lower()
        profile = self._retrieval_profile()
        score = 0.0
        source_type = "unknown"

        exact_terms = self._profile_terms("package", "data_objects", "core_functions")
        for term in exact_terms:
            term_lower = term.lower()
            if term_lower in title:
                score += 0.16
            elif term_lower in text:
                score += 0.08

        positive_terms = [
            "software", "package", "r package", "bioconductor", "method",
            "workflow", "protocol", "vignette", "manual", "algorithm",
            "implemented", "tool",
        ]
        for term in positive_terms:
            if term in title:
                score += 0.07
            elif term in text:
                score += 0.035

        if any(term in text for term in ["software", "package", "bioconductor", "r package"]):
            source_type = "software_or_package"
        elif any(term in text for term in ["workflow", "protocol", "vignette", "manual"]):
            source_type = "workflow_or_documentation"
        elif any(term in text for term in ["method", "algorithm", "model"]):
            source_type = "method_paper"

        preferred = self._as_list(profile.get("preferred_source_types"))
        for term in preferred:
            if term.lower() in text:
                score += 0.05

        return min(score, 0.5), source_type

    def _compute_source_scope_score(self, paper: PaperMetadata, source_type: str) -> Tuple[float, str]:
        """Classify broad source type and score whether it can serve as a skill source.

        This is intentionally generic: it distinguishes reusable method sources
        from wrappers, applications, and reviews without hard-coding task ids.
        """
        text = self._candidate_text(paper)
        title = (paper.title or "").lower()
        publication_type = str(getattr(paper, "publication_type", "") or "").lower()
        profile_terms = self._profile_terms("package", "data_objects", "core_functions")
        title_exact_hit = any(term.lower() in title for term in profile_terms)
        exact_hit = any(term.lower() in text for term in profile_terms)

        doc_terms = ["manual", "vignette", "user guide", "bioconductor", "documentation", "book"]
        canonical_terms = ["software", "package", "r package", "method", "algorithm"]
        protocol_terms = ["workflow", "protocol", "tutorial"]
        wrapper_terms = [
            "gui", "graphical user interface", "wrapper", "web server", "front-end",
            "interface", "pipeline based on", "based on deseq2", "based on edger",
            "based on limma", "uses deseq2", "uses edger", "uses limma",
            "-based r pipeline", " based r pipeline", "pipeline for comprehensive",
            "downstream pipeline", "front end",
        ]
        package_terms = self._profile_terms("package")
        for term in package_terms:
            tl = term.lower()
            wrapper_terms.extend([
                f"based on {tl}",
                f"uses {tl}",
                f"interface for {tl}",
                f"{tl}-based",
                f"{tl} based",
            ])
        application_terms = [
            "patient", "patients", "cohort", "case-control", "disease", "cancer",
            "tumor", "clinical", "biomarker", "prognosis", "association study",
        ]

        score = 0.0
        refined = source_type
        if any(term in text for term in doc_terms):
            refined = "documentation_or_vignette"
            score += 0.14
        if any(term in title for term in canonical_terms) and (title_exact_hit or not profile_terms):
            refined = "canonical_method_or_software"
            score += 0.16
        elif any(term in text for term in canonical_terms) and exact_hit:
            score += 0.08
        if any(term in text for term in protocol_terms):
            refined = "protocol_or_workflow"
            score += 0.08

        wrapper_hit = any(term in text for term in wrapper_terms)
        if wrapper_hit:
            refined = "wrapper_or_downstream_pipeline"
            score -= 0.18
            # A wrapper can be a valid source only when the task profile itself
            # asks for that wrapper, not when it merely mentions a dependency.
            package_title = any(term.lower() in title for term in self._profile_terms("package"))
            if title_exact_hit and package_title and any(term in title for term in ["software", "package", "workflow", "pipeline"]):
                score += 0.08

        application_hit = any(term in title for term in application_terms)
        review_hit = "review" in publication_type or re.search(r"\breview\b", text)
        if application_hit and not title_exact_hit:
            refined = "application_paper"
            score -= 0.12
        if review_hit and not any(term in text for term in doc_terms):
            refined = "broad_review"
            score -= 0.10

        return max(-0.25, min(score, 0.25)), refined

    def _compute_skill_extractability_score(self, paper: PaperMetadata) -> float:
        """Score whether source likely contains executable skill guidance."""
        text = self._candidate_text(paper)
        score = 0.0
        executable_terms = [
            "function", "input", "output", "parameter", "example", "code",
            "workflow", "data object", "object", "data frame", "matrix",
            "normalization", "model matrix",
        ]
        for term in executable_terms:
            if term in text:
                score += 0.025
        for term in self._profile_terms("expected_skill_tags", "data_objects", "core_functions"):
            if term.lower() in text:
                score += 0.04
        return min(score, 0.3)

    def _compute_negative_source_penalty(self, paper: PaperMetadata) -> float:
        """Penalize topical but non-transferable application/review sources."""
        text = self._candidate_text(paper)
        title = (paper.title or "").lower()
        penalty = 0.0
        negative_terms = [
            "patient", "patients", "case-control", "cohort", "disease",
            "association", "biomarker", "clinical samples", "cancer",
            "tumor", "therapy", "prognosis", "syndrome", "disorder",
            "implicated", "identify the potential", "clinical",
        ]
        for term in negative_terms:
            if term in title:
                penalty += 0.06
            elif term in text:
                penalty += 0.025

        profile = self._retrieval_profile()
        for term in self._as_list(profile.get("negative_source_types")):
            if term.lower() in text:
                penalty += 0.05

        publication_type = str(getattr(paper, "publication_type", "") or "").lower()
        if "review" in publication_type or re.search(r"\breview\b", text):
            penalty += 0.08

        # A source that mentions the exact package/object/function can survive
        # disease words because many software papers also include applications.
        exact_hit = any(term.lower() in text for term in self._profile_terms("package", "data_objects", "core_functions"))
        title_method_hit = (
            any(term.lower() in title for term in self._profile_terms("package"))
            and any(term in title for term in ["package", "software", "method", "workflow", "protocol"])
        )
        if exact_hit and title_method_hit:
            penalty *= 0.4
        return min(penalty, 0.35)
    
    def _compute_relevance_score(self, paper: PaperMetadata) -> float:
        """Compute relevance of paper to task (0-0.5)."""
        score = 0.0
        
        if not self.task_context:
            return 0.25  # Default mid-score
        
        # Check title match
        title_lower = paper.title.lower()
        
        # Family match
        family_keywords = {
            "rna": ["rna", "transcript", "gene expression"],
            "spatial_transcriptomics": ["spatial", "spatial transcriptomics", "imaging"],
            "methylation": ["methyl", "epigenetic"],
            "multi_omics": ["multi-omics", "integration", "multiomics"],
            "regulatory_networks": ["regulatory", "regulon", "transcription factor", "gene regulatory"],
        }
        family_terms = family_keywords.get(self.task_context.family, [])
        if any(term in title_lower for term in family_terms):
            score += 0.15
        
        # Analysis type match
        analysis_terms = {
            "differential_expression": ["differential", "de", "deseq", "limma"],
            "qc_metrics": ["quality", "metric", "benchmark", "assessment"],
            "trajectory": ["trajectory", "pseudotime", "branching", "lineage"],
            "methylation_analysis": ["methylation", "bisulfite", "methylkit", "cpg"],
            "regulatory_modules": ["module", "regulon", "transcription factor", "regulatory"],
        }
        analysis_type = self.task_context.analysis_type
        if analysis_type in analysis_terms:
            if any(term in title_lower for term in analysis_terms[analysis_type]):
                score += 0.15
        
        # Tool hint match
        if self.task_context.tool_hint:
            hint_lower = self.task_context.tool_hint.lower()
            if hint_lower in title_lower:
                score += 0.1
            # Check if hint appears in abstract
            if paper.abstract and hint_lower in paper.abstract.lower():
                score += 0.1
        
        # Abstract relevance
        if paper.abstract:
            abstract_lower = paper.abstract.lower()
            method_indicators = [
                "method", "algorithm", "pipeline", "workflow", 
                "implemented", "software", "package", "tool"
            ]
            if any(ind in abstract_lower for ind in method_indicators):
                score += 0.05
        
        return min(score, 0.5)  # Cap at 0.5
    
    def _compute_abstract_quality(self, paper: PaperMetadata) -> float:
        """Compute abstract informativeness (0-0.3)."""
        if not paper.abstract:
            return 0.0
        
        abstract = paper.abstract
        score = 0.0
        
        # Length-based (longer abstracts tend to be more informative)
        words = len(abstract.split())
        if words > 200:
            score += 0.1
        elif words > 100:
            score += 0.05
        
        # Method keywords
        method_keywords = [
            "we propose", "we developed", "we present", "method",
            "approach", "framework", "model", "algorithm"
        ]
        abstract_lower = abstract.lower()
        matches = sum(1 for kw in method_keywords if kw in abstract_lower)
        score += min(matches * 0.03, 0.15)
        
        # Application domain match
        if self.task_context and self.task_context.family:
            if self.task_context.family in abstract_lower:
                score += 0.05
        
        return min(score, 0.3)  # Cap at 0.3
    
    def _select_and_fetch_top_k(self) -> List[PoolEntry]:
        """Select top-K papers and fetch full content."""
        
        # Sort by final score
        sorted_entries = sorted(
            self.pool.values(),
            key=lambda e: e.final_score,
            reverse=True
        )
        
        # Select top-K above threshold
        selected = []
        for entry in sorted_entries[:self.top_k]:
            if entry.final_score >= self.min_relevance_threshold and self._passes_candidate_gate(entry):
                entry.fetch_decision = True
                selected.append(entry)

        # PaperToSkill requires a skill artifact. If the pool is non-empty but
        # scores are below the conservative threshold, still fetch the best
        # candidates and mark them low-confidence rather than returning no skill.
        if not selected and sorted_entries:
            for entry in sorted_entries[:self.top_k]:
                if self._retrieval_profile() and not self._passes_candidate_gate(entry, allow_weak=True):
                    continue
                entry.fetch_decision = True
                entry.low_confidence_selection = True
                selected.append(entry)
        
        # Fetch full content for selected papers
        for entry in selected:
            try:
                content = self._fetch_paper_content(entry)
                entry.fetched_content = content
                
                # Extract methods section
                methods = self._extract_methods(content)
                entry.methods_extracted = methods
                self._generate_skill_for_entry(entry)
                
            except Exception as e:
                print(f"Failed to fetch {entry.paper.pmid}: {e}")
                entry.fetched_content = f"[Error: {e}]"
                self._generate_skill_for_entry(entry)
        
        return selected

    def _passes_candidate_gate(self, entry: PoolEntry, *, allow_weak: bool = False) -> bool:
        """Check whether candidate is likely to yield a useful PaperSkill."""
        if not self._retrieval_profile():
            return True
        method_signal = entry.method_package_score
        extract_signal = entry.skill_extractability_score
        exact_text = self._candidate_text(entry.paper)
        title = (entry.paper.title or "").lower()
        exact_hit = any(
            term.lower() in exact_text
            for term in self._profile_terms("package", "data_objects", "core_functions")
        )
        title_exact_hit = any(
            term.lower() in title
            for term in self._profile_terms("package", "data_objects", "core_functions")
        )
        title_method_hit = title_exact_hit and any(
            term in title
            for term in ["package", "software", "method", "workflow", "protocol", "manual", "vignette", "user guide"]
        )
        if entry.source_type_guess in {"application_paper", "broad_review"} and not title_method_hit:
            return False
        if entry.source_type_guess == "wrapper_or_downstream_pipeline" and not title_method_hit:
            return False
        if title_method_hit:
            return True
        if allow_weak:
            return exact_hit and (method_signal + extract_signal + entry.source_scope_score) >= 0.18 and entry.negative_source_penalty < 0.12
        return (
            title_exact_hit
            and (method_signal >= 0.12 or extract_signal >= 0.12 or entry.source_scope_score >= 0.12)
            and entry.negative_source_penalty < 0.12
        )
    
    def _fetch_paper_content(self, entry: PoolEntry) -> str:
        """Fetch full text content for a paper."""
        from paperskills.library.paper_extraction import PDFExtractor
        
        paper = entry.paper
        extractor = PDFExtractor()
        
        # Try PMC first
        pmc_text: str | None = None
        pmc_source: tuple[str, str, str, bool, str, List[Dict[str, Any]], List[Dict[str, Any]]] | None = None
        if paper.pmcid:
            content = extractor.download_from_pmc(paper.pmcid)
            if content:
                self._record_content_source(entry, extractor)
                if entry.source_access_level in {"pmc_pdf_fulltext", "publisher_pdf_fulltext"}:
                    return extractor.extract_text(content)
                pmc_text = extractor.extract_text(content)
                pmc_source = (
                    entry.source_access_level,
                    entry.source_url,
                    entry.source_local_path,
                    entry.formal_source_valid,
                    entry.formal_source_strength,
                    list(entry.source_attempts),
                    list(entry.source_portfolio),
                )
        
        # Try DOI/publisher PDF before accepting XML fallback.
        if paper.doi:
            content = extractor.download_from_doi(paper.doi)
            if content:
                self._record_content_source(entry, extractor)
                if pmc_source:
                    entry.auxiliary_sources = list(pmc_source[6])
                return extractor.extract_text(content)

        if pmc_text is not None:
            if pmc_source:
                (
                    entry.source_access_level,
                    entry.source_url,
                    entry.source_local_path,
                    entry.formal_source_valid,
                    entry.formal_source_strength,
                    entry.source_attempts,
                    entry.source_portfolio,
                ) = pmc_source
            return pmc_text
        
        # Fallback: just abstract + metadata
        entry.source_access_level = "abstract_only"
        entry.source_url = ""
        entry.source_local_path = ""
        entry.formal_source_valid = False
        entry.formal_source_strength = "low"
        entry.source_attempts = list(getattr(extractor, "last_attempts", []) or [])
        entry.source_portfolio = []
        entry.low_confidence_selection = True
        return f"Title: {paper.title}\nAbstract: {paper.abstract}\n[Full text unavailable]"

    def _record_content_source(self, entry: PoolEntry, extractor: Any) -> None:
        """Copy source acquisition metadata from the downloader into the pool entry."""
        entry.source_access_level = str(getattr(extractor, "last_access_level", "") or "pmc_xml_fulltext")
        entry.source_url = str(getattr(extractor, "last_source_url", "") or "")
        entry.source_local_path = str(getattr(extractor, "last_local_path", "") or "")
        entry.source_attempts = list(getattr(extractor, "last_attempts", []) or [])
        entry.source_portfolio = list(getattr(extractor, "last_source_portfolio", []) or [])
        entry.formal_source_valid = entry.source_access_level in {
            "publisher_pdf_fulltext",
            "pmc_pdf_fulltext",
            "pmc_xml_fulltext",
            "europepmc_xml_fulltext",
            "html_fulltext",
        }
        if entry.source_access_level in {"publisher_pdf_fulltext", "pmc_pdf_fulltext"}:
            entry.formal_source_strength = "high"
        elif entry.source_access_level in {"pmc_xml_fulltext", "europepmc_xml_fulltext", "html_fulltext"}:
            entry.formal_source_strength = "medium_high"
        else:
            entry.formal_source_strength = "low"

    def _failure_category(self, entry: PoolEntry) -> str:
        """Classify the current PaperSearch/PaperSkill failure mode."""
        if entry.paper_fulltext_skill:
            return ""
        if entry.source_type_guess == "wrapper_or_downstream_pipeline":
            return "only_wrapper_sources_found"
        if entry.formal_skill_valid and entry.docs_source_valid and not entry.formal_source_valid:
            return "docs_found_but_no_paper_fulltext"
        if not entry.formal_source_valid and entry.source_access_level == "abstract_only":
            return "paper_fulltext_unavailable"
        if entry.formal_source_valid and not entry.formal_skill_valid:
            return "source_sufficient_but_skill_grounding_failed"
        if entry.source_access_level in {"documentation_fulltext"} and not entry.formal_skill_valid:
            return "source_sufficient_but_skill_grounding_failed"
        return entry.failure_category or "source_scope_mismatch"
    
    def _extract_methods(self, content: str) -> str:
        """Extract methods section from full content."""
        from paperskills.library.paper_extraction import MethodSectionExtractor
        
        extractor = MethodSectionExtractor()
        methods = extractor.extract(content)
        
        if methods and len(methods) > 100:
            return methods
        
        # Fallback: return first 5000 chars as context
        return content[:5000]

    def _generate_skill_for_entry(self, entry: PoolEntry) -> None:
        """Generate, tag, validate, and persist a runtime skill for one paper."""
        from paperskills.library.paper_extraction import ExtractedContent, CodeBlockExtractor
        from paperskills.library.skill_synthesis import SkillGenerator, validate_skill

        paper = entry.paper
        full_text = entry.fetched_content or ""
        methods_text = entry.methods_extracted or ""
        if not methods_text:
            methods_text = f"Title: {paper.title}\nAbstract: {paper.abstract or ''}"
        if not full_text:
            full_text = methods_text

        code_snippets = []
        try:
            code_snippets = [
                snippet.code
                for snippet in CodeBlockExtractor().extract_from_text(full_text)
                if snippet.code.strip()
            ]
        except Exception:
            code_snippets = []

        content = ExtractedContent(
            methods_text=methods_text,
            code_snippets=code_snippets,
            full_text=full_text,
            metadata={
                "pmid": paper.pmid,
                "pmcid": paper.pmcid,
                "doi": paper.doi,
                "title": paper.title,
                "score": entry.final_score,
            },
        )

        if self.paper_skill_extractor in {"paper2skills", "hybrid"}:
            try:
                from paperskills.library.paper2skills_extractor import Paper2SkillsExtractor

                result = Paper2SkillsExtractor(
                    cache_dir=self.skill_output_dir / "source_cache",
                ).extract(
                    task_id=(self.task_context.tool_hint if self.task_context else "") or paper.pmid or paper.doi or "unknown",
                    task_context=self.task_context or TaskContext(),
                    paper=paper,
                    fetched_content=full_text,
                    methods_text=methods_text,
                )
                metadata = result.get("metadata") or {}
                quality = metadata.get("quality") or {}
                access_level = str(quality.get("source_access_level") or "")
                if access_level == "documentation_fulltext" and not entry.formal_source_valid:
                    entry.source_access_level = "documentation_fulltext"
                    entry.formal_source_strength = "implementation_auxiliary"
                entry.docs_source_valid = bool(quality.get("docs_source_valid") or access_level == "documentation_fulltext")
                is_valid = bool(quality.get("evidence_slots_complete"))
                issues = [] if is_valid else ["Paper2Skills evidence slots incomplete"]
                if quality.get("abstract_only"):
                    issues.append("Abstract-only or insufficient source text")
                if quality.get("wrapper_only"):
                    issues.append("Wrapper/GUI/pipeline-only source cannot satisfy formal PaperSkill")

                if is_valid or self.paper_skill_extractor == "paper2skills":
                    skill_md = str(result.get("skill_md") or "")
                    skill_dict = {
                        "name": f"paper2skills-{paper.doi or paper.pmid or 'unknown'}",
                        "source": "paper2skills_extractor",
                        "pmid": paper.pmid,
                        "pmcid": paper.pmcid,
                        "doi": paper.doi,
                        "tool": self.task_context.tool_hint if self.task_context else "",
                        "method_summary": (metadata.get("source_slots") or {}).get("method_scope", ""),
                        "paper_title": paper.title,
                        "paper_authors": paper.authors,
                        "paper_year": paper.year,
                        "completeness_score": 0.8 if is_valid else 0.4,
                    }
                    entry.extractor = "paper2skills"
                    entry.extractor_metadata = metadata
                    entry.source_bundle = list(metadata.get("source_bundle") or [])
                    entry.skill_tags = self._generate_skill_tags(entry, bool((metadata.get("source_slots") or {}).get("code_snippets")))
                    if "paper2skills" not in entry.skill_tags:
                        entry.skill_tags.extend(["paper2skills", "source_aware"])
                    entry.low_confidence_selection = entry.low_confidence_selection or not is_valid
                    entry.formal_skill_valid = bool(is_valid and not quality.get("abstract_only"))
                    entry.paper_fulltext_skill = bool(entry.formal_source_valid and entry.formal_skill_valid)
                    entry.docs_supported_skill = bool(entry.docs_source_valid and entry.formal_skill_valid and not entry.formal_source_valid)
                    entry.abstract_only_skill = bool(quality.get("abstract_only"))
                    entry.failure_category = self._failure_category(entry)
                    skill_dir = self._write_runtime_skill(entry, skill_dict, skill_md, is_valid, issues)
                    entry.generated_skill = skill_dict
                    entry.generated_skill_md = skill_md
                    entry.skill_path = str(skill_dir / "SKILL.md")
                    entry.skill_validation = {
                        "is_valid": is_valid,
                        "issues": issues,
                        "completeness_score": skill_dict["completeness_score"],
                    }
                    return

                entry.extractor = "hybrid_fallback"
                entry.extractor_metadata = metadata
                entry.source_bundle = list(metadata.get("source_bundle") or [])
            except Exception as exc:
                if self.paper_skill_extractor == "paper2skills":
                    entry.extractor = "paper2skills"
                    entry.extractor_metadata = {
                        "error": str(exc),
                        "quality": {"evidence_slots_complete": False},
                    }
                    skill_md = (
                        "---\n"
                        "name: paper2skills-extraction-error\n"
                        "source: paper2skills_extractor\n"
                        "---\n\n"
                        "## Method Scope\n"
                        f"Paper2Skills extraction failed before a grounded skill could be generated: {exc}\n"
                    )
                    skill_dict = {
                        "name": "paper2skills-extraction-error",
                        "source": "paper2skills_extractor",
                        "pmid": paper.pmid,
                        "pmcid": paper.pmcid,
                        "doi": paper.doi,
                        "tool": self.task_context.tool_hint if self.task_context else "",
                        "method_summary": "",
                        "paper_title": paper.title,
                        "paper_authors": paper.authors,
                        "paper_year": paper.year,
                        "completeness_score": 0.0,
                    }
                    entry.skill_tags = self._generate_skill_tags(entry, False) + ["paper2skills", "extraction_error"]
                    entry.low_confidence_selection = True
                    skill_dir = self._write_runtime_skill(entry, skill_dict, skill_md, False, [str(exc)])
                    entry.generated_skill = skill_dict
                    entry.generated_skill_md = skill_md
                    entry.skill_path = str(skill_dir / "SKILL.md")
                    entry.skill_validation = {
                        "is_valid": False,
                        "issues": [str(exc)],
                        "completeness_score": 0.0,
                    }
                    return
                else:
                    entry.extractor = "hybrid_fallback"
                    entry.extractor_metadata = {"fallback_reason": str(exc)}

        skill = SkillGenerator().generate(paper, content, self.task_context or TaskContext())
        skill.tags = self._generate_skill_tags(entry, skill.has_code)
        abstract_only = (
            entry.source_access_level == "abstract_only"
            or "[Full text unavailable]" in full_text
            or not (entry.fetched_content and len(entry.fetched_content) > 1200)
        )
        if abstract_only and not skill.has_code:
            entry.low_confidence_selection = True
        is_valid, issues = validate_skill(skill)
        if abstract_only and "Abstract-only source; formal skill requires full text, documentation, or executable examples" not in issues:
            issues.append("Abstract-only source; formal skill requires full text, documentation, or executable examples")
        if not entry.formal_source_valid and "Formal PaperSkill requires paper full text; documentation-only sources are auxiliary" not in issues:
            issues.append("Formal PaperSkill requires paper full text; documentation-only sources are auxiliary")
        entry.formal_skill_valid = bool(is_valid and not abstract_only and not entry.low_confidence_selection)
        entry.paper_fulltext_skill = bool(entry.formal_source_valid and entry.formal_skill_valid)
        entry.docs_supported_skill = bool(entry.docs_source_valid and entry.formal_skill_valid and not entry.formal_source_valid)
        entry.abstract_only_skill = bool(abstract_only)
        entry.failure_category = self._failure_category(entry)

        skill_md = skill.to_skill_md()
        skill_dir = self._write_runtime_skill(entry, skill.to_dict(), skill_md, is_valid, issues)

        entry.generated_skill = skill.to_dict()
        entry.generated_skill_md = skill_md
        entry.skill_tags = list(skill.tags)
        entry.skill_path = str(skill_dir / "SKILL.md")
        entry.skill_validation = {
            "is_valid": is_valid,
            "issues": issues,
            "completeness_score": skill.completeness_score,
        }

    def _generate_skill_tags(self, entry: PoolEntry, has_code: bool) -> List[str]:
        """Generate discovery tags for a runtime paper skill."""
        tags = ["paper_iterative", "runtime_generated"]
        ctx = self.task_context
        if ctx:
            tags.extend([
                ctx.family,
                ctx.analysis_type,
                ctx.data_type,
                ctx.tool_hint,
                ctx.key_method,
            ])

        paper = entry.paper
        if paper.year:
            tags.append(f"year_{paper.year}")
        if paper.is_open_access:
            tags.append("open_access")
        tags.append("has_code" if has_code else "no_code")
        if entry.source_access_level:
            tags.append(entry.source_access_level)
        tags.append(
            "has_fulltext"
            if entry.source_access_level in {
                "publisher_pdf_fulltext",
                "pmc_pdf_fulltext",
                "pmc_xml_fulltext",
                "europepmc_xml_fulltext",
                "html_fulltext",
                "documentation_fulltext",
            }
            else "abstract_only"
        )

        text = " ".join([paper.title or "", paper.abstract or "", entry.methods_extracted or ""]).lower()
        keyword_tags = {
            "differential_expression": ["differential expression", "deseq", "limma", "edger"],
            "pseudotime": ["pseudotime", "trajectory", "lineage"],
            "single_cell": ["single cell", "scrna", "sc-rna"],
            "spatial": ["spatial transcriptomics", "spatial"],
            "quality_control": ["quality control", "quality metric", "qc"],
            "regulatory_network": ["regulatory network", "transcription factor", "tf"],
            "gam": ["generalized additive", "gam"],
            "jaccard": ["jaccard"],
        }
        for tag, keywords in keyword_tags.items():
            if any(keyword in text for keyword in keywords):
                tags.append(tag)

        cleaned = []
        seen = set()
        for tag in tags:
            if not tag:
                continue
            normalized = re.sub(r"[^a-zA-Z0-9_:+.-]+", "_", str(tag).strip().lower()).strip("_")
            if normalized and normalized not in seen:
                seen.add(normalized)
                cleaned.append(normalized)
        return cleaned

    def _write_runtime_skill(
        self,
        entry: PoolEntry,
        skill_dict: Dict[str, Any],
        skill_md: str,
        is_valid: bool,
        issues: List[str],
    ) -> Path:
        """Persist a generated runtime skill and its metadata."""
        paper = entry.paper
        raw_id = paper.doi or paper.pmid or paper.pmcid or paper.title or "unknown"
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw_id).strip("_")[:120] or "unknown"
        skill_dir = self.skill_output_dir / slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
        metadata = {
            "skill": skill_dict,
            "validation": {
                "is_valid": is_valid,
                "issues": issues,
            },
            "retrieval": {
                "round_added": entry.round_added,
                "search_query": entry.search_query,
                "final_score": entry.final_score,
                "relevance_score": entry.relevance_score,
                "method_package_score": entry.method_package_score,
                "skill_extractability_score": entry.skill_extractability_score,
                "source_scope_score": entry.source_scope_score,
                "negative_source_penalty": entry.negative_source_penalty,
                "source_type_guess": entry.source_type_guess,
                "abstract_quality_score": entry.abstract_quality_score,
                "source_access_level": entry.source_access_level,
                "formal_source_valid": entry.formal_source_valid,
                "formal_source_strength": entry.formal_source_strength,
                "formal_skill_valid": entry.formal_skill_valid,
                "docs_source_valid": entry.docs_source_valid,
                "paper_fulltext_skill": entry.paper_fulltext_skill,
                "docs_supported_skill": entry.docs_supported_skill,
                "abstract_only_skill": entry.abstract_only_skill,
                "failure_category": entry.failure_category,
                "source_url": entry.source_url,
                "source_local_path": entry.source_local_path,
                "source_attempts": entry.source_attempts,
                "source_portfolio": entry.source_portfolio,
                "auxiliary_sources": entry.auxiliary_sources,
            },
            "extractor": {
                "name": entry.extractor,
                "source_bundle_count": len(entry.source_bundle),
                "source_types": sorted({str(s.get("source_type", "")) for s in entry.source_bundle if isinstance(s, dict)}),
                "llm_used_for_skill_extraction": bool(entry.extractor_metadata.get("llm_used_for_skill_extraction")),
                "evidence_slots_complete": bool((entry.extractor_metadata.get("quality") or {}).get("evidence_slots_complete")),
                "metadata": entry.extractor_metadata,
            },
            "source_bundle": entry.source_bundle,
            "paper": paper.to_dict() if hasattr(paper, "to_dict") else {},
        }
        (skill_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return skill_dir
    
    def get_retrieval_report(self) -> Dict[str, Any]:
        """Generate report of the retrieval process."""
        return {
            "task_context": {
                "family": self.task_context.family if self.task_context else None,
                "analysis_type": self.task_context.analysis_type if self.task_context else None,
                "tool_hint": self.task_context.tool_hint if self.task_context else None,
                "retrieval_profile": self.task_context.retrieval_profile if self.task_context else {},
            },
            "research_plan": self.research_plan.to_dict(),
            "research_decisions": [d.to_dict() for d in self.research_decisions],
            "retrieval_rounds": [
                {
                    "round": r.round_num,
                    "query": r.query[:100],
                    "strategy": r.strategy,
                    "new_papers": r.new_papers_added,
                    "pool_size": r.pool_size_after,
                }
                for r in self.rounds
            ],
            "pool_statistics": {
                "total_candidates": len(self.pool),
                "avg_relevance_score": sum(e.relevance_score for e in self.pool.values()) / len(self.pool) if self.pool else 0,
                "avg_final_score": sum(e.final_score for e in self.pool.values()) / len(self.pool) if self.pool else 0,
                "top_3_scores": sorted([e.final_score for e in self.pool.values()], reverse=True)[:3] if self.pool else [],
            },
            "selected_papers": [
                {
                    "pmid": e.paper.pmid,
                    "title": e.paper.title[:80],
                    "final_score": e.final_score,
                    "method_package_score": e.method_package_score,
                    "skill_extractability_score": e.skill_extractability_score,
                    "source_scope_score": e.source_scope_score,
                    "negative_source_penalty": e.negative_source_penalty,
                    "source_type_guess": e.source_type_guess,
                    "round_added": e.round_added,
                    "methods_length": len(e.methods_extracted) if e.methods_extracted else 0,
                    "skill_path": e.skill_path,
                    "skill_tags": e.skill_tags,
                    "skill_valid": e.skill_validation.get("is_valid") if e.skill_validation else None,
                    "low_confidence": e.low_confidence_selection,
                    "extractor": e.extractor,
                    "source_bundle_count": len(e.source_bundle),
                    "source_types": sorted({str(s.get("source_type", "")) for s in e.source_bundle if isinstance(s, dict)}),
                    "llm_used_for_skill_extraction": bool(e.extractor_metadata.get("llm_used_for_skill_extraction")),
                    "evidence_slots_complete": bool((e.extractor_metadata.get("quality") or {}).get("evidence_slots_complete")),
                    "source_access_level": e.source_access_level,
                    "formal_source_valid": e.formal_source_valid,
                    "formal_source_strength": e.formal_source_strength,
                    "formal_skill_valid": e.formal_skill_valid,
                    "docs_source_valid": e.docs_source_valid,
                    "paper_fulltext_skill": e.paper_fulltext_skill,
                    "docs_supported_skill": e.docs_supported_skill,
                    "abstract_only_skill": e.abstract_only_skill,
                    "failure_category": e.failure_category,
                    "source_url": e.source_url,
                    "source_local_path": e.source_local_path,
                    "source_attempts": e.source_attempts,
                    "source_portfolio": e.source_portfolio,
                    "auxiliary_sources": e.auxiliary_sources,
                }
                for e in sorted(self.pool.values(), key=lambda x: x.final_score, reverse=True)[:self.top_k]
                if e.fetch_decision
            ],
        }


# ── Convenience function for agent tools ────────────────────────────────────────

async def iterative_retrieve_papers(
    task_family: str,
    analysis_type: str,
    tool_hint: str = "",
    max_rounds: int = 5,
    top_k: int = 3,
    retrieval_profile: Optional[Dict[str, Any]] = None,
    paper_skill_extractor: str = "heuristic",
) -> str:
    """
    Agent-callable tool for iterative paper retrieval.
    
    Returns JSON with top-K papers and their methods sections.
    """
    try:
        context = normalize_task_context(TaskContext(
            family=task_family,
            analysis_type=analysis_type,
            tool_hint=tool_hint,
            retrieval_profile=retrieval_profile or {},
        ))
        
        skill_output_dir = None
        if os.environ.get("PAPER_ITERATIVE_SKILL_OUTPUT_DIR"):
            skill_output_dir = Path(os.environ["PAPER_ITERATIVE_SKILL_OUTPUT_DIR"])

        retriever = IterativePaperRetriever(
            max_rounds=max_rounds,
            top_k=top_k,
            skill_output_dir=skill_output_dir,
            paper_skill_extractor=paper_skill_extractor,
        )
        
        top_papers = await retriever.retrieve(context)
        report = retriever.get_retrieval_report()
        
        # Format results
        results = {
            "success": True,
            "low_confidence": any(e.low_confidence_selection for e in top_papers),
            "papers_found": len(retriever.pool),
            "papers_selected": len(top_papers),
            "retrieval_process": report,
            "top_papers": [
                {
                    "pmid": e.paper.pmid,
                    "pmcid": e.paper.pmcid,
                    "doi": e.paper.doi,
                    "title": e.paper.title,
                    "authors": e.paper.authors[:3] if e.paper.authors else [],
                    "year": e.paper.year,
                    "journal": e.paper.journal,
                    "score": round(e.final_score, 3),
                    "relevance": round(e.relevance_score, 3),
                    "method_package_score": round(e.method_package_score, 3),
                    "skill_extractability_score": round(e.skill_extractability_score, 3),
                    "source_scope_score": round(e.source_scope_score, 3),
                    "negative_source_penalty": round(e.negative_source_penalty, 3),
                    "source_type_guess": e.source_type_guess,
                    "abstract_quality": round(e.abstract_quality_score, 3),
                    "skill": e.generated_skill,
                    "skill_tags": e.skill_tags,
                    "skill_path": e.skill_path,
                    "skill_validation": e.skill_validation,
                    "low_confidence": e.low_confidence_selection,
                    "extractor": e.extractor,
                    "source_bundle_count": len(e.source_bundle),
                    "source_types": sorted({str(s.get("source_type", "")) for s in e.source_bundle if isinstance(s, dict)}),
                    "llm_used_for_skill_extraction": bool(e.extractor_metadata.get("llm_used_for_skill_extraction")),
                    "evidence_slots_complete": bool((e.extractor_metadata.get("quality") or {}).get("evidence_slots_complete")),
                    "source_access_level": e.source_access_level,
                    "formal_source_valid": e.formal_source_valid,
                    "formal_source_strength": e.formal_source_strength,
                    "formal_skill_valid": e.formal_skill_valid,
                    "docs_source_valid": e.docs_source_valid,
                    "paper_fulltext_skill": e.paper_fulltext_skill,
                    "docs_supported_skill": e.docs_supported_skill,
                    "abstract_only_skill": e.abstract_only_skill,
                    "failure_category": e.failure_category,
                    "source_url": e.source_url,
                    "source_local_path": e.source_local_path,
                    "source_attempts": e.source_attempts,
                    "source_portfolio": e.source_portfolio,
                    "auxiliary_sources": e.auxiliary_sources,
                    "source_bundle": e.source_bundle,
                    "extractor_metadata": e.extractor_metadata,
                    "skill_md": e.generated_skill_md,
                    "methods_section": e.methods_extracted[:2000] if e.methods_extracted else "",
                }
                for e in top_papers
            ],
        }
        
        return json.dumps(results, indent=2, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        }, indent=2)


async def exact_retrieve_paper_skill(
    task_family: str,
    analysis_type: str,
    tool_hint: str = "",
    primary_doi: str = "",
    primary_title: str = "",
) -> str:
    """Resolve one registry paper by DOI/title and generate a PaperSkill.

    This is intentionally stricter than iterative search: it only succeeds when
    the returned paper matches the registry DOI or exact-ish title.
    """
    try:
        context = normalize_task_context(TaskContext(
            family=task_family,
            analysis_type=analysis_type,
            tool_hint=tool_hint,
        ))
        skill_output_dir = None
        if os.environ.get("PAPER_ITERATIVE_SKILL_OUTPUT_DIR"):
            skill_output_dir = Path(os.environ["PAPER_ITERATIVE_SKILL_OUTPUT_DIR"])
        retriever = IterativePaperRetriever(
            max_rounds=1,
            top_k=1,
            skill_output_dir=skill_output_dir,
        )
        retriever.task_context = context

        candidates: List[PaperMetadata] = []
        queries: List[Dict[str, str]] = []
        aggregator = PaperSearchAggregator()
        doi = (primary_doi or "").strip()
        title = (primary_title or "").strip()

        if doi:
            for source, query in (
                ("pubmed", f"{doi}[AID]"),
                ("pubmed", doi),
                ("europepmc", f'DOI:"{doi}"'),
                ("europepmc", doi),
            ):
                queries.append({"source": source, "query": query})
                try:
                    if source == "pubmed":
                        found = aggregator.pubmed.search(query, max_results=3).get("papers", [])
                    else:
                        found = aggregator.europepmc.search(query, max_results=3)
                    candidates.extend(found)
                except Exception:
                    pass

        if title:
            for source, query in (
                ("pubmed", f'"{title}"[Title]'),
                ("europepmc", f'TITLE:"{title}"'),
                ("pubmed", title),
            ):
                queries.append({"source": source, "query": query})
                try:
                    if source == "pubmed":
                        found = aggregator.pubmed.search(query, max_results=3).get("papers", [])
                    else:
                        found = aggregator.europepmc.search(query, max_results=3)
                    candidates.extend(found)
                except Exception:
                    pass

        seen = set()
        unique: List[PaperMetadata] = []
        for paper in candidates:
            key = paper.doi or paper.pmid or paper.pmcid or paper.title
            if key and key not in seen:
                seen.add(key)
                unique.append(paper)

        doi_norm = _norm_identifier(doi)
        title_norm = _norm_title(title)
        selected = None
        for paper in unique:
            if doi_norm and _norm_identifier(paper.doi or "") == doi_norm:
                selected = paper
                break
        if selected is None and title_norm:
            for paper in unique:
                cand = _norm_title(paper.title)
                if cand and _title_token_match(title, paper.title):
                    selected = paper
                    break

        if selected is None:
            return json.dumps({
                "success": False,
                "source": "registry_exact",
                "error": "no_exact_registry_match",
                "queries": queries,
                "candidates_found": len(unique),
                "top_papers": [],
            }, indent=2, ensure_ascii=False)

        entry = PoolEntry(
            paper=selected,
            round_added=0,
            search_query=queries[0]["query"] if queries else primary_doi or primary_title,
            relevance_score=1.0,
            abstract_quality_score=1.0 if selected.abstract and len(selected.abstract.split()) > 80 else 0.5,
            final_score=1.0,
            fetch_decision=True,
        )
        try:
            content = retriever._fetch_paper_content(entry)
            entry.fetched_content = content
            entry.methods_extracted = retriever._extract_methods(content)
        except Exception as e:
            entry.fetched_content = f"[Error: {e}]\nTitle: {entry.paper.title}\nAbstract: {entry.paper.abstract}"
            entry.methods_extracted = entry.fetched_content
        retriever._generate_skill_for_entry(entry)

        result = {
            "success": True,
            "source": "registry_exact",
            "low_confidence": False,
            "papers_found": len(unique),
            "papers_selected": 1,
            "queries": queries,
            "top_papers": [
                {
                    "pmid": entry.paper.pmid,
                    "pmcid": entry.paper.pmcid,
                    "doi": entry.paper.doi,
                    "title": entry.paper.title,
                    "authors": entry.paper.authors[:3] if entry.paper.authors else [],
                    "year": entry.paper.year,
                    "journal": entry.paper.journal,
                    "score": 1.0,
                    "relevance": 1.0,
                    "abstract_quality": round(entry.abstract_quality_score, 3),
                    "skill": entry.generated_skill,
                    "skill_tags": entry.skill_tags,
                    "skill_path": entry.skill_path,
                    "skill_validation": entry.skill_validation,
                    "low_confidence": entry.low_confidence_selection,
                    "source_access_level": entry.source_access_level,
                    "formal_source_valid": entry.formal_source_valid,
                    "formal_source_strength": entry.formal_source_strength,
                    "formal_skill_valid": entry.formal_skill_valid,
                    "docs_source_valid": entry.docs_source_valid,
                    "paper_fulltext_skill": entry.paper_fulltext_skill,
                    "docs_supported_skill": entry.docs_supported_skill,
                    "abstract_only_skill": entry.abstract_only_skill,
                    "failure_category": entry.failure_category,
                    "source_url": entry.source_url,
                    "source_local_path": entry.source_local_path,
                    "source_attempts": entry.source_attempts,
                    "source_portfolio": entry.source_portfolio,
                    "auxiliary_sources": entry.auxiliary_sources,
                    "skill_md": entry.generated_skill_md,
                    "methods_section": entry.methods_extracted[:2000] if entry.methods_extracted else "",
                    "exact_match": {
                        "primary_doi": primary_doi,
                        "primary_title": primary_title,
                    },
                }
            ],
        }
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "source": "registry_exact",
            "error": str(e),
        }, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import asyncio
    
    # Test the retriever
    async def test():
        print("Testing IterativePaperRetriever...\n")
        
        context = TaskContext(
            family="rna",
            analysis_type="trajectory_analysis",
            tool_hint="pseudotime",
        )
        
        retriever = IterativePaperRetriever(max_rounds=3, top_k=2)
        top_papers = await retriever.retrieve(context)
        
        report = retriever.get_retrieval_report()
        print(json.dumps(report, indent=2))
        
        print(f"\n=== Top {len(top_papers)} Papers ===")
        for i, entry in enumerate(top_papers, 1):
            print(f"\n{i}. {entry.paper.title}")
            print(f"   Score: {entry.final_score:.3f} (relevance: {entry.relevance_score:.3f})")
            print(f"   Methods preview: {entry.methods_extracted[:300] if entry.methods_extracted else 'N/A'}...")
    
    asyncio.run(test())
