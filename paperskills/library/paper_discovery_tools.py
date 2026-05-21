#!/usr/bin/env python3
"""Paper Discovery Tools for Agent - Tools to search and extract skills from papers.

This module provides tools that can be registered in RTaskEvalEnv for agent use.
Agents can call these to search papers, extract content, and synthesize skills.

Usage (in agent context):
    result = search_papers("DESeq2 differential expression", source="pubmed")
    papers = json.loads(result)

    for paper in papers:
        content = fetch_paper_content(paper["pmid"])
        print(content)
"""

import json
from typing import Any, Dict, List, Optional

# Import our modules
from paperskills.library.paper_search import (
    PubMedSearcher, EuropePMCSearcher, BioRxivSearcher,
    PaperSearchAggregator, PaperMetadata
)
from paperskills.library.paper_extraction import (
    PaperProcessor, MethodSectionExtractor, CodeBlockExtractor
)
from paperskills.library.skill_synthesis import (
    SkillGenerator, SkillSynthesizer, SkillValidator,
    ExtractedSkill, TaskContext, validate_skill
)
from paperskills.library.query_generator import QueryGenerator, generate_search_query


def search_papers(query: str, source: str = "pubmed", max_results: int = 10) -> str:
    """Search for academic papers matching the query.

    Use this to find relevant papers for your analysis task.

    Args:
        query: Search query string. For best results, include tool names,
               analysis types, and data types (e.g., "DESeq2 apeglm RNA-seq").
        source: Which source to search ("pubmed", "europepmc", "biorxiv", or "all").
                Default is "pubmed".
        max_results: Maximum number of papers to return (default 10).

    Returns:
        JSON string with list of paper metadata including:
        - pmid: PubMed ID
        - pmcid: PMC ID (for full text access)
        - doi: DOI
        - title: Paper title
        - abstract: Abstract text
        - authors: List of authors
        - year: Publication year
        - has_fulltext: Whether full text is available
        - is_open_access: Whether paper is open access

    Example:
        result = search_papers("DESeq2 apeglm shrinkage", source="pubmed", max_results=5)
        papers = json.loads(result)
        # Returns papers about DESeq2 with apeglm shrinkage
    """
    try:
        aggregator = PaperSearchAggregator()

        if source == "all":
            papers = aggregator.search_all(query, max_per_source=max_results // 2)
        elif source == "pubmed":
            searcher = PubMedSearcher()
            papers = searcher.search(query, max_results=max_results)
        elif source == "europepmc":
            searcher = EuropePMCSearcher()
            papers = searcher.search(query, max_results=max_results, open_access_only=True)
        elif source == "biorxiv":
            searcher = BioRxivSearcher()
            papers = searcher.search(query, max_results=max_results)
        else:
            return json.dumps({
                "error": f"Unknown source: {source}",
                "valid_sources": ["pubmed", "europepmc", "biorxiv", "all"]
            }, indent=2)

        # Convert to dicts
        results = [p.to_dict() for p in papers]

        return json.dumps({
            "query": query,
            "source": source,
            "count": len(results),
            "papers": results,
        }, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "query": query,
            "source": source,
        }, indent=2)


def search_papers_for_task(task_family: str,
                           analysis_type: str = "",
                           tool_hint: str = "",
                           key_method: str = "",
                           source: str = "pubmed",
                           max_results: int = 10) -> str:
    """Search papers optimized for a specific task context.

    This automatically generates the best search query based on task parameters.

    Args:
        task_family: Task family ("rna", "methylation", "chipseq", "scrna", etc.)
        analysis_type: Type of analysis ("differential_expression", "peak_calling", etc.)
        tool_hint: Tool name hint ("DESeq2", "limma", etc.)
        key_method: Specific method ("apeglm", "duplicateCorrelation", etc.)
        source: Source to search ("pubmed", "europepmc", "biorxiv")
        max_results: Maximum results to return

    Returns:
        JSON string with search results and the generated query.

    Example:
        result = search_papers_for_task(
            task_family="rna",
            analysis_type="differential_expression",
            tool_hint="DESeq2",
            key_method="apeglm"
        )
    """
    try:
        # Create task context
        context = TaskContext(
            family=task_family,
            analysis_type=analysis_type,
            tool_hint=tool_hint,
            key_method=key_method,
        )

        # Generate optimized query
        query = generate_search_query(
            task_metadata=context.__dict__,
            source=source
        )

        # Perform search
        return search_papers(query, source=source, max_results=max_results)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "task_family": task_family,
            "analysis_type": analysis_type,
            "tool_hint": tool_hint,
        }, indent=2)


def fetch_paper_content(pmid: Optional[str] = None,
                       pmcid: Optional[str] = None,
                       doi: Optional[str] = None,
                       extract_methods: bool = True,
                       extract_code: bool = True) -> str:
    """Fetch and extract content from a paper.

    Provide at least one identifier (PMID, PMCID, or DOI).
    PMCID is preferred as it provides full text access.

    Args:
        pmid: PubMed ID
        pmcid: PMC ID (preferred for full text)
        doi: DOI
        extract_methods: Whether to extract Methods section (default True)
        extract_code: Whether to extract code snippets (default True)

    Returns:
        JSON string with extracted content including:
        - methods_text: Extracted Methods section
        - code_snippets: List of code snippets
        - full_text_length: Length of full text
        - success: Whether extraction succeeded

    Example:
        content = fetch_paper_content(pmcid="PMC1234567")
        data = json.loads(content)
        print(data["methods_text"][:500])
    """
    try:
        if not any([pmid, pmcid, doi]):
            return json.dumps({
                "error": "At least one identifier (pmid, pmcid, or doi) is required"
            }, indent=2)

        # Use PMCID if available
        identifier = pmcid or pmid or doi
        id_type = "pmcid" if pmcid else ("pmid" if pmid else "doi")

        # Create processor
        processor = PaperProcessor()

        # Build metadata
        paper_metadata = {"pmid": pmid, "pmcid": pmcid, "doi": doi}

        # Process paper
        if pmcid:
            content = processor.process_pmc_paper(pmcid)
        else:
            content = processor.process_paper(paper_metadata)

        # Check if we got anything
        if not content.full_text:
            return json.dumps({
                "error": "Failed to retrieve paper content",
                "identifier": identifier,
                "id_type": id_type,
            }, indent=2)

        result = {
            "success": True,
            "identifier": identifier,
            "id_type": id_type,
            "full_text_length": len(content.full_text),
        }

        if extract_methods:
            result["methods_text"] = content.methods_text
            result["methods_length"] = len(content.methods_text)

        if extract_code:
            result["code_snippets"] = content.code_snippets
            result["code_count"] = len(content.code_snippets)

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "pmid": pmid,
            "pmcid": pmcid,
            "doi": doi,
        }, indent=2)


def extract_skill_from_paper(pmid: Optional[str] = None,
                             pmcid: Optional[str] = None,
                             task_family: str = "",
                             analysis_type: str = "",
                             tool_hint: str = "",
                             key_method: str = "") -> str:
    """Extract a skill from a single paper.

    This fetches the paper, extracts content, and generates a structured skill.

    Args:
        pmid: PubMed ID
        pmcid: PMC ID (preferred)
        task_family: Task family context
        analysis_type: Analysis type context
        tool_hint: Tool name hint
        key_method: Specific method hint

    Returns:
        JSON string with extracted skill including:
        - skill: Skill details (name, tool, method_summary, etc.)
        - skill_md: Generated SKILL.md content
        - completeness_score: Quality score (0-1)
        - validation: Validation results

    Example:
        result = extract_skill_from_paper(
            pmcid="PMC1234567",
            task_family="rna",
            tool_hint="DESeq2",
            key_method="apeglm"
        )
    """
    try:
        if not any([pmid, pmcid]):
            return json.dumps({
                "error": "PMID or PMCID is required"
            }, indent=2)

        # Fetch paper content
        content_result = fetch_paper_content(
            pmid=pmid, pmcid=pmcid,
            extract_methods=True, extract_code=True
        )
        content_data = json.loads(content_result)

        if "error" in content_data:
            return content_result

        # Fetch metadata
        if pmcid:
            # Get metadata from PMC
            paper_metadata = PaperMetadata(
                pmid=pmid,
                pmcid=pmcid,
                source="pmc",
            )
        else:
            # Fetch from PubMed
            searcher = PubMedSearcher()
            papers = searcher.fetch_papers_by_pmids([pmid])
            if papers:
                paper_metadata = papers[0]
            else:
                paper_metadata = PaperMetadata(pmid=pmid, source="pubmed")

        # Create task context
        context = TaskContext(
            family=task_family,
            analysis_type=analysis_type,
            tool_hint=tool_hint,
            key_method=key_method,
        )

        # Create content object
        from paperskills.library.paper_extraction import ExtractedContent
        content = ExtractedContent(
            methods_text=content_data.get("methods_text", ""),
            code_snippets=content_data.get("code_snippets", []),
            full_text="",  # Not needed for skill generation
        )

        # Generate skill
        generator = SkillGenerator()
        skill = generator.generate(paper_metadata, content, context)

        # Validate
        is_valid, issues = validate_skill(skill)

        return json.dumps({
            "success": True,
            "pmid": pmid,
            "pmcid": pmcid,
            "skill": skill.to_dict(),
            "skill_md": skill.to_skill_md(),
            "completeness_score": skill.completeness_score,
            "validation": {
                "is_valid": is_valid,
                "issues": issues,
            },
        }, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "pmid": pmid,
            "pmcid": pmcid,
        }, indent=2)


def synthesize_skill_from_papers(paper_ids: List[str],
                                   task_family: str = "",
                                   analysis_type: str = "",
                                   tool_hint: str = "",
                                   key_method: str = "") -> str:
    """Synthesize a skill from multiple papers.

    This merges information from multiple papers to create a comprehensive skill.

    Args:
        paper_ids: List of PMC IDs or PMIDs (PMC IDs preferred, prefixed with "PMC")
        task_family: Task family context
        analysis_type: Analysis type context
        tool_hint: Tool name hint
        key_method: Specific method hint

    Returns:
        JSON string with synthesized skill including:
        - skill: Merged skill details
        - skill_md: Generated SKILL.md content
        - source_papers: List of papers used
        - completeness_score: Quality score

    Example:
        result = synthesize_skill_from_papers(
            paper_ids=["PMC1234567", "PMC1234568"],
            task_family="rna",
            tool_hint="DESeq2",
            key_method="apeglm"
        )
    """
    try:
        if not paper_ids:
            return json.dumps({
                "error": "At least one paper ID is required"
            }, indent=2)

        # Process each paper
        papers = []

        for paper_id in paper_ids:
            # Determine ID type
            if paper_id.startswith("PMC"):
                pmcid = paper_id
                pmid = None
            else:
                pmcid = None
                pmid = paper_id

            # Fetch content
            content_result = fetch_paper_content(
                pmid=pmid, pmcid=pmcid,
                extract_methods=True, extract_code=True
            )
            content_data = json.loads(content_result)

            if "error" in content_data:
                continue  # Skip failed papers

            # Fetch metadata
            if pmcid:
                paper_metadata = PaperMetadata(
                    pmid=content_data.get("pmid"),
                    pmcid=pmcid,
                    source="pmc",
                )
            else:
                searcher = PubMedSearcher()
                pmid_papers = searcher.fetch_papers_by_pmids([pmid])
                if pmid_papers:
                    paper_metadata = pmid_papers[0]
                else:
                    paper_metadata = PaperMetadata(pmid=pmid, source="pubmed")

            # Create content
            from paperskills.library.paper_extraction import ExtractedContent
            content = ExtractedContent(
                methods_text=content_data.get("methods_text", ""),
                code_snippets=content_data.get("code_snippets", []),
                full_text="",
            )

            papers.append((paper_metadata, content))

        if not papers:
            return json.dumps({
                "error": "Failed to process any of the provided papers",
                "paper_ids": paper_ids,
            }, indent=2)

        # Create context
        context = TaskContext(
            family=task_family,
            analysis_type=analysis_type,
            tool_hint=tool_hint,
            key_method=key_method,
        )

        # Synthesize
        synthesizer = SkillSynthesizer()
        skill = synthesizer.synthesize(papers, context)

        # Validate
        is_valid, issues = validate_skill(skill)

        return json.dumps({
            "success": True,
            "source_papers": paper_ids,
            "processed_count": len(papers),
            "skill": skill.to_dict(),
            "skill_md": skill.to_skill_md(),
            "completeness_score": skill.completeness_score,
            "validation": {
                "is_valid": is_valid,
                "issues": issues,
            },
        }, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "paper_ids": paper_ids,
        }, indent=2)


def validate_extracted_skill(skill_json: str) -> str:
    """Validate an extracted skill for completeness.

    Args:
        skill_json: JSON string of the skill to validate

    Returns:
        JSON string with validation results:
        - is_valid: Whether skill is valid
        - completeness_score: Quality score
        - issues: List of issues found
        - recommendations: Suggestions for improvement
    """
    try:
        skill_data = json.loads(skill_json)

        # Reconstruct skill object
        skill = ExtractedSkill(**skill_data)

        # Validate
        validator = SkillValidator()
        is_valid, issues = validator.validate(skill)

        # Generate recommendations
        recommendations = []

        if not skill.has_code:
            recommendations.append("Try to find papers with code in supplementary materials")

        if skill.completeness_score < 0.5:
            recommendations.append("Consider using multiple papers to improve coverage")

        if not skill.use_when:
            recommendations.append("Look for papers with clear usage guidelines")

        return json.dumps({
            "is_valid": is_valid,
            "completeness_score": skill.completeness_score,
            "issues": issues,
            "recommendations": recommendations,
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
        }, indent=2)


# Tool registry for easy import
PAPER_DISCOVERY_TOOLS = {
    "search_papers": search_papers,
    "search_papers_for_task": search_papers_for_task,
    "fetch_paper_content": fetch_paper_content,
    "extract_skill_from_paper": extract_skill_from_paper,
    "synthesize_skill_from_papers": synthesize_skill_from_papers,
    "validate_extracted_skill": validate_extracted_skill,
}


if __name__ == "__main__":
    # Example usage
    print("Testing Paper Discovery Tools...\n")

    # Test 1: Search papers
    print("Test 1: Search papers")
    result = search_papers("DESeq2", source="pubmed", max_results=3)
    data = json.loads(result)
    print(f"Found {data.get('count', 0)} papers")
    if data.get('papers'):
        print(f"First paper: {data['papers'][0].get('title', 'N/A')[:50]}...")
    print()

    # Test 2: Search for task
    print("Test 2: Search for specific task")
    result = search_papers_for_task(
        task_family="rna",
        tool_hint="DESeq2",
        key_method="apeglm",
        max_results=3
    )
    data = json.loads(result)
    print(f"Query: {data.get('query', 'N/A')}")
    print(f"Found {data.get('count', 0)} papers")
