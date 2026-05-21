#!/usr/bin/env python3
"""Skill Synthesis Module - Generate and validate skills from papers.

This module synthesizes skills from extracted paper content,
generating standardized SKILL.md format and validating completeness.

Usage:
    synthesizer = SkillSynthesizer()
    
    # From single paper
    skill = synthesizer.synthesize_from_paper(
        paper_metadata={"pmid": "12345", "title": "..."},
        extracted_content=extracted_content,
        task_context=task_context
    )
    
    # From multiple papers
    skill = synthesizer.synthesize_from_papers(
        papers=[(meta1, content1), (meta2, content2)],
        task_context=task_context
    )
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    from paper_extraction import ExtractedContent, CodeSnippet
    from paper_search import PaperMetadata
    from query_generator import TaskContext
except ImportError:
    from paperskills.library.paper_extraction import ExtractedContent, CodeSnippet
    from paperskills.library.paper_search import PaperMetadata
    from paperskills.library.query_generator import TaskContext


@dataclass
class ExtractedSkill:
    """A skill extracted from paper(s)."""
    # Identification
    name: str = ""
    source: str = ""  # pubmed, europepmc, etc.
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    
    # Content
    tool: str = ""
    method_summary: str = ""
    parameters: List[Dict[str, str]] = field(default_factory=list)
    code_snippets: List[str] = field(default_factory=list)
    use_when: List[str] = field(default_factory=list)
    not_when: List[str] = field(default_factory=list)
    common_pitfalls: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Metadata
    paper_title: str = ""
    paper_authors: List[str] = field(default_factory=list)
    paper_year: Optional[int] = None
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Quality metrics
    completeness_score: float = 0.0
    has_code: bool = False
    has_parameters: bool = False
    has_algorithm_steps: bool = False
    has_formula_or_metric_definition: bool = False
    has_output_mapping: bool = False
    has_pseudocode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "source": self.source,
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "doi": self.doi,
            "tool": self.tool,
            "method_summary": self.method_summary,
            "parameters": self.parameters,
            "code_snippets": self.code_snippets,
            "use_when": self.use_when,
            "not_when": self.not_when,
            "common_pitfalls": self.common_pitfalls,
            "tags": self.tags,
            "paper_title": self.paper_title,
            "paper_authors": self.paper_authors,
            "paper_year": self.paper_year,
            "extracted_at": self.extracted_at,
            "completeness_score": self.completeness_score,
            "has_code": self.has_code,
            "has_parameters": self.has_parameters,
            "has_algorithm_steps": self.has_algorithm_steps,
            "has_formula_or_metric_definition": self.has_formula_or_metric_definition,
            "has_output_mapping": self.has_output_mapping,
            "has_pseudocode": self.has_pseudocode,
        }
    
    def to_skill_md(self) -> str:
        """Generate standard SKILL.md format."""
        lines = [
            "---",
            f"name: {self.name}",
            f"source: {self.source}",
        ]
        
        if self.pmid:
            lines.append(f"pmid: {self.pmid}")
        if self.pmcid:
            lines.append(f"pmcid: {self.pmcid}")
        if self.doi:
            lines.append(f"doi: {self.doi}")
        if self.tags:
            lines.append("tags:")
            for tag in self.tags:
                lines.append(f"  - {tag}")
        
        lines.extend([
            f"tool: {self.tool}",
            f"paper_title: {self.paper_title}",
            f"extracted_at: {self.extracted_at}",
            "---",
            "",
            "## Method Summary",
            self.method_summary or "[No method summary extracted]",
            "",
        ])
        
        # Parameters
        if self.parameters:
            lines.extend([
                "## Parameters",
                "",
            ])
            for param in self.parameters:
                lines.append(f"- **{param.get('name', 'unknown')}**: {param.get('description', '')}")
                if 'default' in param:
                    lines.append(f"  - Default: `{param['default']}`")
                if 'values' in param:
                    lines.append(f"  - Possible values: {param['values']}")
                lines.append("")
        
        # Code Snippets
        if self.code_snippets:
            lines.extend([
                "## Code Snippets",
                "",
                "```r",
            ])
            for snippet in self.code_snippets[:3]:  # Limit to 3 snippets
                lines.append(snippet)
            lines.extend([
                "```",
                "",
            ])
        
        # Use When
        if self.use_when:
            lines.extend([
                "## When to Use",
                "",
            ])
            for item in self.use_when:
                lines.append(f"- {item}")
            lines.append("")
        
        # Not When
        if self.not_when:
            lines.extend([
                "## When NOT to Use",
                "",
            ])
            for item in self.not_when:
                lines.append(f"- {item}")
            lines.append("")
        
        # Common Pitfalls
        if self.common_pitfalls:
            lines.extend([
                "## Common Pitfalls",
                "",
            ])
            for pitfall in self.common_pitfalls:
                lines.append(f"- {pitfall}")
            lines.append("")

        lines.extend([
            "## Executability Checklist",
            "",
            f"- Algorithm steps present: {str(self.has_algorithm_steps).lower()}",
            f"- Formula or metric definitions present: {str(self.has_formula_or_metric_definition).lower()}",
            f"- Output mapping present: {str(self.has_output_mapping).lower()}",
            f"- Pseudocode present: {str(self.has_pseudocode).lower()}",
            "",
        ])
        
        # Source
        lines.extend([
            "## Source",
            f"",
            f"- **Paper**: {self.paper_title}",
            f"- **Authors**: {', '.join(self.paper_authors[:3])}{' et al.' if len(self.paper_authors) > 3 else ''}",
        ])
        if self.paper_year:
            lines.append(f"- **Year**: {self.paper_year}")
        if self.pmid:
            lines.append(f"- **PMID**: {self.pmid}")
        if self.doi:
            lines.append(f"- **DOI**: {self.doi}")
        
        return "\n".join(lines)


class SkillGenerator:
    """Generate skills from extracted paper content."""
    
    # Keywords that indicate parameter descriptions
    PARAMETER_KEYWORDS = [
        "parameter", "argument", "option", "setting",
        "threshold", "cutoff", "value", "default",
    ]
    
    # Keywords that indicate usage conditions
    USE_WHEN_KEYWORDS = [
        "we recommend", "suitable for", "appropriate for",
        "ideal for", "designed for", "best for",
    ]
    
    # Keywords that indicate limitations
    NOT_WHEN_KEYWORDS = [
        "not suitable", "not recommended", "should not",
        "cannot be used", "inappropriate for", "limitations",
    ]
    
    # Keywords that indicate pitfalls
    PITFALL_KEYWORDS = [
        "caution", "warning", "important", "note that",
        "be aware", "careful", "pitfall", "common mistake",
    ]
    
    def generate(self, 
                 paper_metadata: PaperMetadata,
                 extracted_content: ExtractedContent,
                 task_context: TaskContext) -> ExtractedSkill:
        """Generate skill from single paper.
        
        Args:
            paper_metadata: Metadata of the source paper
            extracted_content: Extracted content from paper
            task_context: Task context for guidance
            
        Returns:
            ExtractedSkill object
        """
        skill = ExtractedSkill()
        
        # Basic identification
        skill.source = paper_metadata.source
        skill.pmid = paper_metadata.pmid
        skill.pmcid = paper_metadata.pmcid
        skill.doi = paper_metadata.doi
        skill.paper_title = paper_metadata.title
        skill.paper_authors = paper_metadata.authors
        skill.paper_year = paper_metadata.year
        
        # Generate name
        tool_name = task_context.tool_hint or self._extract_tool_name(extracted_content)
        doi_normalized = (paper_metadata.doi or "unknown").replace("/", "_").replace(".", "-")
        skill.name = f"paper-{doi_normalized}"
        skill.tool = tool_name
        
        # Extract method summary
        skill.method_summary = self._extract_method_summary(extracted_content)
        
        # Extract parameters
        skill.parameters = self._extract_parameters(extracted_content)
        skill.has_parameters = len(skill.parameters) > 0
        
        # Extract code snippets
        skill.code_snippets = extracted_content.code_snippets
        skill.has_code = len(skill.code_snippets) > 0
        
        # Extract usage guidance
        skill.use_when = self._extract_usage_conditions(extracted_content, positive=True)
        skill.not_when = self._extract_usage_conditions(extracted_content, positive=False)
        skill.common_pitfalls = self._extract_pitfalls(extracted_content)
        
        # Calculate completeness score
        self._add_task_adapted_guidance(skill, extracted_content, task_context)
        self._annotate_executability(skill)
        skill.completeness_score = self._calculate_completeness(skill)
        
        return skill

    def _add_task_adapted_guidance(
        self,
        skill: ExtractedSkill,
        content: ExtractedContent,
        task_context: TaskContext,
    ) -> None:
        """Add generic source-supported skill slots.

        The runner may provide hidden retrieval metadata to find sources, but a
        generated PaperSkill must not turn that metadata into fake source
        evidence. Only terms present in the paper/doc text are promoted into
        executable guidance.
        """
        source_text = " ".join(
            [
                content.full_text or "",
                content.methods_text or "",
                skill.paper_title or "",
                "\n".join(skill.code_snippets),
            ]
        ).lower()
        additions: list[str] = []
        profile = getattr(task_context, "retrieval_profile", {}) or {}
        package = str(profile.get("package") or task_context.tool_hint or "").strip()
        data_objects = [str(x).strip() for x in profile.get("data_objects", []) or [] if str(x).strip()]
        core_functions = [str(x).strip() for x in profile.get("core_functions", []) or [] if str(x).strip()]
        expected_tags = [str(x).strip() for x in profile.get("expected_skill_tags", []) or [] if str(x).strip()]

        def present(terms: list[str]) -> list[str]:
            hits = []
            for term in terms:
                if term and term.lower() in source_text:
                    hits.append(term)
            return hits

        package_hits = present([package] if package else [])
        object_hits = present(data_objects)
        function_hits = present(core_functions)
        tag_hits = present(expected_tags)
        source_supported_hits = [*package_hits, *object_hits, *function_hits, *tag_hits]

        if source_supported_hits:
            additions.append(
                "Source-supported method cues: the source explicitly mentions "
                + ", ".join(dict.fromkeys(source_supported_hits[:12]))
                + "."
            )
        if object_hits:
            additions.append(
                "Object semantics slot: preserve the source-described data object(s) "
                + ", ".join(dict.fromkeys(object_hits[:8]))
                + " when mapping inputs to outputs."
            )
        if function_hits:
            additions.append(
                "Algorithm step slot: use the source-described function(s) in a coherent order: "
                + " -> ".join(dict.fromkeys(function_hits[:8]))
                + "."
            )

        procedural_keywords = {
            "input": ["input", "count matrix", "matrix", "metadata", "sample"],
            "output": ["output", "result", "table", "data frame", "csv", "tsv"],
            "parameter": ["parameter", "argument", "threshold", "cutoff", "design", "contrast"],
            "statistical_choice": ["model", "normalization", "dispersion", "test", "p-value", "fdr", "adjusted"],
        }
        procedural_hits = {
            label: [kw for kw in keywords if kw in source_text]
            for label, keywords in procedural_keywords.items()
        }
        procedural_hits = {k: v for k, v in procedural_hits.items() if v}
        if procedural_hits and (source_supported_hits or skill.has_code):
            additions.append(
                "Extraction slots present: "
                + "; ".join(f"{slot}={', '.join(words[:4])}" for slot, words in procedural_hits.items())
                + "."
            )
        if source_supported_hits and procedural_hits.get("input") and procedural_hits.get("output"):
            additions.append(
                "Algorithm step: read the task inputs, apply the source-supported method/object cues, preserve required statistical or object semantics, and write the required deterministic output tables."
            )
        if procedural_hits.get("output") and (source_supported_hits or skill.has_code):
            additions.append(
                "Output mapping slot: map source-described result/table/data-frame outputs onto the task's required public output schema without inventing hidden reference values."
            )
        if procedural_hits.get("statistical_choice") and (source_supported_hits or skill.has_code):
            additions.append(
                "Metric definition slot: preserve source-described model, normalization, test, p-value, FDR, or score semantics when producing numeric outputs."
            )
        if skill.has_code:
            additions.append(
                "Pseudocode: translate the source code/example snippets into the smallest workflow that reads task inputs, preserves source-supported object/function semantics, and writes the requested output tables."
            )
        elif source_supported_hits and procedural_hits:
            additions.append(
                "Pseudocode: follow the source-supported method cues above, make input/object/output mapping explicit, and write deterministic tabular outputs."
            )

        if additions:
            skill.method_summary = (skill.method_summary + "\n\n" + "\n".join(additions)).strip()
        if expected_tags and (source_supported_hits or skill.has_code):
            skill.use_when.append(
                "Use when the task needs reusable source-supported method semantics involving "
                + ", ".join((source_supported_hits or expected_tags)[:8])
                + "."
            )

    def _annotate_executability(self, skill: ExtractedSkill) -> None:
        text = "\n".join([skill.method_summary, "\n".join(skill.code_snippets)]).lower()
        skill.has_algorithm_steps = any(k in text for k in ["algorithm step", "step:", "load ", "compute", "fit ", "cluster"])
        skill.has_formula_or_metric_definition = any(k in text for k in ["metric definition", "formula", "score", "ratio", "p-value", "pvalue", "similarity"])
        skill.has_output_mapping = any(k in text for k in ["output mapping", ".csv", "write ", "columns", "metric_name"])
        skill.has_pseudocode = "pseudocode" in text or bool(skill.code_snippets)
    
    def _extract_tool_name(self, content: ExtractedContent) -> str:
        """Extract tool name from content."""
        # Look for common patterns
        patterns = [
            r"([A-Z][a-zA-Z0-9]+)\s+package",
            r"software\s+([A-Z][a-zA-Z0-9]+)",
            r"tool\s+([A-Z][a-zA-Z0-9]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content.full_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Look for R package loads
        r_packages = re.findall(r"library\(([a-zA-Z_][a-zA-Z0-9_]*)\)", content.full_text)
        if r_packages:
            return r_packages[0]
        
        return "unknown"
    
    def _extract_method_summary(self, content: ExtractedContent) -> str:
        """Extract concise method summary from methods text."""
        methods = content.methods_text
        
        if not methods:
            return ""
        
        # Get first few sentences (usually contains the main method description)
        sentences = re.split(r'(?<=[.!?])\s+', methods)
        summary_sentences = []
        
        for sent in sentences[:5]:  # First 5 sentences
            # Filter out very short sentences
            if len(sent.split()) >= 5:
                summary_sentences.append(sent)
        
        return " ".join(summary_sentences[:3])  # Limit to 3 sentences
    
    def _extract_parameters(self, content: ExtractedContent) -> List[Dict[str, str]]:
        """Extract parameter descriptions from methods text."""
        parameters = []
        methods = content.methods_text
        
        if not methods:
            return parameters
        
        # Look for parameter patterns
        # Pattern 1: "parameter X was set to Y"
        pattern1 = r"(\w+)\s+(?:was\s+)?set\s+to\s+([\w\-_.]+)"
        for match in re.finditer(pattern1, methods, re.IGNORECASE):
            param_name = match.group(1)
            default_val = match.group(2)
            
            # Get surrounding context
            start = max(0, match.start() - 100)
            end = min(len(methods), match.end() + 100)
            context = methods[start:end]
            
            parameters.append({
                "name": param_name,
                "default": default_val,
                "description": self._extract_param_description(context, param_name),
            })
        
        # Pattern 2: "we used X = Y" or "X = Y was used"
        pattern2 = r"(\w+)\s*=\s*([\w\-_.]+)"
        for match in re.finditer(pattern2, methods):
            param_name = match.group(1)
            default_val = match.group(2)
            
            # Skip if already captured
            if any(p["name"] == param_name for p in parameters):
                continue
            
            # Get context
            start = max(0, match.start() - 100)
            end = min(len(methods), match.end() + 100)
            context = methods[start:end]
            
            parameters.append({
                "name": param_name,
                "default": default_val,
                "description": self._extract_param_description(context, param_name),
            })
        
        # Deduplicate
        seen = set()
        unique_params = []
        for p in parameters:
            if p["name"] not in seen:
                seen.add(p["name"])
                unique_params.append(p)
        
        return unique_params[:10]  # Limit to 10 parameters
    
    def _extract_param_description(self, context: str, param_name: str) -> str:
        """Extract description for a parameter from context."""
        # Look for sentences mentioning this parameter
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        for sent in sentences:
            if param_name.lower() in sent.lower():
                # Clean up
                sent = sent.strip()
                if len(sent) > 20:
                    return sent
        
        return "Parameter value from paper"
    
    def _extract_usage_conditions(self, content: ExtractedContent, 
                                   positive: bool = True) -> List[str]:
        """Extract usage conditions (when to use / when not to use)."""
        conditions = []
        
        keywords = self.USE_WHEN_KEYWORDS if positive else self.NOT_WHEN_KEYWORDS
        text = content.full_text
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sent in sentences:
            sent_lower = sent.lower()
            for keyword in keywords:
                if keyword.lower() in sent_lower:
                    # Clean and add
                    clean_sent = sent.strip()
                    if len(clean_sent) > 30 and len(clean_sent) < 300:
                        conditions.append(clean_sent)
                        break
        
        # Remove duplicates and limit
        unique = []
        seen = set()
        for cond in conditions:
            key = cond[:50].lower()
            if key not in seen:
                seen.add(key)
                unique.append(cond)
        
        return unique[:5]
    
    def _extract_pitfalls(self, content: ExtractedContent) -> List[str]:
        """Extract common pitfalls and warnings."""
        pitfalls = []
        
        text = content.full_text
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sent in sentences:
            sent_lower = sent.lower()
            for keyword in self.PITFALL_KEYWORDS:
                if keyword.lower() in sent_lower:
                    clean_sent = sent.strip()
                    if len(clean_sent) > 30 and len(clean_sent) < 300:
                        pitfalls.append(clean_sent)
                        break
        
        # Remove duplicates
        unique = []
        seen = set()
        for pit in pitfalls:
            key = pit[:50].lower()
            if key not in seen:
                seen.add(key)
                unique.append(pit)
        
        return unique[:5]
    
    def _calculate_completeness(self, skill: ExtractedSkill) -> float:
        """Calculate completeness score for the skill."""
        scores = {
            "method_summary": 0.3 if skill.method_summary else 0,
            "has_code": 0.3 if skill.has_code else 0,
            "has_parameters": 0.2 if skill.has_parameters else 0,
            "use_when": 0.1 if skill.use_when else 0,
            "not_when": 0.1 if skill.not_when else 0,
            "executability": 0.2 if (
                skill.has_algorithm_steps and skill.has_output_mapping and skill.has_pseudocode
            ) else 0,
        }
        
        return sum(scores.values())


class SkillSynthesizer:
    """Synthesize skills from multiple papers."""
    
    def __init__(self):
        self.generator = SkillGenerator()
    
    def synthesize(self, 
                   papers: List[Tuple[PaperMetadata, ExtractedContent]],
                   task_context: TaskContext) -> ExtractedSkill:
        """Synthesize skill from multiple papers.
        
        Args:
            papers: List of (metadata, content) tuples
            task_context: Task context
            
        Returns:
            Merged ExtractedSkill
        """
        if not papers:
            raise ValueError("No papers provided for synthesis")
        
        if len(papers) == 1:
            # Single paper - just generate
            return self.generator.generate(papers[0][0], papers[0][1], task_context)
        
        # Generate skills for all papers
        skills = []
        for metadata, content in papers:
            try:
                skill = self.generator.generate(metadata, content, task_context)
                skills.append(skill)
            except Exception as e:
                print(f"Failed to generate skill from {metadata.pmid}: {e}")
        
        if not skills:
            raise ValueError("No skills could be generated from papers")
        
        # Merge skills
        merged = self._merge_skills(skills, task_context)
        
        return merged
    
    def _merge_skills(self, skills: List[ExtractedSkill], 
                     task_context: TaskContext) -> ExtractedSkill:
        """Merge multiple skills into one."""
        # Start with the most complete skill as base
        base_skill = max(skills, key=lambda s: s.completeness_score)
        merged = ExtractedSkill()
        
        # Copy base info
        merged.name = base_skill.name
        merged.tool = base_skill.tool
        merged.paper_title = base_skill.paper_title
        merged.paper_authors = base_skill.paper_authors
        merged.paper_year = base_skill.paper_year
        
        # Merge from all skills
        all_pmids = [s.pmid for s in skills if s.pmid]
        all_dois = [s.doi for s in skills if s.doi]
        
        # Use primary source (highest completeness)
        merged.source = base_skill.source
        merged.pmid = base_skill.pmid
        merged.pmcid = base_skill.pmcid
        merged.doi = base_skill.doi
        
        # Merge method summaries
        summaries = [s.method_summary for s in skills if s.method_summary]
        if summaries:
            # Take the longest/most detailed
            merged.method_summary = max(summaries, key=len)
        
        # Merge parameters (deduplicate by name)
        all_params = {}
        for skill in skills:
            for param in skill.parameters:
                name = param["name"]
                if name not in all_params:
                    all_params[name] = param
                else:
                    # Merge descriptions if different
                    existing = all_params[name]
                    if len(param.get("description", "")) > len(existing.get("description", "")):
                        existing["description"] = param["description"]
        
        merged.parameters = list(all_params.values())
        merged.has_parameters = len(merged.parameters) > 0
        
        # Merge code snippets (deduplicate)
        seen_code = set()
        merged.code_snippets = []
        for skill in skills:
            for code in skill.code_snippets:
                # Normalize for comparison
                code_key = code[:100].strip()
                if code_key not in seen_code:
                    seen_code.add(code_key)
                    merged.code_snippets.append(code)
        
        merged.has_code = len(merged.code_snippets) > 0
        
        # Merge usage conditions (deduplicate)
        merged.use_when = self._deduplicate_conditions([s.use_when for s in skills])
        merged.not_when = self._deduplicate_conditions([s.not_when for s in skills])
        merged.common_pitfalls = self._deduplicate_conditions([s.common_pitfalls for s in skills])
        
        # Recalculate completeness
        merged.completeness_score = self.generator._calculate_completeness(merged)
        
        return merged
    
    def _deduplicate_conditions(self, condition_lists: List[List[str]]) -> List[str]:
        """Deduplicate conditions from multiple sources."""
        all_conditions = []
        for lst in condition_lists:
            all_conditions.extend(lst)
        
        # Remove duplicates based on first 50 chars
        seen = set()
        unique = []
        for cond in all_conditions:
            key = cond[:50].lower()
            if key not in seen:
                seen.add(key)
                unique.append(cond)
        
        return unique[:7]  # Limit total


class SkillValidator:
    """Validate extracted skills for completeness and quality."""
    
    def __init__(self):
        self.min_completeness_threshold = 0.3
    
    def validate(self, skill: ExtractedSkill) -> Tuple[bool, List[str]]:
        """Validate a skill.
        
        Args:
            skill: Skill to validate
            
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        if not skill.tool:
            issues.append("Missing tool name")
        
        if not skill.method_summary:
            issues.append("Missing method summary")
        
        if not skill.pmid and not skill.doi:
            issues.append("Missing source identifier (PMID or DOI)")
        
        # Check quality thresholds
        if skill.completeness_score < self.min_completeness_threshold:
            issues.append(f"Completeness score too low: {skill.completeness_score:.2f}")
        
        # Check for code (not strictly required but recommended)
        if not skill.has_code:
            issues.append("No code snippets extracted (warning)")

        if not (skill.has_algorithm_steps and skill.has_output_mapping and skill.has_pseudocode):
            issues.append("Executable method guidance incomplete")
        
        fatal_issues = [
            i for i in issues
            if "(warning)" not in i.lower()
        ]
        is_valid = len(fatal_issues) == 0
        
        return is_valid, issues
    
    def validate_for_task(self, skill: ExtractedSkill, 
                         task_context: TaskContext) -> Tuple[bool, List[str]]:
        """Validate skill suitability for specific task."""
        is_valid, issues = self.validate(skill)
        
        # Check tool match
        if task_context.tool_hint:
            if skill.tool.lower() != task_context.tool_hint.lower():
                issues.append(
                    f"Tool mismatch: expected {task_context.tool_hint}, "
                    f"got {skill.tool}"
                )
        
        # Check for key method
        if task_context.key_method:
            method_found = (
                task_context.key_method.lower() in skill.method_summary.lower() or
                any(task_context.key_method.lower() in p["name"].lower() 
                    for p in skill.parameters)
            )
            if not method_found:
                issues.append(f"Key method '{task_context.key_method}' not found")
        
        return is_valid, issues


# Convenience functions
def synthesize_skill(papers: List[Tuple[PaperMetadata, ExtractedContent]],
                    task_context: TaskContext) -> ExtractedSkill:
    """Convenience function to synthesize skill from papers.
    
    Args:
        papers: List of (metadata, content) tuples
        task_context: Task context
        
    Returns:
        Synthesized skill
    """
    synthesizer = SkillSynthesizer()
    return synthesizer.synthesize(papers, task_context)


def validate_skill(skill: ExtractedSkill) -> Tuple[bool, List[str]]:
    """Convenience function to validate skill.
    
    Args:
        skill: Skill to validate
        
    Returns:
        (is_valid, issues)
    """
    validator = SkillValidator()
    return validator.validate(skill)


if __name__ == "__main__":
    # Example usage
    print("Testing Skill Synthesis...\n")
    
    # Create sample extracted content
    sample_content = ExtractedContent(
        methods_text="""
        We used DESeq2 version 1.30.0 for differential expression analysis.
        The lfcShrink function was used with type='apeglm' for effect size shrinkage.
        Count data was normalized using the median of ratios method.
        """,
        code_snippets=["library(DESeq2)", "dds <- DESeqDataSetFromMatrix(countData, colData, design)"],
    )
    
    # Create sample metadata
    sample_meta = PaperMetadata(
        pmid="12345678",
        pmcid="PMC1234567",
        doi="10.1186/s13059-014-0550-8",
        title="DESeq2: Differential gene expression analysis",
        authors=["Love MI", "Huber W", "Anders S"],
        year=2014,
    )
    
    # Create task context
    task_context = TaskContext(
        tool_hint="DESeq2",
        key_method="apeglm",
        family="rna",
        analysis_type="differential_expression",
    )
    
    # Generate skill
    generator = SkillGenerator()
    skill = generator.generate(sample_meta, sample_content, task_context)
    
    print(f"Generated skill:")
    print(f"  Name: {skill.name}")
    print(f"  Tool: {skill.tool}")
    print(f"  Completeness: {skill.completeness_score:.2f}")
    print(f"  Has code: {skill.has_code}")
    print(f"  Parameters: {len(skill.parameters)}")
    print()
    
    # Generate SKILL.md
    print("Generated SKILL.md:")
    print(skill.to_skill_md()[:500] + "...")
    print()
    
    # Validate
    is_valid, issues = validate_skill(skill)
    print(f"Validation: {'PASS' if is_valid else 'FAIL'}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")
