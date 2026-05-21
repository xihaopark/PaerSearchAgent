"""PaperToSkill preflight agent extracted from the experiment runner.

This module is a cleaned public extraction of the `paper_to_skill` and
`paper_to_skill_v2` branches from
`main/paper_primary_benchmark/ldp_r_task_eval/run_unified_paper_experiment.py`.
It intentionally keeps the same preflight shape:

1. Build task retrieval context from a registry/task dictionary.
2. Run v1 iterative paper retrieval or v2 technical-doc planning.
3. Persist/copy generated skills into the task workspace.
4. Format the generated skill block for the downstream ReAct coding agent.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperskills.agents.prompts import (
    PAPER_ITERATIVE_PROMPT,
    PAPER_TO_SKILL_V2_PROMPT,
    REGISTRY_SKILL_BLOCK,
)
from paperskills.library.iterative_paper_retrieval import (
    exact_retrieve_paper_skill,
    iterative_retrieve_papers,
)
from paperskills.library.persistent_skill_library import (
    PersistentPaperSkillLibrary,
    skill_quality_from_markdown,
)
from paperskills.v2 import PaperToSkillV2Builder


def _registry_paper_hint(task: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("primary_doi", "primary_paper_title"):
        if task.get(key):
            parts.append(str(task[key]))
    for val in task.get("search_terms", []) or []:
        parts.append(str(val))
    key_methods = task.get("key_methods") or task.get("key_method")
    if isinstance(key_methods, list):
        parts.extend(str(x) for x in key_methods)
    elif key_methods:
        parts.append(str(key_methods))
    if not parts:
        for key in ("family", "name", "id"):
            if task.get(key):
                parts.append(str(task[key]))
    return " ".join(dict.fromkeys(p.strip() for p in parts if p and p.strip()))[:900]


def _registry_analysis_type(task: dict[str, Any]) -> str:
    if task.get("analysis_type"):
        return str(task["analysis_type"])
    families = [str(x).lower() for x in task.get("families", []) or []]
    text = " ".join([task.get("id", ""), task.get("name", ""), task.get("family", ""), *families]).lower()
    if "quality" in text or "metric" in text or "spatial" in text:
        return "quality_control"
    if "trajectory" in text or "pseudotime" in text:
        return "pseudotime"
    if "regulatory" in text or "regulon" in text:
        return "regulatory_module"
    if "methyl" in text:
        return "methylation_analysis"
    if "deseq" in text or "rna" in text:
        return "differential_expression"
    return "method"


def _registry_retrieval_profile(task: dict[str, Any]) -> dict[str, Any]:
    """Hidden-method-aware profile extraction used by the experiment runner.

    Explicit `retrieval_profile` wins. Otherwise this infers package/method
    profiles from task ids and source-script paths without requiring the task
    prompt itself to reveal the method.
    """
    if task.get("retrieval_profile"):
        return dict(task["retrieval_profile"])

    text = " ".join([
        str(task.get("id", "")),
        str(task.get("name", "")),
        str(task.get("family", "")),
        str(task.get("r_script_src", "")),
    ]).lower()

    common_negative = [
        "disease application",
        "case study",
        "broad review",
        "association study",
    ]

    if "methylkit" in text or "methyl" in text:
        return {
            "package": "methylKit",
            "language": "R",
            "ecosystem": "Bioconductor",
            "analysis_family": "DNA methylation",
            "data_objects": ["methylRaw", "methylRawList", "methylBase", "methylBaseDB"],
            "core_functions": ["methRead", "filterByCoverage", "normalizeCoverage", "unite", "getData", "percMethylation"],
            "method_terms": ["bisulfite sequencing", "DNA methylation analysis", "differential methylation"],
            "search_terms": [
                "methylKit R package DNA methylation analysis Genome Biology",
                "methylKit Bioconductor methylRaw methylBase methylBaseDB",
                "methylKit getData percMethylation data frame",
            ],
            "preferred_source_types": ["software paper", "package paper", "Bioconductor manual", "vignette", "workflow"],
            "negative_source_types": common_negative,
            "expected_skill_tags": ["methylKit", "methylBase", "methylRaw", "coverage", "methylated count", "methylation percentage"],
        }

    if "deseq" in text:
        return {
            "package": "DESeq2",
            "language": "R",
            "ecosystem": "Bioconductor",
            "analysis_family": "RNA-seq differential expression",
            "data_objects": ["DESeqDataSet", "DESeqResults", "SummarizedExperiment"],
            "core_functions": ["DESeqDataSetFromMatrix", "DESeq", "results", "counts", "estimateSizeFactors", "vst"],
            "method_terms": ["negative binomial model", "size factor normalization", "differential expression"],
            "search_terms": [
                "DESeq2 Bioconductor differential expression RNA-seq method",
                "DESeq2 moderated estimation fold change dispersion RNA-seq",
                "DESeq2 DESeqDataSet results normalized counts",
            ],
            "preferred_source_types": ["software paper", "method paper", "Bioconductor manual", "vignette", "workflow"],
            "negative_source_types": common_negative,
            "expected_skill_tags": ["DESeq2", "DESeqDataSet", "size factors", "normalized counts", "differential expression"],
        }

    if "limma" in text or "voom" in text:
        return {
            "package": "limma",
            "language": "R",
            "ecosystem": "Bioconductor",
            "analysis_family": "RNA-seq differential expression",
            "data_objects": ["DGEList", "EList", "design matrix", "MArrayLM"],
            "core_functions": ["voom", "lmFit", "eBayes", "topTable", "removeBatchEffect"],
            "method_terms": ["linear models", "empirical Bayes", "voom precision weights", "differential expression"],
            "search_terms": [
                "limma Bioconductor software package differential expression",
                "limma voom RNA-seq differential expression workflow",
                "limma lmFit eBayes topTable model matrix",
            ],
            "preferred_source_types": ["software paper", "method paper", "Bioconductor manual", "vignette", "workflow"],
            "negative_source_types": common_negative,
            "expected_skill_tags": ["limma", "voom", "lmFit", "eBayes", "topTable", "model matrix"],
        }

    if "edger" in text:
        return {
            "package": "edgeR",
            "language": "R",
            "ecosystem": "Bioconductor",
            "analysis_family": "RNA-seq normalization",
            "data_objects": ["DGEList"],
            "core_functions": ["DGEList", "calcNormFactors", "cpm"],
            "method_terms": ["TMM normalization", "RNA-seq count normalization"],
            "search_terms": [
                "edgeR Bioconductor RNA-seq count normalization",
                "edgeR TMM normalization DGEList calcNormFactors",
            ],
            "preferred_source_types": ["software paper", "method paper", "Bioconductor manual", "vignette", "workflow"],
            "negative_source_types": common_negative,
            "expected_skill_tags": ["edgeR", "DGEList", "TMM", "normalized counts"],
        }

    if "phantompeak" in text or "crosscorr" in text:
        return {
            "package": "phantompeakqualtools",
            "language": "R",
            "analysis_family": "ChIP-seq quality control",
            "data_objects": ["strand cross-correlation", "NSC", "RSC"],
            "core_functions": ["phantompeakqualtools", "strand cross-correlation"],
            "method_terms": ["ChIP-seq quality control", "strand cross-correlation", "phantom peak"],
            "search_terms": [
                "phantompeakqualtools ChIP-seq strand cross-correlation quality control",
                "ChIP-seq phantom peak quality metrics NSC RSC",
            ],
            "preferred_source_types": ["software paper", "method paper", "workflow", "protocol"],
            "negative_source_types": common_negative,
            "expected_skill_tags": ["ChIP-seq", "cross-correlation", "NSC", "RSC", "quality control"],
        }

    if "macs" in text:
        return {
            "package": "MACS2",
            "language": "R",
            "analysis_family": "ChIP-seq peak calling QC",
            "data_objects": ["peak table", "FRiP", "peak count"],
            "core_functions": ["MACS2", "callpeak"],
            "method_terms": ["model-based ChIP-seq peak calling", "peak quality control"],
            "search_terms": [
                "MACS2 model-based analysis ChIP-seq peak calling",
                "MACS2 ChIP-seq peak calling quality control",
            ],
            "preferred_source_types": ["software paper", "method paper", "workflow", "protocol"],
            "negative_source_types": common_negative,
            "expected_skill_tags": ["MACS2", "ChIP-seq", "peak calling", "peak count", "quality control"],
        }

    if "homer" in text or "annotatepeaks" in text:
        return {
            "package": "HOMER",
            "analysis_family": "ChIP-seq peak annotation",
            "data_objects": ["annotated peak table", "genomic annotation"],
            "core_functions": ["annotatePeaks.pl"],
            "method_terms": ["peak annotation", "motif discovery", "genomic annotation"],
            "search_terms": [
                "HOMER annotatePeaks ChIP-seq peak annotation",
                "HOMER motif discovery genomic annotation software",
            ],
            "preferred_source_types": ["software paper", "method paper", "manual", "workflow", "protocol"],
            "negative_source_types": common_negative,
            "expected_skill_tags": ["HOMER", "annotatePeaks", "peak annotation", "genomic annotation"],
        }

    return {}


def _paper_is_formal_skill(paper: dict[str, Any], *, base_dir: Path | None = None) -> bool:
    validation = paper.get("skill_validation") or {}
    if validation.get("is_valid") is False:
        return False
    if paper.get("low_confidence"):
        return False
    source_access_level = str(paper.get("source_access_level") or "")
    if paper.get("formal_source_valid") is False:
        return False
    if source_access_level in {"abstract_only", "metadata_only", "unavailable", "documentation_fulltext"}:
        return False
    if source_access_level and source_access_level not in {
        "publisher_pdf_fulltext",
        "pmc_pdf_fulltext",
        "pmc_xml_fulltext",
        "europepmc_xml_fulltext",
        "html_fulltext",
    }:
        return False
    if paper.get("formal_skill_valid") is False:
        return False
    if "paper_fulltext_skill" in paper and not paper.get("paper_fulltext_skill"):
        return False
    skill_path = Path(str(paper.get("skill_path") or ""))
    if base_dir and not skill_path.is_absolute():
        skill_path = base_dir / skill_path
    if not skill_path.is_file() or skill_path.stat().st_size <= 0:
        return False
    quality = paper.get("skill_quality")
    if not isinstance(quality, dict):
        quality = skill_quality_from_markdown(skill_path.read_text(encoding="utf-8", errors="replace"))
        paper["skill_quality"] = quality
    return not quality.get("metadata_only_or_generic")


def _format_registry_skill_block(preflight: dict[str, Any]) -> str:
    papers = [
        p for p in preflight.get("top_papers", []) or []
        if p.get("snapshot_skill_path") or _paper_is_formal_skill(p)
    ]
    if not papers:
        return "No runtime paper skills were generated."
    chunks = []
    for i, paper in enumerate(papers, start=1):
        skill_md = paper.get("skill_md") or ""
        header = (
            f"## Runtime Paper Skill {i}\n"
            f"- title: {paper.get('title')}\n"
            f"- pmid: {paper.get('pmid')}\n"
            f"- pmcid: {paper.get('pmcid')}\n"
            f"- doi: {paper.get('doi')}\n"
            f"- skill_path: {paper.get('skill_path')}\n"
            f"- low_confidence: {paper.get('low_confidence')}\n"
            f"- tags: {', '.join(paper.get('skill_tags') or [])}\n\n"
        )
        chunks.append(header + skill_md[:12000])
    return "\n\n".join(chunks)


@dataclass
class PreparedPaperToSkillRun:
    """Preflight artifact passed to the downstream ReAct coding agent."""

    prompt: str
    env: Any
    preflight: dict[str, Any]
    generated_skill_count: int


class PaperToSkillAgent:
    """Prepare v1/v2 PaperToSkill runs exactly like the experiment runner."""

    def __init__(self, library_root: str | Path | None = None, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
        self.library_root = Path(library_root).resolve() if library_root else self.repo_root / "paperskills" / "library"

    async def prepare_v1(
        self,
        task: dict[str, Any],
        *,
        work_dir: Path,
        idx_dir: Path,
        objective_file: Path,
        success_artifact_glob: str,
        max_steps: int = 40,
        retrieval_policy: str = "query_only",
        paper_skill_library_mode: str = "online_only",
        paper_skill_extractor: str = "hybrid",
    ) -> PreparedPaperToSkillRun:
        from paperskills.agents.live_paper_discovery_runner import LivePaperDiscoveryEnv

        runtime_skill_dir = idx_dir / "runtime_paper_skills"
        paper_preflight = None
        library = PersistentPaperSkillLibrary(self.library_root, repo_root=self.repo_root)
        if paper_skill_library_mode == "reuse":
            paper_preflight = self._persistent_reuse_preflight(task, library=library)
        if paper_preflight is None:
            paper_preflight = await self._run_registry_paper_preflight(
                task,
                runtime_skill_dir=runtime_skill_dir,
                retrieval_policy=retrieval_policy,
                paper_skill_extractor=paper_skill_extractor,
            )
            paper_preflight.setdefault("source", "registry_retrieval_generated")
        paper_preflight["paper_skill_library_mode"] = paper_skill_library_mode
        paper_preflight = self._persist_and_snapshot_paper_skills(
            task=task,
            paper_preflight=paper_preflight,
            idx_dir=idx_dir,
            work_dir=work_dir,
            library=library,
        )
        generated_skills = [
            p for p in paper_preflight.get("top_papers", []) or []
            if _paper_is_formal_skill(p, base_dir=work_dir)
        ]
        skill_block = _format_registry_skill_block(paper_preflight)
        prompt = PAPER_ITERATIVE_PROMPT.rstrip() + "\n\n" + REGISTRY_SKILL_BLOCK.replace("{{PAPER_SKILLS_MD}}", skill_block)
        env = LivePaperDiscoveryEnv(
            task_id=str(task["id"]),
            work_dir=work_dir,
            objective_file=objective_file,
            success_artifact_glob=success_artifact_glob,
            registry_entry=task,
            max_steps_soft_trunc=max_steps,
            max_paper_calls=50,
            paper_tool_set="iterative_only",
            require_paper_skill=True,
            min_paper_skills=1,
            runtime_skill_dir=runtime_skill_dir,
        )
        env.paper_call_count += 1
        env.paper_actions.append(f"registry_preflight:{task.get('primary_doi') or task.get('primary_paper_title') or task.get('id')}")
        env.paper_retrieval_results.append(paper_preflight)
        env.paper_skill_count += len(generated_skills)
        for paper in generated_skills:
            if paper.get("skill_path"):
                env.paper_skill_paths.append(str(paper["skill_path"]))
        return PreparedPaperToSkillRun(prompt, env, paper_preflight, len(generated_skills))

    def prepare_v2(
        self,
        task: dict[str, Any],
        *,
        work_dir: Path,
        idx_dir: Path,
        objective_file: Path,
        success_artifact_glob: str,
        max_steps: int = 40,
    ) -> PreparedPaperToSkillRun:
        from paperskills.agents.live_paper_discovery_runner import LivePaperDiscoveryEnv

        runtime_skill_dir = idx_dir / "runtime_paper_skills_v2"
        paper_preflight = self._run_papertoskill_v2_preflight(
            task,
            objective_file=objective_file,
            idx_dir=idx_dir,
            work_dir=work_dir,
        )
        skill_block = _format_registry_skill_block(paper_preflight)
        prompt = PAPER_TO_SKILL_V2_PROMPT.rstrip() + "\n\n" + REGISTRY_SKILL_BLOCK.replace("{{PAPER_SKILLS_MD}}", skill_block)
        env = LivePaperDiscoveryEnv(
            task_id=str(task["id"]),
            work_dir=work_dir,
            objective_file=objective_file,
            success_artifact_glob=success_artifact_glob,
            registry_entry=task,
            max_steps_soft_trunc=max_steps,
            max_paper_calls=50,
            paper_tool_set="iterative_only",
            require_paper_skill=False,
            min_paper_skills=0,
            runtime_skill_dir=runtime_skill_dir,
        )
        env.paper_call_count += 1
        env.paper_actions.append(f"papertoskill_v2_preflight:{task['id']}")
        env.paper_retrieval_results.append(paper_preflight)
        env.paper_skill_count += int(paper_preflight.get("skills_generated") or 0)
        if paper_preflight.get("agent_skill_path"):
            env.paper_skill_paths.append(str(paper_preflight["agent_skill_path"]))
        return PreparedPaperToSkillRun(prompt, env, paper_preflight, int(paper_preflight.get("skills_generated") or 0))

    async def _run_registry_paper_preflight(
        self,
        task: dict[str, Any],
        *,
        runtime_skill_dir: Path,
        retrieval_policy: str = "query_only",
        max_rounds: int = 5,
        top_k: int = 3,
        paper_skill_extractor: str = "hybrid",
    ) -> dict[str, Any]:
        old_skill_dir = os.environ.get("PAPER_ITERATIVE_SKILL_OUTPUT_DIR")
        runtime_skill_dir.mkdir(parents=True, exist_ok=True)
        os.environ["PAPER_ITERATIVE_SKILL_OUTPUT_DIR"] = str(runtime_skill_dir.resolve())
        raw = ""
        try:
            allow_exact = retrieval_policy in {"exact_first", "query_first"}
            if retrieval_policy == "exact_first" and (task.get("primary_doi") or task.get("primary_paper_title")):
                raw = await exact_retrieve_paper_skill(
                    task_family=str(task.get("family") or ""),
                    analysis_type=_registry_analysis_type(task),
                    tool_hint=_registry_paper_hint(task),
                    primary_doi=str(task.get("primary_doi") or ""),
                    primary_title=str(task.get("primary_paper_title") or ""),
                )
                exact_result = json.loads(raw)
                if exact_result.get("success"):
                    return self._annotate_preflight(exact_result, task, runtime_skill_dir, retrieval_policy, paper_skill_extractor, human_like=False)
            raw = await iterative_retrieve_papers(
                task_family=str(task.get("family") or ""),
                analysis_type=_registry_analysis_type(task),
                tool_hint=_registry_paper_hint(task),
                max_rounds=max_rounds,
                top_k=top_k,
                retrieval_profile=_registry_retrieval_profile(task),
                paper_skill_extractor=paper_skill_extractor,
            )
            query_result = json.loads(raw)
            if query_result.get("success") and query_result.get("papers_selected", 0) > 0:
                return self._annotate_preflight(query_result, task, runtime_skill_dir, retrieval_policy, paper_skill_extractor, human_like=True)
            if allow_exact and (task.get("primary_doi") or task.get("primary_paper_title")):
                raw = await exact_retrieve_paper_skill(
                    task_family=str(task.get("family") or ""),
                    analysis_type=_registry_analysis_type(task),
                    tool_hint=_registry_paper_hint(task),
                    primary_doi=str(task.get("primary_doi") or ""),
                    primary_title=str(task.get("primary_paper_title") or ""),
                )
                exact_result = json.loads(raw)
                if exact_result.get("success"):
                    exact_result["source"] = "registry_exact_fallback_after_query_failure"
                    return self._annotate_preflight(exact_result, task, runtime_skill_dir, retrieval_policy, paper_skill_extractor, human_like=False)
        finally:
            if old_skill_dir is None:
                os.environ.pop("PAPER_ITERATIVE_SKILL_OUTPUT_DIR", None)
            else:
                os.environ["PAPER_ITERATIVE_SKILL_OUTPUT_DIR"] = old_skill_dir
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"success": False, "error": "invalid_json_from_iterative_retrieve", "raw": raw[:2000]}
        return self._annotate_preflight(result, task, runtime_skill_dir, retrieval_policy, paper_skill_extractor, human_like=(retrieval_policy != "exact_first"))

    def _annotate_preflight(
        self,
        result: dict[str, Any],
        task: dict[str, Any],
        runtime_skill_dir: Path,
        retrieval_policy: str,
        paper_skill_extractor: str,
        *,
        human_like: bool,
    ) -> dict[str, Any]:
        result["registry_driven"] = True
        result["tool_hint"] = _registry_paper_hint(task)
        result["analysis_type"] = _registry_analysis_type(task)
        result["retrieval_profile"] = _registry_retrieval_profile(task)
        result["runtime_skill_dir"] = str(runtime_skill_dir)
        result["retrieval_policy"] = retrieval_policy
        result["paper_skill_extractor"] = paper_skill_extractor
        result["human_like_query_first"] = human_like
        return result

    def _run_papertoskill_v2_preflight(
        self,
        task: dict[str, Any],
        *,
        objective_file: Path,
        idx_dir: Path,
        work_dir: Path,
    ) -> dict[str, Any]:
        task_id = str(task["id"])
        objective_text = objective_file.read_text(encoding="utf-8", errors="replace") if objective_file.is_file() else ""
        task_context = {
            "id": task.get("id"),
            "title": task.get("title"),
            "difficulty": task.get("difficulty"),
            "analysis_type": _registry_analysis_type(task),
            "tool_hint": _registry_paper_hint(task),
            "primary_paper_title": task.get("primary_paper_title"),
            "primary_doi": task.get("primary_doi"),
            "paper_sensitive": task.get("paper_sensitive", task.get("paper_covered")),
        }
        build_input = objective_text.rstrip() + "\n\nREGISTRY CONTEXT:\n" + json.dumps(task_context, indent=2, ensure_ascii=False)
        artifact_dir = idx_dir / "papertoskill_v2"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        result = PaperToSkillV2Builder().build(
            build_input,
            task_id=task_id,
            fetch_technical_docs=True,
            cache_dir=artifact_dir / "technical_docs",
        )
        plan_path = artifact_dir / "plan.json"
        snapshot_skill_path = artifact_dir / "SKILL.md"
        plan_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        snapshot_skill_path.write_text(result.skill_markdown, encoding="utf-8")

        agent_skill_root = work_dir / "paper_skills" / "papertoskill_v2"
        agent_skill_root.mkdir(parents=True, exist_ok=True)
        agent_skill_path = agent_skill_root / "SKILL.md"
        shutil.copy2(snapshot_skill_path, agent_skill_path)
        (agent_skill_root / "metadata.json").write_text(
            json.dumps({"source": "papertoskill_v2", "task_id": task_id, "plan_path": str(plan_path)}, indent=2) + "\n",
            encoding="utf-8",
        )

        bundle = result.bundle
        paper_like_skill = {
            "title": "PaperToSkill 2.0 operational technical documentation skill",
            "pmid": None,
            "pmcid": None,
            "doi": task.get("primary_doi"),
            "skill_path": str(agent_skill_path.relative_to(work_dir)),
            "snapshot_skill_path": str(snapshot_skill_path),
            "skill_md": result.skill_markdown,
            "skill_tags": ["papertoskill-v2", "technical-documentation", *[p.package for p in bundle.candidate_packages]],
            "low_confidence": False,
            "source_type_guess": "technical_documentation_bundle",
            "source_access_level": "technical_documentation_plan",
            "paper_fulltext_skill": False,
            "docs_supported_skill": True,
            "formal_skill_valid": True,
            "skill_validation": {"is_valid": True, "issues": []},
            "skill_quality": skill_quality_from_markdown(result.skill_markdown),
        }
        return {
            "success": True,
            "source": "papertoskill_v2_preflight",
            "registry_driven": True,
            "task_id": task_id,
            "tool_hint": _registry_paper_hint(task),
            "analysis_type": _registry_analysis_type(task),
            "skills_generated": 1,
            "paper_skill_library_mode": "v2_operational",
            "paper_skill_extractor": "papertoskill_v2",
            "task_intent": bundle.task_intent.to_dict(),
            "candidate_packages": [p.to_dict() for p in bundle.candidate_packages],
            "technical_sources": [s.to_dict() for s in bundle.technical_sources],
            "scientific_sources": [s.to_dict() for s in bundle.scientific_sources],
            "debug_sources": [s.to_dict() for s in bundle.debug_sources],
            "extracted_operations": [op.to_dict() for op in bundle.extracted_operations],
            "skill_payload": bundle.skill_payload,
            "artifact_dir": str(artifact_dir),
            "plan_path": str(plan_path),
            "agent_skill_path": str(agent_skill_path.relative_to(work_dir)),
            "top_papers": [paper_like_skill],
        }

    def _persistent_reuse_preflight(self, task: dict[str, Any], *, library: PersistentPaperSkillLibrary) -> dict[str, Any] | None:
        existing = library.find_for_task(task)
        if not existing:
            return None
        return {
            "success": True,
            "source": "persistent_library_reuse",
            "registry_driven": True,
            "tool_hint": _registry_paper_hint(task),
            "analysis_type": _registry_analysis_type(task),
            "papers_found": 1,
            "papers_selected": 1,
            "low_confidence": bool(existing.get("low_confidence")),
            "top_papers": [existing],
        }

    def _persist_and_snapshot_paper_skills(
        self,
        *,
        task: dict[str, Any],
        paper_preflight: dict[str, Any],
        idx_dir: Path,
        work_dir: Path,
        library: PersistentPaperSkillLibrary,
    ) -> dict[str, Any]:
        persisted: list[dict[str, Any]] = []
        for paper in paper_preflight.get("top_papers", []) or []:
            if (paper.get("persisted_skill") or {}).get("library_skill_path"):
                persisted.append(paper)
            else:
                persisted.append(library.persist_from_retrieval(paper, task=task, preflight=paper_preflight, runner_name=Path(__file__).name))
        paper_preflight["top_papers"] = persisted
        snapshot = library.create_snapshot(
            persisted,
            snapshot_dir=idx_dir / "paper_skill_snapshot",
            include_low_confidence=False,
        )
        agent_skill_root = work_dir / "paper_skills"
        for entry in snapshot.get("entries", []):
            source = Path(str(entry.get("snapshot_skill_path") or ""))
            if not source.is_file():
                continue
            target_dir = agent_skill_root / str(entry.get("slug") or source.parent.name)
            target_dir.mkdir(parents=True, exist_ok=True)
            target_skill = target_dir / "SKILL.md"
            shutil.copy2(source, target_skill)
            entry["agent_skill_path"] = str(target_skill.relative_to(work_dir))
            for paper in persisted:
                if str((paper.get("persisted_skill") or {}).get("library_skill_path") or "") == str(entry.get("persistent_skill_path") or ""):
                    paper["snapshot_skill_path"] = entry.get("snapshot_skill_path")
                    paper["agent_skill_path"] = entry["agent_skill_path"]
                    paper["skill_path"] = entry["agent_skill_path"]
        formal_skills = [p for p in persisted if _paper_is_formal_skill(p, base_dir=work_dir)]
        paper_preflight["persistent_library"] = {
            "library_root": str(library.library_root),
            "persisted_count": len(persisted),
            "formal_skill_count": len(formal_skills),
        }
        paper_preflight["paper_skill_snapshot"] = snapshot
        return paper_preflight
