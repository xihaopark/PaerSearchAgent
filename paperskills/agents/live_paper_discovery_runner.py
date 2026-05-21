#!/usr/bin/env python3
"""Live Paper Discovery Mode Runner - Paper-to-Skill system with real paper retrieval.

This runner enables agents to:
1. Search PubMed/Europe PMC/bioRxiv for relevant papers
2. Fetch and extract paper content (methods, code)
3. Synthesize skills from papers
4. Implement solutions based on extracted skills

Usage:
    python live_paper_discovery_runner.py \
        --registry r_tasks/top5_live_paper_test.json \
        --config config/paper_live_discovery_mode.yaml \
        --run-id live_paper_test_$(date +%Y%m%dT%H%M%S) \
        --task-ids dea_limma
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASK_ROOT = _REPO_ROOT / "tasks"
_RUN_ROOT = _REPO_ROOT / "runs"

# Import paper discovery tools
from paperskills.library.paper_discovery_tools import (
    search_papers,
    search_papers_for_task,
    fetch_paper_content,
    extract_skill_from_paper,
    synthesize_skill_from_papers,
    validate_extracted_skill,
    PAPER_DISCOVERY_TOOLS,
)
from paperskills.library.paper_search import PaperMetadata

from paperskills.agents.llm_env import apply_openrouter_key_from_file
from paperskills.agents.rollout import (
    save_run_artifacts,
    vanilla_r_task_rollout,
)
from paperskills.agents.r_task_env import RTaskEvalEnv
from ldp.agent.simple_agent import SimpleAgent

logger = logging.getLogger(__name__)


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class LivePaperDiscoveryEnv(RTaskEvalEnv):
    """Extended RTaskEvalEnv with live paper discovery tools."""

    def __init__(
        self,
        *args,
        max_paper_calls: int = 10,
        paper_tool_set: str = "full",
        require_paper_skill: bool = False,
        min_paper_skills: int = 1,
        runtime_skill_dir: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.paper_call_count = 0
        self.max_paper_calls = max_paper_calls
        self.paper_tool_set = paper_tool_set
        self.require_paper_skill = require_paper_skill
        self.min_paper_skills = min_paper_skills
        self.runtime_skill_dir = Path(runtime_skill_dir).resolve() if runtime_skill_dir else None

        # Add paper discovery tools (full granular API vs iterative-only online retrieval)
        from aviary.core import Tool

        if paper_tool_set == "iterative_only":
            self.tools.extend(
                [Tool.from_function(self.iterative_retrieve_papers)]
            )
        else:
            self.tools.extend(
                [
                    Tool.from_function(self.search_papers),
                    Tool.from_function(self.search_papers_for_task),
                    Tool.from_function(self.fetch_paper_content),
                    Tool.from_function(self.extract_skill_from_paper),
                    Tool.from_function(self.synthesize_skill_from_papers),
                    Tool.from_function(self.validate_extracted_skill),
                    Tool.from_function(self.iterative_retrieve_papers),
                ]
            )

        # Track paper discovery actions
        self.paper_actions = []
        self.paper_skill_count = 0
        self.paper_skill_paths: list[str] = []
        self.paper_retrieval_results: list[dict[str, Any]] = []

    def _check_paper_limit(self) -> bool:
        """Check if paper calls within limit."""
        if self.paper_call_count >= self.max_paper_calls:
            return False
        self.paper_call_count += 1
        return True

    async def search_papers(self, query: str, source: str = "pubmed", max_results: int = 10) -> str:
        """Search academic papers matching the query.

        Args:
            query: Search query (e.g., "DESeq2 apeglm RNA-seq")
            source: Source to search ("pubmed", "europepmc", "biorxiv", "all")
            max_results: Maximum papers to return
        """
        if not self._check_paper_limit():
            return json.dumps({"error": "Paper discovery call limit reached"})

        result = search_papers(query, source, max_results)
        self.paper_actions.append(f"search:{source}:{query[:50]}")
        return result

    async def search_papers_for_task(
        self,
        task_family: str,
        analysis_type: str = "",
        tool_hint: str = "",
        key_method: str = "",
        source: str = "pubmed",
        max_results: int = 10,
    ) -> str:
        """Search papers optimized for a specific task.

        Args:
            task_family: Task family ("rna", "methylation", etc.)
            analysis_type: Analysis type ("differential_expression", etc.)
            tool_hint: Tool name hint ("DESeq2", etc.)
            key_method: Specific method hint ("apeglm", etc.)
            source: Source to search
            max_results: Maximum results
        """
        if not self._check_paper_limit():
            return json.dumps({"error": "Paper discovery call limit reached"})

        result = search_papers_for_task(
            task_family, analysis_type, tool_hint, key_method, source, max_results
        )
        self.paper_actions.append(f"search_task:{task_family}:{tool_hint}:{key_method}")
        return result

    async def fetch_paper_content(
        self,
        pmid: str = "",
        pmcid: str = "",
        doi: str = "",
        extract_methods: bool = True,
        extract_code: bool = True,
    ) -> str:
        """Fetch and extract content from a paper.

        Args:
            pmid: PubMed ID
            pmcid: PMC ID (preferred for full text)
            doi: DOI
            extract_methods: Whether to extract Methods section
            extract_code: Whether to extract code snippets
        """
        if not self._check_paper_limit():
            return json.dumps({"error": "Paper discovery call limit reached"})

        result = fetch_paper_content(pmid, pmcid, doi, extract_methods, extract_code)
        self.paper_actions.append(f"fetch:{pmcid or pmid}")
        return result

    async def extract_skill_from_paper(
        self,
        pmid: str = "",
        pmcid: str = "",
        task_family: str = "",
        analysis_type: str = "",
        tool_hint: str = "",
        key_method: str = "",
    ) -> str:
        """Extract a skill from a single paper.

        Args:
            pmid: PubMed ID
            pmcid: PMC ID (preferred)
            task_family: Task family context
            analysis_type: Analysis type context
            tool_hint: Tool name hint
            key_method: Specific method hint
        """
        if not self._check_paper_limit():
            return json.dumps({"error": "Paper discovery call limit reached"})

        result = extract_skill_from_paper(
            pmid, pmcid, task_family, analysis_type, tool_hint, key_method
        )
        self.paper_actions.append(f"extract_skill:{pmcid or pmid}")
        return result

    async def synthesize_skill_from_papers(
        self,
        paper_ids: list,
        task_family: str = "",
        analysis_type: str = "",
        tool_hint: str = "",
        key_method: str = "",
    ) -> str:
        """Synthesize a skill from multiple papers.

        Args:
            paper_ids: List of PMC IDs or PMIDs
            task_family: Task family context
            analysis_type: Analysis type context
            tool_hint: Tool name hint
            key_method: Specific method hint
        """
        if not self._check_paper_limit():
            return json.dumps({"error": "Paper discovery call limit reached"})

        result = synthesize_skill_from_papers(
            paper_ids, task_family, analysis_type, tool_hint, key_method
        )
        self.paper_actions.append(f"synthesize:{len(paper_ids)}_papers")
        return result

    async def validate_extracted_skill(self, skill_json: str) -> str:
        """Validate an extracted skill for completeness.

        Args:
            skill_json: JSON string of the skill to validate
        """
        if not self._check_paper_limit():
            return json.dumps({"error": "Paper discovery call limit reached"})

        result = validate_extracted_skill(skill_json)
        self.paper_actions.append("validate_skill")
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def iterative_retrieve_papers(
        self,
        task_family: str,
        analysis_type: str,
        tool_hint: str = "",
        max_rounds: int = 5,
        top_k: int = 3,
    ) -> str:
        """Iterative paper retrieval with multi-round search and adaptive scoring.

        This tool performs:
        1. Multi-round search with query refinement
        2. Paper pool building with relevance scoring
        3. Top-K selection based on combined scores
        4. Full text fetch for selected papers only

        Args:
            task_family: Task family (e.g., 'rna', 'scrna', 'spatial')
            analysis_type: Type of analysis (e.g., 'differential_expression')
            tool_hint: Specific tool or method name to search for
            max_rounds: Maximum search rounds (default: 5)
            top_k: Number of top papers to fetch (default: 3)

        Returns:
            JSON string with top-K papers and their methods sections
        """
        # Import here to avoid circular imports
        from paperskills.library.iterative_paper_retrieval import iterative_retrieve_papers as _iterative_retrieve


        # Paper discovery calls are free (don't count against limit)
        # but we still track them for metrics
        self.paper_call_count += 1
        self.paper_actions.append(f"iterative_retrieval:{task_family}:{analysis_type}:{tool_hint}")

        old_skill_dir = os.environ.get("PAPER_ITERATIVE_SKILL_OUTPUT_DIR")
        if self.runtime_skill_dir:
            self.runtime_skill_dir.mkdir(parents=True, exist_ok=True)
            os.environ["PAPER_ITERATIVE_SKILL_OUTPUT_DIR"] = str(self.runtime_skill_dir)
        try:
            result = await _iterative_retrieve(
                task_family=task_family,
                analysis_type=analysis_type,
                tool_hint=tool_hint,
                max_rounds=max_rounds,
                top_k=top_k,
            )
        finally:
            if old_skill_dir is None:
                os.environ.pop("PAPER_ITERATIVE_SKILL_OUTPUT_DIR", None)
            else:
                os.environ["PAPER_ITERATIVE_SKILL_OUTPUT_DIR"] = old_skill_dir

        try:
            parsed = json.loads(result)
            self.paper_retrieval_results.append(parsed)
            generated = [
                p for p in parsed.get("top_papers", [])
                if p.get("skill_md") or p.get("skill_path") or p.get("skill")
            ]
            self.paper_skill_count += len(generated)
            for paper in generated:
                if paper.get("skill_path"):
                    self.paper_skill_paths.append(str(paper["skill_path"]))
            if not generated:
                self.paper_actions.append("paper_skill_missing")
        except (json.JSONDecodeError, AttributeError):
            self.paper_actions.append("paper_skill_parse_failed")

        return result

    def submit_done(self, success: bool, summary: str = "") -> str:
        """End the episode, enforcing PaperToSkill completion when configured.

        Args:
            success: Whether the agent believes the task objective is satisfied.
            summary: Short optional note for logging.
        """
        if success and self.require_paper_skill and self.paper_skill_count < self.min_paper_skills:
            success = False
            summary = (
                summary
                + f" [PaperToSkill gate failed: generated {self.paper_skill_count} "
                f"paper skills, required {self.min_paper_skills}.]"
            )
        return super().submit_done(success=success, summary=summary)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_registry(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _resolve_work_dir(entry: dict[str, Any]) -> Path:
    rel = entry.get("work_dir", "")
    if not rel:
        # Default path construction
        task_id = entry.get("id", "unknown")
        return _TASK_ROOT / task_id

    p = Path(rel)
    if p.is_absolute():
        return p.resolve()
    return (_REPO_ROOT / p).resolve()


async def run_live_paper_task(
    task_entry: dict[str, Any],
    config: dict[str, Any],
    run_id: str,
    output_root: Path,
    *,
    registry_index: int = 0,
) -> dict[str, Any]:
    """Run single task in live paper discovery mode."""

    task_id = task_entry["id"]
    canonical_work_dir = _resolve_work_dir(task_entry)

    if not canonical_work_dir.exists():
        logger.error(f"Task {task_id}: work_dir not found: {canonical_work_dir}")
        return {
            "task_id": task_id,
            "error": f"work_dir not found: {canonical_work_dir}",
            "live_paper_mode": True,
        }

    success_glob = task_entry.get("success_artifact_glob", "output/result.txt")
    max_steps = int(config.get("max_steps", 20))
    max_paper_calls = int(config.get("live_paper_discovery", {}).get("max_paper_calls", 10))

    run_dir = (output_root / run_id / f"{registry_index:03d}_{task_id}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir = run_dir / "workspace"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(canonical_work_dir, work_dir, symlinks=False)

    logger.info("[Live Paper Mode] Starting %s (workspace=%s)", task_id, work_dir)

    env = LivePaperDiscoveryEnv(
        task_id=task_id,
        work_dir=work_dir,
        objective_file=work_dir / "OBJECTIVE.md",
        success_artifact_glob=success_glob,
        max_steps_soft_trunc=max_steps,
        max_paper_calls=max_paper_calls,
    )

    agent_config = config.get("agent", {})
    sys_prompt = agent_config.get("sys_prompt", "")

    agent = SimpleAgent(
        llm_model=agent_config.get("llm_model", {"name": "openrouter/openai/gpt-5.4"}),
        sys_prompt=sys_prompt,
    )

    rollout_coro = vanilla_r_task_rollout(
        agent,
        env,
        max_steps=max_steps,
    )

    wall_s = config.get("per_task_timeout_s")
    if wall_s is not None and int(wall_s) > 0:
        try:
            trajectory, _ = await asyncio.wait_for(rollout_coro, timeout=float(wall_s))
        except TimeoutError as e:
            logger.error(
                "[Live Paper Mode] %s exceeded per_task_timeout_s=%s",
                task_id,
                wall_s,
            )
            raise RuntimeError(
                f"per_task_timeout_s={wall_s}s exceeded for task {task_id}"
            ) from e
    else:
        trajectory, _ = await rollout_coro

    metadata: dict[str, Any] = {
        "task_id": task_id,
        "canonical_work_dir": str(canonical_work_dir),
        "work_dir": str(work_dir),
        "smoke": False,
        "batch_run_id": run_id,
        "registry_index": registry_index,
        "git_sha": _git_sha(_REPO_ROOT),
        "agent": {"llm_model": agent_config.get("llm_model")},
        "live_paper_mode": True,
        "paper_calls": env.paper_call_count,
        "paper_actions": env.paper_actions,
    }

    await save_run_artifacts(run_dir, trajectory, metadata)

    logger.info(
        "[Live Paper Mode] Finished %s: %s paper tool actions",
        task_id,
        len(env.paper_actions),
    )

    return metadata


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True, type=Path, help="Registry JSON path")
    ap.add_argument("--config", required=True, type=Path, help="Config YAML path")
    ap.add_argument("--run-id", required=True, help="Run identifier")
    ap.add_argument("--output-root", type=Path, default=_RUN_ROOT)
    ap.add_argument("--openrouter-key-file", type=Path, default=_REPO_ROOT / "openrouterkey.txt")
    ap.add_argument("--task-ids", nargs="+", help="Specific task IDs to run (default: all)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load config and registry
    config = _load_yaml(args.config)
    registry = _load_registry(args.registry)

    logger.info("=" * 60)
    logger.info("Live Paper Discovery Mode Runner")
    logger.info("=" * 60)
    logger.info(f"Registry: {args.registry}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Run ID: {args.run_id}")

    # Set API key
    if args.openrouter_key_file.exists():
        apply_openrouter_key_from_file(args.openrouter_key_file)
        logger.info(f"API key loaded from: {args.openrouter_key_file}")
    else:
        logger.warning(f"API key file not found: {args.openrouter_key_file}")

    # Filter tasks if specific IDs provided
    all_tasks = registry.get("tasks", [])
    if args.task_ids:
        tasks = [t for t in all_tasks if t.get("id") in args.task_ids]
        logger.info(f"Running {len(tasks)} selected tasks from {len(all_tasks)} total")
    else:
        tasks = all_tasks
        logger.info(f"Running all {len(tasks)} tasks")

    # Run tasks
    results = []

    for idx, task_entry in enumerate(tasks):
        logger.info(f"\n--- Task {idx+1}/{len(tasks)}: {task_entry.get('id')} ---")
        try:
            result = await run_live_paper_task(
                task_entry=task_entry,
                config=config,
                run_id=args.run_id,
                output_root=args.output_root,
                registry_index=idx,
            )
            results.append(result)

            # Log paper discovery stats
            if result.get("paper_calls"):
                logger.info(f"  Paper discovery calls: {result['paper_calls']}")

        except Exception as e:
            logger.error(f"Task {task_entry.get('id')} failed: {e}")
            results.append({
                "task_id": task_entry.get("id"),
                "error": str(e),
                "live_paper_mode": True,
            })

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Live Paper Discovery Summary")
    logger.info("=" * 60)
    logger.info(f"Total tasks: {len(tasks)}")
    completed = len([r for r in results if "error" not in r])
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {len(tasks) - completed}")

    total_paper_calls = sum(r.get("paper_calls", 0) for r in results)
    logger.info(f"Total paper discovery calls: {total_paper_calls}")
    if completed > 0:
        logger.info(f"Avg paper calls per completed task: {total_paper_calls / completed:.1f}")

    # Save summary
    summary_path = args.output_root / args.run_id / "live_paper_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "run_id": args.run_id,
            "mode": "live_paper_discovery",
            "registry": str(args.registry),
            "config": str(args.config),
            "total_tasks": len(tasks),
            "completed": completed,
            "failed": len(tasks) - completed,
            "total_paper_calls": total_paper_calls,
            "results": results,
        }, f, indent=2)

    logger.info(f"\nSummary saved: {summary_path}")
    logger.info(f"Run directory: {args.output_root / args.run_id}")


if __name__ == "__main__":
    asyncio.run(main())
