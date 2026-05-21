# Architecture

PaperSearchAgent contains two related agents.

## PaperToSkill v1

V1 is paper-first. It starts with task metadata, generates method-aware paper
queries, searches PubMed, builds a candidate pool, scores papers, fetches full
text for selected sources when possible, and generates runtime `SKILL.md`
artifacts.

Core modules:

- `paperskills.library.query_generator`
- `paperskills.library.paper_search`
- `paperskills.library.paper_extraction`
- `paperskills.library.paper2skills_extractor`
- `paperskills.library.iterative_paper_retrieval`
- `paperskills.library.persistent_skill_library`

V1 is best for method/software-paper grounding. It is weaker when task success
depends on package-specific APIs, object classes, current function signatures,
or vignette workflow conventions.

## PaperToSkill v2

V2 adds an operational technical-documentation layer. It routes task text to
candidate R/Bioconductor packages, plans official package pages, vignettes,
manuals, and documentation queries, fetches technical docs when requested, and
renders a docs-grounded operational skill.

Core modules:

- `paperskills.v2.domain_router`
- `paperskills.v2.technical_docs`
- `paperskills.v2.orchestrator`
- `paperskills.v2.skill_renderer`

The intended evidence split is:

- papers answer why a method is appropriate;
- technical docs answer how to call package functions correctly.

## Extracted Experiment Agent Layer

The public `paperskills.agents` package contains the agent code used around the
retrieval builders in the recent experiments.

Core modules:

- `paperskills.agents.papertoskill_agent`
- `paperskills.agents.prompts`
- `paperskills.agents.live_paper_discovery_runner`
- `paperskills.agents.r_task_env`
- `paperskills.agents.rollout`
- `paperskills.agents.llm_env`

`PaperToSkillAgent.prepare_v1(...)` is the extracted `paper_to_skill` branch. It
runs iterative paper retrieval, persists selected runtime skills, snapshots them
into the task workspace, formats the skill block, and returns the prompt/env
pair used by the downstream coding agent.

`PaperToSkillAgent.prepare_v2(...)` is the extracted `paper_to_skill_v2` branch.
It runs the v2 technical-doc builder, writes `plan.json` and `SKILL.md`, copies
the generated skill into `workspace/paper_skills/papertoskill_v2/`, and returns
the prompt/env pair for execution.

`LivePaperDiscoveryEnv` and `RTaskEvalEnv` are the ReAct-style task environment:
the agent can inspect files, execute commands, submit completion, and optionally
call paper retrieval tools. `vanilla_r_task_rollout(...)` is the rollout loop
used with `ldp.agent.simple_agent.SimpleAgent` in the original experiments.

The extraction intentionally leaves benchmark datasets, hidden references,
cached runs, local keys, and large workspaces out of the public repo.
