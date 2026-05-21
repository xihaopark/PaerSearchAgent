# Agent Runtime

This directory is the public extraction of the actual agent path used in the
recent PaperToSkill experiments.

## What Was Extracted

| Module | Origin in the experiment code | Role |
|---|---|---|
| `paperskills.agents.papertoskill_agent` | `run_unified_paper_experiment.py` `paper_to_skill` / `paper_to_skill_v2` branches | preflight retrieval, skill snapshotting, prompt construction |
| `paperskills.agents.prompts` | `run_unified_paper_experiment.py` prompt constants | v1/v2 system prompts and runtime skill block wrapper |
| `paperskills.agents.live_paper_discovery_runner` | `live_paper_discovery_runner.py` | paper-tool-augmented ReAct environment and runner |
| `paperskills.agents.r_task_env` | `r_task_env.py` | file/shell/submission task tools |
| `paperskills.agents.rollout` | `rollout.py` | LDP rollout loop and artifact saving |
| `paperskills.agents.llm_env` | `llm_env.py` | OpenRouter key helper |

## Runtime Flow

```text
registry task + workspace OBJECTIVE.md
  -> PaperToSkillAgent.prepare_v1 / prepare_v2
  -> runtime SKILL.md under workspace/paper_skills/
  -> prompt with embedded skill evidence
  -> LivePaperDiscoveryEnv
  -> SimpleAgent + vanilla_r_task_rollout
  -> task artifacts + evaluator/submission result
```

V1 uses iterative paper retrieval and generated PaperSkills. V2 uses the
technical documentation builder first, then exposes the generated operational
skill to the same downstream coding environment.

## Dependency Boundary

The retrieval/planning layers are ordinary Python modules. The ReAct execution
layer expects the LDP/Aviary runtime used in the original benchmark harness.
Public users who only want paper/doc retrieval can use `papertoskill-v1` and
`papertoskill-v2` without that runtime.
