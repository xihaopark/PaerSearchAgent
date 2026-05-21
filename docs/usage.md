# Usage

Install locally:

```bash
python -m pip install -e .
```

Run v1 paper-first retrieval:

```bash
papertoskill-v1 \
  --task-family rna \
  --analysis-type differential_expression \
  --tool-hint "edgeR TMM normalization DGEList calcNormFactors cpm" \
  --json-out outputs/v1_edger.json
```

Run v2 paper + technical-document planning:

```bash
papertoskill-v2 \
  --task-file examples/spilterlize_norm_edger/task.md \
  --task-id spilterlize_norm_edger \
  --fetch-technical-docs \
  --cache-dir outputs/v2_docs \
  --json-out outputs/v2_plan.json \
  --skill-out outputs/v2_SKILL.md
```

For network-restricted environments, omit `--fetch-technical-docs`; v2 will
still create a documentation plan and package/function routing result.

## Running the Extracted Agent Path

The recent experiments used a two-part path:

1. `PaperToSkillAgent` prepares the runtime skill and prompt.
2. `ldp.agent.simple_agent.SimpleAgent` runs against `LivePaperDiscoveryEnv`
   through `vanilla_r_task_rollout`.

Minimal sketch:

```python
from pathlib import Path

from ldp.agent.simple_agent import SimpleAgent
from paperskills.agents import PaperToSkillAgent, vanilla_r_task_rollout

task = {
    "id": "spilterlize_norm_edger",
    "family": "rna",
    "analysis_type": "differential_expression",
    "success_artifact_glob": "output/all/normTMM.csv",
}

prepared = await PaperToSkillAgent().prepare_v2(
    task,
    work_dir=Path("runs/example/workspace"),
    idx_dir=Path("runs/example"),
    objective_file=Path("runs/example/workspace/OBJECTIVE.md"),
    success_artifact_glob=task["success_artifact_glob"],
    max_steps=40,
)

agent = SimpleAgent(
    llm_model={"name": "openrouter/openai/gpt-5.4"},
    sys_prompt=prepared.prompt,
)
trajectory, _ = await vanilla_r_task_rollout(agent, prepared.env, max_steps=40)
```

This path requires the same LDP/Aviary runtime stack used by the experiments and
an LLM backend key, for example `OPENROUTER_API_KEY`. The retrieval-only CLIs
above remain usable without the ReAct runtime.
