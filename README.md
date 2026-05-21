# PaperSearchAgent

PaperSearchAgent is a cleaned public extraction of the PaperToSkill research
agents. It contains two complementary agents for acquiring executable scientific
programming knowledge from live sources.

## Agents

| Agent layer | Main question | Evidence/tools | Output |
|---|---|---|---|
| PaperToSkill v1 | Which paper should ground this task? | method papers, software papers, protocol papers | runtime `SKILL.md` from selected paper/source bundle |
| PaperToSkill v2 | How is the method executed in R/Bioconductor? | package pages, vignettes, manuals, function docs | technical-doc-grounded operational `SKILL.md` |
| ReAct task agent | Can the task be solved using the generated skill? | shell/file tools plus optional paper tools | task outputs evaluated by the benchmark harness |

The key distinction is evidence type. Papers are useful for method rationale;
technical documentation is usually necessary for exact package APIs, object
classes, arguments, and runnable workflow patterns.

The code here is extracted from the recent experiment path in
`main/paper_primary_benchmark/ldp_r_task_eval/run_unified_paper_experiment.py`,
not a fresh rewrite. The extraction keeps the same preflight shape:
retrieve/build runtime skills, copy them into `workspace/paper_skills/`, inject
them into the prompt, then run the downstream ReAct coding environment.

## Repository Layout

```text
paperskills/
  library/              # PaperToSkill v1 paper-first retrieval
  v2/                   # PaperToSkill v2 technical-doc grounding
  agents/               # Extracted preflight + ReAct execution layer
docs/
  architecture.md
  agent_runtime.md
  usage.md
examples/
  spilterlize_norm_edger/task.md
```

## Quick Start

```bash
python -m pip install -e .
```

V1 paper-first retrieval:

```bash
papertoskill-v1 \
  --task-family rna \
  --analysis-type differential_expression \
  --tool-hint "edgeR TMM normalization DGEList calcNormFactors cpm" \
  --json-out outputs/v1_edger.json
```

V2 paper + technical-doc skill planning:

```bash
papertoskill-v2 \
  --task-file examples/spilterlize_norm_edger/task.md \
  --task-id spilterlize_norm_edger \
  --fetch-technical-docs \
  --cache-dir outputs/v2_docs \
  --json-out outputs/v2_plan.json \
  --skill-out outputs/v2_SKILL.md
```

## Notes

- This public repo intentionally excludes benchmark runs, hidden reference
  outputs, caches, local credentials, and experiment workspaces.
- Network access is required for live paper/document retrieval.
- The full ReAct execution layer uses the LDP/Aviary runtime and an LLM backend
  such as OpenRouter. Retrieval-only CLIs do not require that stack.
- The historical Python namespace `paperskills` is preserved to keep the
  research prototype imports stable.
