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

