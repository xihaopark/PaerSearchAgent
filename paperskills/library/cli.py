"""Command-line entry point for PaperToSkill v1 paper retrieval."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from paperskills.library.iterative_paper_retrieval import iterative_retrieve_papers


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PaperToSkill v1 paper-first retrieval.")
    parser.add_argument("--task-family", required=True, help="Task family, e.g. rna, methylation, chipseq.")
    parser.add_argument("--analysis-type", required=True, help="Analysis type, e.g. differential_expression.")
    parser.add_argument("--tool-hint", default="", help="Optional package/tool/method hint.")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--skill-output-dir", type=Path, default=Path(".cache/papertoskill_v1_skills"))
    parser.add_argument("--json-out", type=Path, help="Optional path for the JSON retrieval report.")
    args = parser.parse_args()

    args.skill_output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PAPER_ITERATIVE_SKILL_OUTPUT_DIR"] = str(args.skill_output_dir.resolve())

    raw = asyncio.run(
        iterative_retrieve_papers(
            task_family=args.task_family,
            analysis_type=args.analysis_type,
            tool_hint=args.tool_hint,
            max_rounds=args.max_rounds,
            top_k=args.top_k,
        )
    )
    parsed = json.loads(raw)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
