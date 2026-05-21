"""Command-line entry point for PaperToSkill 2.0 planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paperskills.v2.orchestrator import PaperToSkillV2Builder


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a PaperToSkill 2.0 evidence plan.")
    parser.add_argument("--task-text", help="Task text. Use --task-file for longer prompts.")
    parser.add_argument("--task-file", type=Path, help="Path to task prompt/objective text.")
    parser.add_argument("--task-id", default="", help="Optional task identifier.")
    parser.add_argument("--error-message", action="append", default=[], help="Execution error to route into debug search.")
    parser.add_argument("--fetch-technical-docs", action="store_true", help="Fetch official technical documentation and include excerpts.")
    parser.add_argument("--cache-dir", type=Path, help="Directory for fetched technical documentation.")
    parser.add_argument("--json-out", type=Path, help="Write full JSON artifact to this path.")
    parser.add_argument("--skill-out", type=Path, help="Write SKILL.md draft to this path.")
    args = parser.parse_args()

    if args.task_file:
        task_text = args.task_file.read_text(encoding="utf-8")
    elif args.task_text:
        task_text = args.task_text
    else:
        parser.error("Provide --task-file or --task-text")

    cache_dir = args.cache_dir
    if args.fetch_technical_docs and cache_dir is None:
        cache_dir = (args.json_out.parent / "technical_docs") if args.json_out else Path(".cache/papertoskill_v2_docs")

    result = PaperToSkillV2Builder().build(
        task_text=task_text,
        task_id=args.task_id,
        error_messages=args.error_message,
        fetch_technical_docs=args.fetch_technical_docs,
        cache_dir=cache_dir,
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    if args.skill_out:
        args.skill_out.parent.mkdir(parents=True, exist_ok=True)
        args.skill_out.write_text(result.skill_markdown, encoding="utf-8")

    if not args.json_out and not args.skill_out:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
