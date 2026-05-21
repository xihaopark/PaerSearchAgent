"""Agent-facing helpers for PaperToSkill 2.0 planning."""

from __future__ import annotations

import json
from typing import Iterable, Optional

from paperskills.v2.orchestrator import PaperToSkillV2Builder


def plan_papertoskill_v2(
    task_text: str,
    task_id: str = "",
    error_messages: Optional[Iterable[str]] = None,
    fetch_technical_docs: bool = False,
    cache_dir: str = "",
) -> str:
    """Return a JSON PaperToSkill 2.0 evidence plan for a task."""

    result = PaperToSkillV2Builder().build(
        task_text=task_text,
        task_id=task_id,
        error_messages=error_messages,
        fetch_technical_docs=fetch_technical_docs,
        cache_dir=cache_dir or None,
    )
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


def draft_papertoskill_v2_skill(
    task_text: str,
    task_id: str = "",
    error_messages: Optional[Iterable[str]] = None,
    fetch_technical_docs: bool = False,
    cache_dir: str = "",
) -> str:
    """Return the operational SKILL.md draft from the v2 planner."""

    result = PaperToSkillV2Builder().build(
        task_text=task_text,
        task_id=task_id,
        error_messages=error_messages,
        fetch_technical_docs=fetch_technical_docs,
        cache_dir=cache_dir or None,
    )
    return result.skill_markdown
