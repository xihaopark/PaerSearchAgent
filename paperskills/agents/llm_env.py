"""Optional helpers for LLM backends used by `ldp` / LiteLLM (e.g. OpenRouter)."""

from __future__ import annotations

import os
from pathlib import Path


def apply_openrouter_key_from_file(path: Path) -> bool:
    """Set ``OPENROUTER_API_KEY`` from the first non-empty, non-comment line.

    LiteLLM reads this env var for ``openrouter/...`` models. Returns whether a key
    was loaded.
    """
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        key = line.strip()
        if key and not key.startswith("#"):
            os.environ["OPENROUTER_API_KEY"] = key
            return True
    return False
