#!/usr/bin/env python3
"""Unified index for paper skills library.

Supports:
- DOI lookup (canonical and slug forms)
- Tool name fuzzy search
- Family/field filtering
- Task recommendation lookup
- Full-text search over SKILL.md content
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PaperEntry:
    """Single paper skill entry."""

    doi: str
    doi_slug: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int = 0
    tool: str = ""
    family: str = ""
    kind: str = "method"  # method/workflow/reference
    source: str = ""
    skill_md_path: str = ""
    pdf_path: str = ""
    brat_annotated: bool = False
    tasks_recommended: list[str] = field(default_factory=list)
    skill_md_content: str = ""  # cached for search

    @classmethod
    def from_dict(cls, d: dict) -> PaperEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {
            "doi": self.doi,
            "doi_slug": self.doi_slug,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "tool": self.tool,
            "family": self.family,
            "kind": self.kind,
            "source": self.source,
            "skill_md_path": self.skill_md_path,
            "pdf_path": self.pdf_path,
            "brat_annotated": self.brat_annotated,
            "tasks_recommended": self.tasks_recommended,
        }


class LibraryIndex:
    """In-memory index for paper skills."""

    def __init__(self, library_root: Path | None = None):
        self.root = library_root or Path(__file__).resolve().parents[1]
        self.master_path = self.root / "indices" / "master_index.json"
        self.entries: dict[str, PaperEntry] = {}  # by DOI (canonical)
        self._by_tool: dict[str, list[str]] = defaultdict(list)  # tool -> DOIs
        self._by_family: dict[str, list[str]] = defaultdict(list)  # family -> DOIs
        self._by_task: dict[str, list[str]] = defaultdict(list)  # task -> DOIs
        self._loaded = False

    def _canonical_doi(self, doi: str) -> str:
        """Normalize DOI string."""
        d = doi.strip().lower()
        d = re.sub(r"^https?://doi\.org/", "", d)
        d = re.sub(r"_", "/", d)
        return d

    def _doi_to_slug(self, doi: str) -> str:
        """Convert canonical DOI to slug used in folder names."""
        d = self._canonical_doi(doi)
        d = re.sub(r"[./-]", "_", d)
        return d

    def load(self) -> LibraryIndex:
        """Load master_index.json and build auxiliary indices."""
        if not self.master_path.exists():
            self._loaded = True
            return self

        data = json.loads(self.master_path.read_text(encoding="utf-8"))
        for item in data.get("entries", []):
            entry = PaperEntry.from_dict(item)
            canonical = self._canonical_doi(entry.doi)
            self.entries[canonical] = entry
            if entry.tool:
                self._by_tool[entry.tool.lower()].append(canonical)
            if entry.family:
                self._by_family[entry.family.lower()].append(canonical)
            for t in entry.tasks_recommended:
                self._by_task[t.lower()].append(canonical)
        self._loaded = True
        return self

    def save(self) -> None:
        """Write current entries back to master_index.json."""
        data = {
            "version": "2026-05-01",
            "count": len(self.entries),
            "entries": [e.to_dict() for e in self.entries.values()],
        }
        self.master_path.parent.mkdir(parents=True, exist_ok=True)
        self.master_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def by_doi(self, doi: str) -> PaperEntry | None:
        """Lookup by DOI (canonical or slug)."""
        canonical = self._canonical_doi(doi)
        return self.entries.get(canonical)

    def by_tool(self, tool: str, fuzzy: bool = False) -> list[PaperEntry]:
        """Lookup by tool name (exact or fuzzy)."""
        key = tool.lower()
        if not fuzzy:
            return [self.entries[d] for d in self._by_tool.get(key, [])]
        # Simple fuzzy: substring match
        results = []
        for t, dois in self._by_tool.items():
            if key in t or t in key:
                results.extend(self.entries[d] for d in dois)
        return list({e.doi: e for e in results}.values())

    def by_family(self, family: str) -> list[PaperEntry]:
        """Lookup by technical family (rna, chip, methyl, etc.)."""
        key = family.lower()
        return [self.entries[d] for d in self._by_family.get(key, [])]

    def recommended_for_task(self, task_id: str) -> list[PaperEntry]:
        """Get papers recommended for a specific task."""
        key = task_id.lower()
        return [self.entries[d] for d in self._by_task.get(key, [])]

    def search(self, query: str, limit: int = 10) -> list[tuple[PaperEntry, float]]:
        """Simple full-text search over SKILL.md content.
        Returns [(entry, score), ...] sorted by score desc.
        """
        q = query.lower()
        scored = []
        for e in self.entries.values():
            content = (e.title + " " + e.skill_md_content).lower()
            # Simple scoring: exact match count / length
            if q in content:
                score = content.count(q) * len(q) / max(1, len(content))
                scored.append((e, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def add(self, entry: PaperEntry) -> None:
        """Add or update entry."""
        canonical = self._canonical_doi(entry.doi)
        self.entries[canonical] = entry
        # Rebuild aux indices for this entry
        if entry.tool:
            self._by_tool[entry.tool.lower()].append(canonical)
        if entry.family:
            self._by_family[entry.family.lower()].append(canonical)
        for t in entry.tasks_recommended:
            self._by_task[t.lower()].append(canonical)

    def all_tools(self) -> list[str]:
        return sorted(self._by_tool.keys())

    def all_families(self) -> list[str]:
        return sorted(self._by_family.keys())

    def check_overlap(self, other_dois: list[str]) -> tuple[list[str], list[str]]:
        """Compare with external DOI list. Returns (overlap, unique_in_other)."""
        other_canonical = {self._canonical_doi(d) for d in other_dois}
        existing = set(self.entries.keys())
        overlap = sorted(other_canonical & existing)
        unique = sorted(other_canonical - existing)
        return overlap, unique
