#!/usr/bin/env python3
"""Persistent PaperToSkill library utilities.

Generated PaperToSkill artifacts are durable method assets. Experiments should
consume a frozen snapshot of this library, not ad-hoc runtime files directly.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paperskills.library.indices.library_index import LibraryIndex, PaperEntry


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _slug(raw: str) -> str:
    s = re.sub(r"^https?://doi\.org/", "", str(raw or "").strip(), flags=re.I)
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_")
    return s[:140] or "unknown"


def paper_slug(paper: dict[str, Any]) -> str:
    if paper.get("doi"):
        return _slug(str(paper["doi"]))
    if paper.get("pmid"):
        return _slug(f"pmid_{paper['pmid']}")
    if paper.get("pmcid"):
        return _slug(f"pmcid_{paper['pmcid']}")
    title = str(paper.get("title") or "unknown")
    return _slug(f"title_{sha256_text(title)[:16]}")


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _norm_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _title_token_overlap(query_title: str, candidate_title: str) -> float:
    stop = {
        "the", "and", "for", "with", "from", "using", "along", "analysis",
        "method", "methods", "single", "cell", "cells", "differential",
    }
    q_tokens = {x for x in _norm_title(query_title).split() if len(x) >= 4 and x not in stop}
    c_tokens = {x for x in _norm_title(candidate_title).split() if len(x) >= 4 and x not in stop}
    if not q_tokens or not c_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens)


def skill_quality_from_markdown(skill_md: str) -> dict[str, Any]:
    """Heuristic executability checks for generated PaperSkills."""
    text = str(skill_md or "").lower()
    metadata_like = len(text) > 0 and sum(
        token in text
        for token in [
            "algorithm step",
            "metric definition",
            "output mapping",
            "pseudocode",
            ".csv",
            "compute",
            "fit ",
            "cluster",
            "module",
            "metric_name",
        ]
    ) < 2
    return {
        "has_algorithm_steps": any(x in text for x in ["algorithm step", "step:", "compute", "fit ", "cluster"]),
        "has_formula_or_metric_definition": any(
            x in text for x in ["metric definition", "formula", "ratio", "score", "similarity", "p-value", "pvalue"]
        ),
        "has_output_mapping": any(x in text for x in ["output mapping", ".csv", "columns", "metric_name", "write "]),
        "has_pseudocode": "pseudocode" in text or "```" in text,
        "metadata_only_or_generic": metadata_like,
    }


class PersistentPaperSkillLibrary:
    """Read/write persistent paper skills and create per-run snapshots."""

    def __init__(self, library_root: str | Path | None = None, repo_root: str | Path | None = None):
        self.library_root = Path(library_root) if library_root else Path(__file__).resolve().parent
        self.library_root = self.library_root.resolve()
        self.repo_root = Path(repo_root).resolve() if repo_root else self.library_root.parents[1].resolve()
        self.methods_dir = self.library_root / "methods"
        self.index = LibraryIndex(self.library_root).load()

    def _skill_path_from_entry(self, entry: PaperEntry) -> Path:
        p = Path(entry.skill_md_path)
        if not p.is_absolute():
            p = self.repo_root / p
        return p.resolve()

    def find_for_task(self, task: dict[str, Any]) -> dict[str, Any] | None:
        """Find an existing skill by DOI first, then by exact title substring."""
        doi = str(task.get("primary_doi") or "").strip()
        if doi:
            entry = self._best_entry_for_doi(doi, task)
            if entry:
                found = self._entry_to_paper(entry, reason="primary_doi")
                if found:
                    return found

        title = str(task.get("primary_paper_title") or "").strip().lower()
        if title:
            title_norm = _norm_title(title)
            best: tuple[float, PaperEntry] | None = None
            for entry in self.index.entries.values():
                candidate = _norm_title(entry.title or "")
                overlap = _title_token_overlap(title, entry.title or "")
                if candidate and (title_norm in candidate or candidate in title_norm or overlap >= 0.35):
                    if best is None or overlap > best[0]:
                        best = (overlap, entry)
            if best:
                found = self._entry_to_paper(best[1], reason="primary_title")
                if found:
                    return found
            if self.index.master_path.is_file():
                data = json.loads(self.index.master_path.read_text(encoding="utf-8"))
                for item in data.get("entries", []):
                    entry = PaperEntry.from_dict(item)
                    candidate = _norm_title(entry.title or "")
                    overlap = _title_token_overlap(title, entry.title or "")
                    if candidate and (title_norm in candidate or candidate in title_norm or overlap >= 0.35):
                        if best is None or overlap > best[0]:
                            best = (overlap, entry)
                if best:
                    found = self._entry_to_paper(best[1], reason="primary_title")
                    if found:
                        return found
        return None

    def _best_entry_for_doi(self, doi: str, task: dict[str, Any]) -> PaperEntry | None:
        canonical = self.index._canonical_doi(doi)  # keep consistent with LibraryIndex.
        candidates = [e for e in self.index.entries.values() if self.index._canonical_doi(e.doi) == canonical]
        if self.index.master_path.is_file():
            data = json.loads(self.index.master_path.read_text(encoding="utf-8"))
            candidates.extend(
                PaperEntry.from_dict(item)
                for item in data.get("entries", [])
                if self.index._canonical_doi(str(item.get("doi", ""))) == canonical
            )
        if not candidates:
            entry = self.index.by_doi(doi)
            return entry
        candidates = list({e.skill_md_path or e.doi_slug: e for e in candidates}.values())
        hints: list[str] = []
        for raw in task.get("search_terms", []) or []:
            hints.append(str(raw))
        km = task.get("key_methods") or task.get("key_method") or ""
        if isinstance(km, list):
            hints.extend(str(x) for x in km)
        else:
            hints.append(str(km))
        task_text = " ".join(hints).lower()

        def score(entry: PaperEntry) -> tuple[int, str]:
            hay = " ".join([entry.title, entry.tool, entry.doi_slug, entry.skill_md_content]).lower()
            generic = {"method", "analysis", "small", "samples", "sample", "comparison", "shrinkage"}
            hits = sum(
                1
                for token in re.split(r"[^a-z0-9]+", task_text)
                if len(token) >= 4 and token not in generic and token in hay
            )
            return (hits, entry.doi_slug)

        return sorted(candidates, key=score, reverse=True)[0]

    def _entry_to_paper(self, entry: PaperEntry, *, reason: str) -> dict[str, Any] | None:
        skill_path = self._skill_path_from_entry(entry)
        if not skill_path.is_file() or skill_path.stat().st_size == 0:
            return None
        skill_md = skill_path.read_text(encoding="utf-8", errors="replace")
        meta_path = skill_path.parent / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        retrieval_meta = meta.get("retrieval") or {}
        extractor_meta = meta.get("extractor") or {}
        skill_quality = meta.get("skill_quality") or skill_quality_from_markdown(skill_md)
        return {
            "doi": entry.doi,
            "pmid": meta.get("paper", {}).get("pmid", ""),
            "pmcid": meta.get("paper", {}).get("pmcid", ""),
            "title": entry.title,
            "year": entry.year,
            "journal": meta.get("paper", {}).get("journal", ""),
            "skill_md": skill_md,
            "skill_path": str(skill_path),
            "skill_tags": meta.get("tags", []) or [entry.family, entry.tool],
            "skill_validation": meta.get("validation", {"is_valid": True, "issues": ["legacy_metadata_missing"]}),
            "skill_quality": skill_quality,
            "source_type_guess": retrieval_meta.get("source_type_guess", ""),
            "source_scope_score": retrieval_meta.get("source_scope_score"),
            "negative_source_penalty": retrieval_meta.get("negative_source_penalty"),
            "extractor": extractor_meta.get("name", ""),
            "source_bundle_count": extractor_meta.get("source_bundle_count", 0),
            "source_types": extractor_meta.get("source_types", []),
            "llm_used_for_skill_extraction": bool(extractor_meta.get("llm_used_for_skill_extraction")),
            "evidence_slots_complete": bool(extractor_meta.get("evidence_slots_complete")),
            "source_bundle": meta.get("source_bundle", []),
            "extractor_metadata": extractor_meta.get("metadata", {}),
            "low_confidence": bool(meta.get("low_confidence", False)),
            "persisted_skill": {
                "status": "reused",
                "reuse_reason": reason,
                "library_skill_path": str(skill_path),
                "skill_sha256": sha256_file(skill_path),
                "metadata_path": str(meta_path) if meta_path.is_file() else "",
            },
        }

    def persist_from_retrieval(
        self,
        paper: dict[str, Any],
        *,
        task: dict[str, Any],
        preflight: dict[str, Any],
        runner_name: str,
    ) -> dict[str, Any]:
        """Persist one retrieved/generated paper skill and return an enriched paper dict."""
        out = dict(paper)
        skill_md = str(out.get("skill_md") or "")
        source_skill_path = Path(str(out.get("skill_path") or ""))
        if not skill_md and source_skill_path.is_file():
            skill_md = source_skill_path.read_text(encoding="utf-8", errors="replace")
        if not skill_md.strip():
            out["persisted_skill"] = {"status": "skipped", "reason": "missing_skill_md"}
            return out

        slug = paper_slug(out)
        target_dir = self.methods_dir / slug
        skill_sha = sha256_text(skill_md)
        if (target_dir / "SKILL.md").is_file():
            existing_sha = sha256_file(target_dir / "SKILL.md")
            if existing_sha == skill_sha:
                status = "reused"
            else:
                target_dir = self.methods_dir / f"{slug}_v{skill_sha[:10]}"
                status = "versioned"
        else:
            status = "created"

        target_dir.mkdir(parents=True, exist_ok=True)
        skill_path = target_dir / "SKILL.md"
        if not skill_path.is_file():
            skill_path.write_text(skill_md, encoding="utf-8")

        methods_text = str(out.get("methods_section") or "")
        validation = out.get("skill_validation") or {}
        tags = [str(x) for x in out.get("skill_tags", []) or [] if x]
        paper_meta = {
            "doi": out.get("doi") or "",
            "pmid": out.get("pmid") or "",
            "pmcid": out.get("pmcid") or "",
            "title": out.get("title") or "",
            "year": out.get("year") or 0,
            "journal": out.get("journal") or "",
            "authors": out.get("authors") or [],
        }
        metadata = {
            "schema_version": 1,
            "paper": paper_meta,
            "task_context": {
                "task_id": task.get("id"),
                "family": task.get("family"),
                "analysis_type": preflight.get("analysis_type"),
                "tool_hint": preflight.get("tool_hint"),
                "key_methods": task.get("key_methods") or task.get("key_method"),
            },
            "retrieval": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query_hint": preflight.get("tool_hint"),
                "score": out.get("score"),
                "relevance": out.get("relevance"),
                "method_package_score": out.get("method_package_score"),
                "skill_extractability_score": out.get("skill_extractability_score"),
                "source_scope_score": out.get("source_scope_score"),
                "source_type_guess": out.get("source_type_guess"),
                "negative_source_penalty": out.get("negative_source_penalty"),
                "abstract_quality": out.get("abstract_quality"),
                "low_confidence": bool(out.get("low_confidence")),
                "source_access_level": out.get("source_access_level"),
                "formal_source_valid": bool(out.get("formal_source_valid")),
                "formal_source_strength": out.get("formal_source_strength"),
                "formal_skill_valid": bool(out.get("formal_skill_valid")),
                "docs_source_valid": bool(out.get("docs_source_valid")),
                "paper_fulltext_skill": bool(out.get("paper_fulltext_skill")),
                "docs_supported_skill": bool(out.get("docs_supported_skill")),
                "abstract_only_skill": bool(out.get("abstract_only_skill")),
                "failure_category": out.get("failure_category"),
                "source_url": out.get("source_url"),
                "source_local_path": out.get("source_local_path"),
                "source_attempts": out.get("source_attempts", []),
                "source_portfolio": out.get("source_portfolio", []),
                "auxiliary_sources": out.get("auxiliary_sources", []),
                "retrieval_process": preflight.get("retrieval_process"),
            },
            "extractor": {
                "name": out.get("extractor") or preflight.get("paper_skill_extractor") or "heuristic",
                "source_bundle_count": out.get("source_bundle_count", 0),
                "source_types": out.get("source_types", []),
                "llm_used_for_skill_extraction": bool(out.get("llm_used_for_skill_extraction")),
                "evidence_slots_complete": bool(out.get("evidence_slots_complete")),
                "metadata": out.get("extractor_metadata", {}),
            },
            "source_bundle": out.get("source_bundle", []),
            "validation": validation,
            "skill_quality": skill_quality_from_markdown(skill_md),
            "tags": tags,
            "low_confidence": bool(out.get("low_confidence")),
            "checksums": {
                "skill_md_sha256": sha256_file(skill_path),
                "methods_section_sha256": sha256_text(methods_text) if methods_text else "",
            },
            "generation": {
                "runner": runner_name,
                "repo_git_sha": _git_sha(self.repo_root),
                "source_runtime_skill_path": str(source_skill_path) if source_skill_path else "",
            },
        }
        (target_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        doi = str(paper_meta["doi"] or slug)
        entry = PaperEntry(
            doi=doi,
            doi_slug=target_dir.name,
            title=str(paper_meta["title"] or target_dir.name),
            authors=list(paper_meta["authors"] or []),
            year=int(paper_meta["year"] or 0),
            tool=str((task.get("key_method") or task.get("key_methods") or "") if task else ""),
            family=str(task.get("family") or ""),
            kind="method",
            source="paper_to_skill_generated",
            skill_md_path=_relative_to_repo(skill_path, self.repo_root),
            tasks_recommended=[str(task.get("id"))] if task.get("id") else [],
            skill_md_content=skill_md[:5000],
        )
        self._upsert_master_entry(entry)
        self.index.add(entry)

        out["skill_md"] = skill_md
        out["skill_path"] = str(skill_path)
        out["skill_quality"] = metadata["skill_quality"]
        out["extractor"] = metadata["extractor"]["name"]
        out["source_bundle_count"] = metadata["extractor"]["source_bundle_count"]
        out["source_types"] = metadata["extractor"]["source_types"]
        out["llm_used_for_skill_extraction"] = metadata["extractor"]["llm_used_for_skill_extraction"]
        out["evidence_slots_complete"] = metadata["extractor"]["evidence_slots_complete"]
        out["persisted_skill"] = {
            "status": status,
            "library_skill_path": str(skill_path),
            "metadata_path": str(target_dir / "metadata.json"),
            "skill_sha256": sha256_file(skill_path),
            "slug": target_dir.name,
        }
        return out

    def _upsert_master_entry(self, entry: PaperEntry) -> None:
        """Update master_index.json without collapsing same-DOI variant entries."""
        path = self.index.master_path
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = list(data.get("entries", []))
        else:
            data = {"version": datetime.now(timezone.utc).date().isoformat(), "entries": []}
            entries = []
        new_item = entry.to_dict()
        key = new_item.get("skill_md_path") or new_item.get("doi_slug")
        replaced = False
        for i, item in enumerate(entries):
            item_key = item.get("skill_md_path") or item.get("doi_slug")
            if item_key == key:
                entries[i] = {**item, **new_item}
                replaced = True
                break
        if not replaced:
            entries.append(new_item)
        data["count"] = len(entries)
        data["entries"] = entries
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def create_snapshot(
        self,
        papers: list[dict[str, Any]],
        *,
        snapshot_dir: str | Path,
        include_low_confidence: bool = False,
    ) -> dict[str, Any]:
        """Copy selected persistent skills into a frozen per-run snapshot."""
        snapshot_root = Path(snapshot_dir)
        snapshot_root.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        for paper in papers:
            persisted = paper.get("persisted_skill") or {}
            source = Path(str(persisted.get("library_skill_path") or paper.get("skill_path") or ""))
            if not source.is_file():
                continue
            validation = paper.get("skill_validation") or {}
            skill_quality = paper.get("skill_quality") or skill_quality_from_markdown(source.read_text(encoding="utf-8", errors="replace"))
            is_valid = validation.get("is_valid", True)
            low_conf = bool(paper.get("low_confidence"))
            if low_conf and not include_low_confidence:
                continue
            if is_valid is False:
                continue
            if skill_quality.get("metadata_only_or_generic"):
                continue
            slug = persisted.get("slug") or source.parent.name
            target_dir = snapshot_root / str(slug)
            target_dir.mkdir(parents=True, exist_ok=True)
            target_skill = target_dir / "SKILL.md"
            shutil.copy2(source, target_skill)
            source_meta = source.parent / "metadata.json"
            if source_meta.is_file():
                shutil.copy2(source_meta, target_dir / "metadata.json")
            entries.append({
                "slug": slug,
                "snapshot_skill_path": str(target_skill),
                "persistent_skill_path": str(source),
                "skill_sha256": sha256_file(target_skill),
                "doi": paper.get("doi"),
                "pmid": paper.get("pmid"),
                "pmcid": paper.get("pmcid"),
                "title": paper.get("title"),
                "low_confidence": low_conf,
                "validation": validation,
                "skill_quality": skill_quality,
                "persisted_status": persisted.get("status"),
            })

        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "library_root": str(self.library_root),
            "include_low_confidence": include_low_confidence,
            "skill_count": len(entries),
            "entries": entries,
        }
        manifest_path = snapshot_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        manifest["manifest_sha256"] = sha256_file(manifest_path)
        return manifest
