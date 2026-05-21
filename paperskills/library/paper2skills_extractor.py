#!/usr/bin/env python3
"""Source-aware Paper2Skills adapter for benchmark PaperSkills.

This is a lightweight bridge to the external Paper2Skills idea: read source
bundles strategically, preserve provenance, fill evidence-backed method slots,
and emit the SKILL.md/metadata.json shape consumed by the benchmark runner.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests

from paperskills.library.method_source_resolver import MethodSource, resolve_method_sources
from paperskills.library.paper_extraction import CodeBlockExtractor, PDFExtractor


class _StructuredHTMLParser(HTMLParser):
    """Small dependency-free parser that keeps headings, code and table cells."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._capture: list[str] = []
        self._tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        self._tag_stack.append(tag.lower())
        if tag.lower() in {"h1", "h2", "h3", "h4", "p", "li", "pre", "code", "td", "th"}:
            self._capture = []

    def handle_data(self, data: str) -> None:
        if self._tag_stack and self._tag_stack[-1] in {"script", "style"}:
            return
        if self._tag_stack and self._tag_stack[-1] in {"h1", "h2", "h3", "h4", "p", "li", "pre", "code", "td", "th"}:
            self._capture.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        text = html.unescape(" ".join(self._capture))
        text = re.sub(r"\s+", " ", text).strip()
        if text and tag in {"h1", "h2", "h3", "h4", "p", "li", "pre", "code", "td", "th"}:
            if tag in {"h1", "h2", "h3", "h4"}:
                level = int(tag[1])
                self.parts.append(f"{'#' * min(level, 6)} {text}")
            elif tag in {"pre", "code"}:
                self.parts.append(f"```r\n{text}\n```")
            else:
                self.parts.append(text)
        self._capture = []
        if self._tag_stack:
            self._tag_stack.pop()

    def text(self) -> str:
        return "\n".join(self.parts)


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    return [str(value)] if str(value).strip() else []


class Paper2SkillsExtractor:
    """Convert a method-source bundle into a benchmark PaperSkill."""

    def __init__(self, cache_dir: str | Path | None = None, *, request_timeout: int = 20):
        self.cache_dir = Path(cache_dir or ".cache/paper2skills_sources")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_timeout = request_timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Paper2SkillsExtractor/1.0"})

    def extract(
        self,
        *,
        task_id: str,
        task_context: Any,
        paper: Any,
        fetched_content: str,
        methods_text: str,
    ) -> dict[str, Any]:
        profile = dict(getattr(task_context, "retrieval_profile", {}) or {})
        sources = resolve_method_sources(
            profile,
            paper=paper,
            fetched_content=fetched_content,
            methods_text=methods_text,
        )
        enriched = [self._read_source(src) for src in sources]
        source_text = "\n\n".join(src.text for src in enriched if src.text)
        slots = self._extract_slots(source_text, profile)
        evidence_ledger = self._build_evidence_ledger(slots, enriched)
        quality = self._quality(slots, enriched)
        skill_md = self._render_skill_md(task_id, paper, profile, enriched, slots, quality)
        metadata = {
            "extractor": "paper2skills",
            "llm_used_for_skill_extraction": False,
            "source_bundle_count": len(enriched),
            "source_types": sorted({src.source_type for src in enriched}),
            "source_bundle": [src.to_dict() for src in enriched],
            "source_slots": slots,
            "evidence_ledger": evidence_ledger,
            "quality": quality,
            "provenance": {
                "cache_dir": str(self.cache_dir),
                "source_text_sha256": _sha_text(source_text) if source_text else "",
                "prompt_style": "external/Paper2Skills strategic document reading adapter",
            },
        }
        return {"skill_md": skill_md, "metadata": metadata}

    def _read_source(self, src: MethodSource) -> MethodSource:
        out = replace(src)
        if out.text.strip():
            out.checksum = _sha_text(out.text)
            return out
        if not out.url:
            return out
        try:
            content = self._fetch_url(out.url)
        except Exception:
            return out
        if not content:
            return out
        if content[:4] == b"%PDF" or out.url.lower().endswith(".pdf"):
            try:
                out.text = PDFExtractor(cache_dir=self.cache_dir).extract_text(content)
            except Exception:
                out.text = ""
        else:
            raw = content.decode("utf-8", errors="replace")
            if "<html" in raw[:1000].lower() or "<body" in raw[:1000].lower():
                parser = _StructuredHTMLParser()
                parser.feed(raw)
                out.text = parser.text()
            else:
                out.text = re.sub(r"\s+", " ", raw).strip()
        out.checksum = _sha_text(out.text) if out.text else ""
        return out

    def _fetch_url(self, url: str) -> bytes:
        cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        suffix = ".pdf" if url.lower().endswith(".pdf") else ".html"
        path = self.cache_dir / f"{cache_key}{suffix}"
        if path.is_file() and path.stat().st_size > 200:
            return path.read_bytes()
        resp = self.session.get(url, timeout=self.request_timeout)
        if resp.status_code != 200 or len(resp.content) < 200:
            return b""
        path.write_bytes(resp.content)
        return resp.content

    def _extract_slots(self, text: str, profile: dict[str, Any]) -> dict[str, Any]:
        low = text.lower()
        package = str(profile.get("package") or "").strip()
        data_objects = _as_list(profile.get("data_objects"))
        core_functions = _as_list(profile.get("core_functions"))
        expected_tags = _as_list(profile.get("expected_skill_tags"))

        def hits(terms: list[str]) -> list[str]:
            return [term for term in terms if term and term.lower() in low]

        object_hits = hits(data_objects)
        function_hits = hits(core_functions)
        tag_hits = hits(expected_tags)
        package_hit = [package] if package and package.lower() in low else []

        code_snippets = []
        try:
            code_snippets = [s.code for s in CodeBlockExtractor().extract_from_text(text) if s.code.strip()]
        except Exception:
            code_snippets = []

        evidence = self._evidence_snippets(text, [*package_hit, *object_hits, *function_hits, *tag_hits])
        statistical_terms = [
            term
            for term in [
                "normalization", "dispersion", "linear model", "negative binomial",
                "contrast", "p-value", "fdr", "false discovery", "empirical bayes",
                "quasi-likelihood", "log fold change", "size factor",
            ]
            if term in low
        ]
        io_terms = [
            term
            for term in ["input", "output", "matrix", "data frame", "table", "columns", "result", "write"]
            if term in low
        ]
        return {
            "method_scope": self._first_nonempty_sentence(text),
            "package": package,
            "package_hits": package_hit,
            "data_objects": object_hits,
            "function_chain": function_hits,
            "statistical_choices": statistical_terms[:12],
            "input_output_mapping": io_terms[:12],
            "parameters": self._parameter_terms(text),
            "code_snippets": code_snippets[:5],
            "limitations": self._limitation_sentences(text),
            "evidence_snippets": evidence[:12],
            "expected_tag_hits": tag_hits,
        }

    def _quality(self, slots: dict[str, Any], sources: list[MethodSource]) -> dict[str, Any]:
        authoritative = [
            s for s in sources
            if s.source_type in {"paper", "manual", "vignette", "function_doc", "package_page"}
            and (s.text or s.url or s.doi or s.pmid)
        ]
        covered = sum(
            bool(slots.get(key))
            for key in ["data_objects", "function_chain", "input_output_mapping", "statistical_choices", "code_snippets"]
        )
        abstract_only = not any(len(s.text or "") > 1200 for s in sources if s.source_type != "paper") and not any(
            len(s.text or "") > 2000 for s in sources
        )
        wrapper_only = bool(sources) and all(s.source_type == "wrapper_or_downstream_pipeline" for s in sources)
        has_official_docs = any(
            s.source_type in {"manual", "vignette", "function_doc", "package_page"}
            and len(s.text or "") > 1200
            for s in sources
        )
        paper_fulltext_available = any(
            s.source_type == "paper"
            and len(s.text or "") > 2000
            and "[Full text unavailable]" not in (s.text or "")
            for s in sources
        )
        return {
            "authoritative_source_count": len(authoritative),
            "covered_slot_count": covered,
            "has_object_semantics": bool(slots.get("data_objects")),
            "has_function_chain": bool(slots.get("function_chain")),
            "has_input_output_mapping": bool(slots.get("input_output_mapping")),
            "has_statistical_choice": bool(slots.get("statistical_choices")),
            "has_executable_pseudocode": bool(slots.get("code_snippets") or slots.get("function_chain")),
            "abstract_only": abstract_only,
            "wrapper_only": wrapper_only,
            "docs_source_valid": has_official_docs,
            "source_access_level": "documentation_fulltext" if has_official_docs else "",
            "formal_source_valid": paper_fulltext_available,
            "paper_fulltext_source_valid": paper_fulltext_available,
            "evidence_slots_complete": len(authoritative) >= 1 and covered >= 2 and not abstract_only and not wrapper_only,
        }

    def _build_evidence_ledger(self, slots: dict[str, Any], sources: list[MethodSource]) -> list[dict[str, Any]]:
        """Tie key extracted claims to compact source evidence."""
        terms = []
        for key in ["package_hits", "data_objects", "function_chain", "expected_tag_hits"]:
            terms.extend(_as_list(slots.get(key)))
        snippets = slots.get("evidence_snippets") or []
        ledger = []
        for term in terms:
            supporting = next((s for s in snippets if term.lower() in s.lower()), "")
            source = next((src for src in sources if term.lower() in (src.text or "").lower()), None)
            if not supporting or source is None:
                continue
            ledger.append({
                "claim": f"Source supports method term: {term}",
                "source_id": source.url or source.doi or source.pmid or source.title,
                "source_type": source.source_type,
                "evidence_text_checksum": _sha_text(supporting),
                "covered_tags": [term],
            })
        return ledger[:20]

    def _render_skill_md(
        self,
        task_id: str,
        paper: Any,
        profile: dict[str, Any],
        sources: list[MethodSource],
        slots: dict[str, Any],
        quality: dict[str, Any],
    ) -> str:
        title = getattr(paper, "title", "") or profile.get("package") or task_id
        doi = getattr(paper, "doi", "") or ""
        pmid = getattr(paper, "pmid", "") or ""
        source_lines = []
        for src in sources:
            ident = src.url or src.doi or src.pmid or src.title
            source_lines.append(f"- {src.source_type}: {src.title} ({ident})")
        funcs = slots.get("function_chain") or []
        objs = slots.get("data_objects") or []
        stats = slots.get("statistical_choices") or []
        io = slots.get("input_output_mapping") or []
        snippets = slots.get("code_snippets") or []
        evidence = slots.get("evidence_snippets") or []

        lines = [
            "---",
            f"name: paper2skills-{re.sub(r'[^a-zA-Z0-9_.-]+', '_', doi or pmid or task_id).strip('_')}",
            "source: paper2skills_extractor",
            f"doi: {doi}",
            f"pmid: {pmid}",
            "tags:",
            "  - paper2skills",
            "  - source_aware",
            f"  - {str(profile.get('package') or task_id).lower()}",
            f"paper_title: {title}",
            "---",
            "",
            "## Method Scope",
            slots.get("method_scope") or "Source-aware method guidance extracted from authoritative paper/documentation sources.",
            "",
            "## Data Objects",
        ]
        lines.extend([f"- {x}" for x in objs] or ["- No explicit data object evidence found."])
        lines.extend(["", "## Function / API Chain"])
        lines.extend([f"- {x}" for x in funcs] or ["- No explicit function/API chain evidence found."])
        lines.extend(["", "## Statistical Choices"])
        lines.extend([f"- {x}" for x in stats] or ["- No explicit statistical choice evidence found."])
        lines.extend(["", "## Input / Output Mapping"])
        lines.extend([f"- {x}" for x in io] or ["- No explicit input/output mapping evidence found."])
        lines.extend(["", "## Algorithm Steps"])
        if funcs:
            lines.append("Pseudocode:")
            lines.append("1. Read the task inputs and construct the source-supported data object(s).")
            lines.append("2. Apply the documented function/API chain in order: " + " -> ".join(funcs[:8]) + ".")
            lines.append("3. Compute the requested values while preserving documented statistical choices.")
            lines.append("4. Write deterministic tabular outputs using the required columns.")
        else:
            lines.append("Pseudocode:")
            lines.append("1. Use the source evidence below to map inputs to the requested output tables without inventing hidden values.")
        if snippets:
            lines.extend(["", "## Source Code Snippets"])
            for code in snippets[:3]:
                lines.extend(["```r", code.strip(), "```"])
        lines.extend(["", "## Limitations"])
        lines.extend([f"- {x}" for x in slots.get("limitations") or []] or ["- Treat claims not supported by the source bundle as task context, not paper evidence."])
        lines.extend(["", "## Evidence Snippets"])
        lines.extend([f"- {x}" for x in evidence] or ["- No compact evidence snippets extracted."])
        lines.extend(["", "## Source Bundle"])
        lines.extend(source_lines or ["- No source bundle entries."])
        lines.extend(["", "## Quality Gate"])
        for key, val in quality.items():
            lines.append(f"- {key}: {str(val).lower() if isinstance(val, bool) else val}")
        return "\n".join(lines).strip() + "\n"

    def _first_nonempty_sentence(self, text: str) -> str:
        for sent in re.split(r"(?<=[.!?])\s+", text):
            sent = re.sub(r"\s+", " ", sent).strip()
            if 40 <= len(sent) <= 400:
                return sent
        return ""

    def _parameter_terms(self, text: str) -> list[str]:
        patterns = [
            r"\b([A-Za-z][A-Za-z0-9_.-]{2,})\s*=\s*([A-Za-z0-9_.-]+)",
            r"\b(argument|parameter|threshold|cutoff|design|contrast)\b[^.]{0,160}",
        ]
        out: list[str] = []
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.I):
                val = re.sub(r"\s+", " ", m.group(0)).strip()
                if 5 <= len(val) <= 220 and val not in out:
                    out.append(val)
                if len(out) >= 10:
                    return out
        return out

    def _limitation_sentences(self, text: str) -> list[str]:
        out = []
        for sent in re.split(r"(?<=[.!?])\s+", text):
            low = sent.lower()
            if any(k in low for k in ["limitation", "caution", "warning", "note that", "not recommended"]):
                clean = re.sub(r"\s+", " ", sent).strip()
                if 30 <= len(clean) <= 240:
                    out.append(clean)
            if len(out) >= 5:
                break
        return out

    def _evidence_snippets(self, text: str, terms: list[str]) -> list[str]:
        snippets = []
        for term in terms:
            if not term:
                continue
            m = re.search(re.escape(term), text, re.I)
            if not m:
                continue
            start = max(0, m.start() - 140)
            end = min(len(text), m.end() + 220)
            snippet = re.sub(r"\s+", " ", text[start:end]).strip()
            if snippet and snippet not in snippets:
                snippets.append(snippet)
        return snippets
