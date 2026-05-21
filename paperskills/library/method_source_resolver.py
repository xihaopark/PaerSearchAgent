#!/usr/bin/env python3
"""Generic method-source discovery for PaperToSkill.

This module deliberately resolves source *types* rather than task ids.  The
runner may provide a hidden retrieval profile, but the resolver only uses it to
discover authoritative method sources such as package pages, manuals,
vignettes, and function documentation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass
class MethodSource:
    source_type: str
    title: str
    url: str = ""
    local_path: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    checksum: str = ""
    authority: str = ""
    source_role: str = "supporting"
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("text"):
            data["text_checksum"] = hashlib.sha256(data["text"].encode("utf-8")).hexdigest()
            data["text_preview"] = data["text"][:1000]
            data.pop("text", None)
        return data


def classify_source(title: str, abstract: str = "", publication_type: str = "") -> str:
    """Classify source scope without task-specific rules."""
    text = " ".join([title or "", abstract or "", publication_type or ""]).lower()
    title_l = (title or "").lower()
    if any(x in text for x in ["manual", "vignette", "user guide", "documentation", "reference manual"]):
        return "documentation_or_vignette"
    if any(x in title_l for x in [
        "gui", "graphical user interface", "wrapper", "web server", "front-end",
        "-based r pipeline", " based r pipeline", "downstream pipeline",
    ]):
        return "wrapper_or_downstream_pipeline"
    if any(x in title_l for x in ["review", "survey"]) or "review" in publication_type.lower():
        return "broad_review"
    if any(x in title_l for x in ["patient", "patients", "cohort", "case-control", "clinical", "cancer", "tumor"]):
        return "application_paper"
    if any(x in text for x in ["software", "r package", "bioconductor", "package", "algorithm", "method"]):
        return "canonical_method_or_software"
    if any(x in text for x in ["workflow", "protocol", "tutorial"]):
        return "protocol_or_workflow"
    return "unknown"


def _clean_package(package: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.]+", "", str(package or "").strip())


def _dedupe_sources(sources: Iterable[MethodSource]) -> list[MethodSource]:
    seen = set()
    out: list[MethodSource] = []
    for src in sources:
        key = (src.source_type, src.url or src.local_path or src.title)
        if key in seen:
            continue
        seen.add(key)
        out.append(src)
    return out


def resolve_method_sources(
    retrieval_profile: dict[str, Any] | None,
    *,
    paper: Any | None = None,
    fetched_content: str = "",
    methods_text: str = "",
) -> list[MethodSource]:
    """Build a source bundle around a candidate paper and method profile."""
    profile = retrieval_profile or {}
    package = _clean_package(str(profile.get("package") or ""))
    ecosystem = str(profile.get("ecosystem") or "").lower()
    core_functions = [
        _clean_package(str(x))
        for x in profile.get("core_functions", []) or []
        if _clean_package(str(x))
    ]

    sources: list[MethodSource] = []
    if paper is not None:
        title = getattr(paper, "title", "") or ""
        abstract = getattr(paper, "abstract", "") or ""
        source_type = classify_source(title, abstract, getattr(paper, "publication_type", "") or "")
        sources.append(
            MethodSource(
                source_type="paper",
                title=title or "Retrieved paper",
                doi=getattr(paper, "doi", "") or "",
                pmid=getattr(paper, "pmid", "") or "",
                pmcid=getattr(paper, "pmcid", "") or "",
                authority=getattr(paper, "journal", "") or "PubMed",
                source_role="canonical" if source_type == "canonical_method_or_software" else "candidate",
                text="\n\n".join(x for x in [methods_text, fetched_content] if x),
            )
        )

    if package and ("bioconductor" in ecosystem or ecosystem == "bioc"):
        base = f"https://bioconductor.org/packages/release/bioc"
        sources.extend(
            [
                MethodSource(
                    source_type="package_page",
                    title=f"{package} Bioconductor package page",
                    url=f"{base}/html/{package}.html",
                    authority="Bioconductor",
                    source_role="authoritative_doc",
                ),
                MethodSource(
                    source_type="vignette",
                    title=f"{package} Bioconductor vignette",
                    url=f"{base}/vignettes/{package}/inst/doc/{package}.html",
                    authority="Bioconductor",
                    source_role="authoritative_doc",
                ),
                MethodSource(
                    source_type="manual",
                    title=f"{package} Bioconductor reference manual",
                    url=f"{base}/manuals/{package}/man/{package}.pdf",
                    authority="Bioconductor",
                    source_role="authoritative_doc",
                ),
            ]
        )
        for func in core_functions[:8]:
            sources.append(
                MethodSource(
                    source_type="function_doc",
                    title=f"{package}::{func} function documentation",
                    url=f"https://rdrr.io/bioc/{package}/man/{func}.html",
                    authority="rdrr.io/Bioconductor",
                    source_role="api_doc",
                )
            )

    return _dedupe_sources(sources)


def source_bundle_to_json(sources: list[MethodSource]) -> str:
    return json.dumps([src.to_dict() for src in sources], indent=2, ensure_ascii=False)
