"""Technical documentation discovery for PaperToSkill 2.0."""

from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urljoin

import requests

from paperskills.v2.models import CandidatePackage, EvidenceSource, TaskIntent


BIOC_PACKAGE_PAGE = "https://bioconductor.org/packages/release/bioc/html/{package}.html"
BIOC_VIGNETTE_PAGE = "https://bioconductor.org/packages/release/bioc/vignettes/{package}/inst/doc/"
BIOC_MANUAL_PAGE = "https://bioconductor.org/packages/release/bioc/manuals/{package}/man/{package}.pdf"


class TechnicalDocPlanner:
    """Build operational-documentation search targets."""

    def plan_for_package(
        self,
        package: CandidatePackage,
        intent: Optional[TaskIntent] = None,
    ) -> List[EvidenceSource]:
        useful_sections = [
            "installation and loading",
            "object classes",
            "canonical workflow",
            "function signatures",
            "input/output coercion",
            "examples",
        ]
        functions = list(package.functions)
        query_context = " ".join(
            [
                intent.analysis_intent if intent else "",
                " ".join(intent.input_types if intent else []),
                " ".join(intent.output_types if intent else []),
            ]
        ).strip()

        base_query = f"Bioconductor {package.package} vignette {' '.join(package.query_hints)} {query_context}".strip()
        manual_query = f"Bioconductor {package.package} reference manual {' '.join(functions[:5])}".strip()
        pkgdown_query = f"{package.package} documentation examples {' '.join(functions[:5])}".strip()

        return [
            EvidenceSource(
                source_type="Bioconductor package page",
                title=f"{package.package} Bioconductor package page",
                url=BIOC_PACKAGE_PAGE.format(package=package.package),
                package=package.package,
                evidence_role="package identity and dependency context",
                query=base_query,
                useful_sections=["description", "biocViews", "vignettes", "depends/imports"],
                functions=functions,
            ),
            EvidenceSource(
                source_type="Bioconductor vignette",
                title=f"{package.package} task-oriented vignette",
                url=BIOC_VIGNETTE_PAGE.format(package=package.package),
                package=package.package,
                evidence_role="workflow",
                query=base_query,
                useful_sections=useful_sections,
                functions=functions,
            ),
            EvidenceSource(
                source_type="reference manual",
                title=f"{package.package} reference manual",
                url=BIOC_MANUAL_PAGE.format(package=package.package),
                package=package.package,
                evidence_role="function signature and parameter semantics",
                query=manual_query,
                useful_sections=["function arguments", "value/object class", "examples"],
                functions=functions,
            ),
            EvidenceSource(
                source_type="package documentation search",
                title=f"{package.package} examples and pkgdown documentation",
                package=package.package,
                evidence_role="examples and version-specific usage",
                query=pkgdown_query,
                useful_sections=["examples", "articles", "reference"],
                functions=functions,
            ),
        ]

    def plan_debug_sources(
        self,
        package: CandidatePackage,
        error_message: str,
    ) -> List[EvidenceSource]:
        if not error_message:
            return []
        query = f"{package.package} {error_message} Bioconductor support"
        return [
            EvidenceSource(
                source_type="Bioconductor support/debug search",
                title=f"{package.package} error-specific support search",
                package=package.package,
                evidence_role="failure repair",
                query=query,
                useful_sections=["accepted answer", "reproducible example", "version notes"],
                functions=package.functions,
            )
        ]

    def plan_many(
        self,
        packages: Iterable[CandidatePackage],
        intent: Optional[TaskIntent] = None,
    ) -> List[EvidenceSource]:
        sources: List[EvidenceSource] = []
        for package in packages:
            sources.extend(self.plan_for_package(package, intent=intent))
        return sources


class TechnicalDocFetcher:
    """Fetch and lightly extract technical documentation pages.

    The fetcher is intentionally conservative: it retrieves official package
    pages and task-oriented vignette HTML where available, records downloaded
    artifacts, and stores short excerpts for prompt/skill grounding. Failures
    are attached to the source objects instead of raising, so experiments can
    continue when a documentation site is temporarily unavailable.
    """

    def __init__(self, cache_dir: Path, *, timeout: int = 20) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "PaperToSkillV2/0.1 (+technical-doc-grounding)",
                "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.8,*/*;q=0.5",
            }
        )

    def fetch_many(
        self,
        sources: Iterable[EvidenceSource],
        *,
        max_sources: int = 8,
        max_vignettes_per_package: int = 2,
    ) -> List[EvidenceSource]:
        fetched: List[EvidenceSource] = []
        seen_urls: set[str] = set()
        for source in sources:
            if len(fetched) >= max_sources:
                break
            if not source.url:
                source.fetch_status = "skipped:no_url"
                fetched.append(source)
                continue
            if source.url in seen_urls:
                continue
            seen_urls.add(source.url)
            if source.source_type == "Bioconductor vignette":
                expanded = self._fetch_vignette_source(
                    source,
                    max_vignettes=max_vignettes_per_package,
                )
                fetched.extend(expanded[: max_sources - len(fetched)])
            else:
                fetched.append(self.fetch_source(source))
        return fetched

    def fetch_source(self, source: EvidenceSource) -> EvidenceSource:
        try:
            response = self.session.get(source.url, timeout=self.timeout)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            suffix = ".pdf" if "pdf" in content_type or source.url.lower().endswith(".pdf") else ".html"
            local_path = self._write_bytes(source, response.content, suffix=suffix)
            source.local_path = str(local_path)
            source.content_sha256 = self._sha256(response.content)
            if suffix == ".pdf":
                source.source_access_level = "technical_pdf_downloaded"
                pdf_text = self._pdf_to_text(local_path)
                if pdf_text:
                    source.fetch_status = "fetched_pdf_text"
                    source.excerpt = self._focused_excerpt(pdf_text, source)
                else:
                    source.fetch_status = "downloaded_pdf:not_text_extracted"
                    source.excerpt = "PDF reference manual downloaded; use it for exact function signatures if text extraction is available."
            else:
                text = self._html_to_text(response.text)
                source.source_access_level = "technical_html_fulltext"
                source.fetch_status = "fetched"
                source.excerpt = self._focused_excerpt(text, source)
        except Exception as exc:  # network/doc failures should not kill benchmark runs
            source.fetch_status = f"failed:{type(exc).__name__}:{str(exc)[:180]}"
            source.source_access_level = "planned_unfetched"
        return source

    def _fetch_vignette_source(
        self,
        source: EvidenceSource,
        *,
        max_vignettes: int,
    ) -> List[EvidenceSource]:
        index = self.fetch_source(source)
        results = [index]
        if not index.local_path or not str(index.local_path).endswith(".html"):
            for url in self._candidate_vignette_urls(source):
                child = EvidenceSource(
                    source_type="Bioconductor vignette HTML",
                    title=f"{source.package} vignette: {Path(url).name}",
                    url=url,
                    package=source.package,
                    evidence_role=source.evidence_role,
                    query=source.query,
                    useful_sections=source.useful_sections,
                    functions=source.functions,
                )
                fetched = self.fetch_source(child)
                results.append(fetched)
                if fetched.fetch_status.startswith("fetched"):
                    break
            return results
        try:
            html_text = Path(index.local_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return results
        links = self._extract_links(html_text, source.url)
        vignette_links = [
            link for link in links
            if re.search(r"\.(html|htm)$", link, re.I)
            and not link.rstrip("/").endswith("/index.html")
        ]
        ranked = self._rank_links(vignette_links, source)
        for url in ranked[:max_vignettes]:
            child = EvidenceSource(
                source_type="Bioconductor vignette HTML",
                title=f"{source.package} vignette: {Path(url).name}",
                url=url,
                package=source.package,
                evidence_role=source.evidence_role,
                query=source.query,
                useful_sections=source.useful_sections,
                functions=source.functions,
            )
            results.append(self.fetch_source(child))
        return results

    def _candidate_vignette_urls(self, source: EvidenceSource) -> List[str]:
        if not source.package:
            return []
        base = source.url.rstrip("/") + "/"
        names = [
            f"{source.package}.html",
            f"{source.package}.Rmd",
            f"{source.package}.pdf",
            "vignette.html",
            "intro.html",
        ]
        return [urljoin(base, name) for name in names]

    def _extract_links(self, html_text: str, base_url: str) -> List[str]:
        links: List[str] = []
        for match in re.finditer(r"""href=["']([^"']+)["']""", html_text, re.I):
            href = html.unescape(match.group(1))
            if href.startswith("#") or href.startswith("mailto:"):
                continue
            links.append(urljoin(base_url, href))
        return list(dict.fromkeys(links))

    def _rank_links(self, links: List[str], source: EvidenceSource) -> List[str]:
        needles = [source.package.lower(), *(fn.lower() for fn in source.functions), "vignette", "workflow", "intro"]

        def score(url: str) -> tuple[int, str]:
            lower = url.lower()
            return (sum(1 for needle in needles if needle and needle in lower), url)

        return sorted(links, key=score, reverse=True)

    def _write_bytes(self, source: EvidenceSource, content: bytes, *, suffix: str) -> Path:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{source.package}_{source.source_type}_{Path(source.url).name or 'index'}")
        if not stem or stem == "_":
            stem = self._sha256(source.url.encode("utf-8"))[:16]
        if stem.lower().endswith(suffix.lower()):
            stem = stem[: -len(suffix)]
        path = self.cache_dir / f"{stem[:120]}{suffix}"
        path.write_bytes(content)
        return path

    def _html_to_text(self, html_text: str) -> str:
        text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html_text)
        text = re.sub(r"(?is)<br\s*/?>", "\n", text)
        text = re.sub(r"(?is)</(p|div|li|tr|h[1-6]|section|article)>", "\n", text)
        text = re.sub(r"(?is)<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _focused_excerpt(self, text: str, source: EvidenceSource, *, max_chars: int = 3500) -> str:
        if not text:
            return ""
        terms = [source.package, *source.functions, *source.useful_sections]
        lower = text.lower()
        windows: List[str] = []
        for term in terms:
            if not term:
                continue
            idx = lower.find(term.lower())
            if idx < 0:
                continue
            start = max(0, idx - 600)
            end = min(len(text), idx + 1400)
            windows.append(text[start:end].strip())
            if sum(len(w) for w in windows) >= max_chars:
                break
        if not windows:
            return text[:max_chars].strip()
        excerpt = "\n\n...\n\n".join(windows)
        return excerpt[:max_chars].strip()

    def _pdf_to_text(self, path: Path, *, max_pages: int = 8) -> str:
        try:
            import pdfplumber  # type: ignore

            chunks: List[str] = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages[:max_pages]:
                    chunks.append(page.extract_text() or "")
            return "\n".join(chunks).strip()
        except Exception:
            return ""

    def _sha256(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()
