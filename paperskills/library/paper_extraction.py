#!/usr/bin/env python3
"""Paper Content Extraction Module - Extract methods and code from papers.

This module provides functionality to:
- Download PDFs from PMC and other sources
- Extract text from PDFs
- Identify and extract Methods sections
- Extract code snippets from papers and supplementary materials

Usage:
    extractor = PDFExtractor()
    pdf_content = extractor.download_from_pmc("PMC1234567")
    text = extractor.extract_text(pdf_content)
    
    method_extractor = MethodSectionExtractor()
    methods = method_extractor.extract(text)
"""

import io
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import quote, urljoin

import requests

# Optional imports with fallbacks
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False


@dataclass
class ExtractedContent:
    """Content extracted from a paper."""
    methods_text: str = ""
    code_snippets: List[str] = field(default_factory=list)
    supplementary_files: List[str] = field(default_factory=list)
    full_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeSnippet:
    """A code snippet extracted from a paper."""
    language: str  # "R", "Python", "bash", etc.
    code: str
    context: str = ""  # Surrounding text for context
    source: str = ""  # "main_text", "supplementary", "figure_caption"
    page_number: Optional[int] = None


class PDFExtractor:
    """Download and extract content from PDF papers."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize extractor.
        
        Args:
            cache_dir: Directory to cache downloaded PDFs
        """
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(".cache/papers")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PaperToSkillBot/1.0 (Academic Research)"
        })
        self.last_access_level: str = ""
        self.last_source_url: str = ""
        self.last_local_path: str = ""
        self.last_attempts: List[Dict[str, Any]] = []
        self.last_source_portfolio: List[Dict[str, Any]] = []

    def _record_source(self, *, access_level: str, source_url: str = "", local_path: Path | None = None) -> None:
        self.last_access_level = access_level
        self.last_source_url = source_url
        self.last_local_path = str(local_path) if local_path else ""
        if access_level:
            self.last_source_portfolio.append({
                "access_level": access_level,
                "source_url": source_url,
                "local_path": str(local_path) if local_path else "",
            })

    def _record_attempt(
        self,
        *,
        route: str,
        url: str,
        success: bool,
        access_level: str = "",
        failure_reason: str = "",
        status_code: int | None = None,
        content_type: str = "",
    ) -> None:
        self.last_attempts.append({
            "route": route,
            "url": url,
            "success": success,
            "access_level": access_level,
            "failure_reason": failure_reason,
            "status_code": status_code,
            "content_type": content_type,
        })

    def _download_pdf_url(
        self,
        url: str,
        cache_file: Path,
        *,
        access_level: str,
        route: str,
        timeout: int = 60,
    ) -> Optional[bytes]:
        """Download a URL only if the response is a real PDF."""
        try:
            resp = self.session.get(
                url,
                timeout=timeout,
                allow_redirects=True,
                headers={"Accept": "application/pdf,text/html;q=0.8"},
            )
            if resp.status_code == 200 and resp.content[:4] == b'%PDF':
                cache_file.write_bytes(resp.content)
                self._record_attempt(
                    route=route,
                    url=url,
                    success=True,
                    access_level=access_level,
                    status_code=resp.status_code,
                    content_type=resp.headers.get("Content-Type", ""),
                )
                self._record_source(access_level=access_level, source_url=url, local_path=cache_file)
                return resp.content
            self._record_attempt(
                route=route,
                url=url,
                success=False,
                failure_reason="not_pdf_content" if resp.status_code == 200 else "http_error",
                status_code=resp.status_code,
                content_type=resp.headers.get("Content-Type", ""),
            )
        except Exception as exc:
            self._record_attempt(route=route, url=url, success=False, failure_reason=f"exception:{type(exc).__name__}")
        return None

    def _discover_pdf_links_from_landing(self, landing_url: str) -> List[str]:
        """Find likely PDF links from a publisher landing page."""
        try:
            resp = self.session.get(
                landing_url,
                timeout=30,
                allow_redirects=True,
                headers={"Accept": "text/html,application/xhtml+xml"},
            )
            if resp.status_code != 200 or "html" not in resp.headers.get("Content-Type", "").lower():
                return []
            html = resp.text
        except Exception:
            return []

        links = []
        patterns = [
            r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
            r'href=["\']([^"\']*pdf[^"\']*)["\']',
        ]
        for pattern in patterns:
            for href in re.findall(pattern, html, flags=re.I):
                if href and "javascript:" not in href.lower():
                    links.append(urljoin(resp.url, href))

        cleaned = []
        seen = set()
        for link in links:
            key = link.split("#", 1)[0]
            if key not in seen:
                seen.add(key)
                cleaned.append(key)
        return cleaned[:10]
    
    def download_from_pmc(self, pmcid: str) -> Optional[bytes]:
        """Download full text from PubMed Central.

        Tries routes in order:
        1. PMC/Europe PMC PDF (preferred for Paper2Skills-style reading)
        2. PMC OA XML / Europe PMC full text XML (structured full text)
        
        Args:
            pmcid: PMC ID (with or without "PMC" prefix)
            
        Returns:
            Text content as bytes (XML or PDF) or None if all routes fail
        """
        # Normalize PMC ID
        if pmcid.startswith("PMC"):
            pmcid_num = pmcid[3:]
        else:
            pmcid_num = pmcid
        
        self._record_source(access_level="")
        self.last_attempts = []
        self.last_source_portfolio = []

        cache_xml = self.cache_dir / f"PMC{pmcid_num}.xml"
        cache_pdf = self.cache_dir / f"PMC{pmcid_num}.pdf"
        if cache_pdf.exists():
            content = cache_pdf.read_bytes()
            if content[:4] == b'%PDF':
                self._record_attempt(route="pmc_pdf_cache", url=str(cache_pdf), success=True, access_level="pmc_pdf_fulltext")
                self._record_source(access_level="pmc_pdf_fulltext", local_path=cache_pdf)
                return content
            else:
                # Corrupt cache - delete and re-download
                cache_pdf.unlink()

        # --- Route 1: Publisher/PMC PDF (best for source-aware reading) ---
        pdf_urls = [
            f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid_num}/pdf/",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid_num}/pdf/",
            f"https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC{pmcid_num}&blobtype=pdf",
        ]
        for url in pdf_urls:
            content = self._download_pdf_url(
                url,
                cache_pdf,
                access_level="pmc_pdf_fulltext",
                route="pmc_pdf",
            )
            if content:
                return content
            time.sleep(0.5)

        # Check XML cache only after PDF attempts. XML is still useful full
        # text, but for this benchmark it is a fallback behind PDF.
        if cache_xml.exists() and cache_xml.stat().st_size > 5000:
            self._record_attempt(route="pmc_xml_cache", url=str(cache_xml), success=True, access_level="pmc_xml_fulltext")
            self._record_source(access_level="pmc_xml_fulltext", local_path=cache_xml)
            return cache_xml.read_bytes()

        # --- Route 2: PMC OA XML (structured full text, open access) ---
        try:
            xml_url = (
                f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
                f"?verb=GetRecord"
                f"&identifier=oai:pubmedcentral.nih.gov:{pmcid_num}"
                f"&metadataPrefix=pmc"
            )
            resp = self.session.get(xml_url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 5000:
                # Verify it's XML, not an error page
                text_start = resp.content[:200].decode('utf-8', errors='ignore')
                if '<?xml' in text_start or '<OAI-PMH' in text_start:
                    cache_xml.write_bytes(resp.content)
                    self._record_attempt(
                        route="pmc_oai_xml",
                        url=xml_url,
                        success=True,
                        access_level="pmc_xml_fulltext",
                        status_code=resp.status_code,
                        content_type=resp.headers.get("Content-Type", ""),
                    )
                    self._record_source(access_level="pmc_xml_fulltext", source_url=xml_url, local_path=cache_xml)
                    return resp.content
                self._record_attempt(route="pmc_oai_xml", url=xml_url, success=False, failure_reason="not_xml_content", status_code=resp.status_code, content_type=resp.headers.get("Content-Type", ""))
            else:
                self._record_attempt(route="pmc_oai_xml", url=xml_url, success=False, failure_reason="http_error", status_code=resp.status_code, content_type=resp.headers.get("Content-Type", ""))
        except Exception:
            self._record_attempt(route="pmc_oai_xml", url=xml_url, success=False, failure_reason="exception")
        
        time.sleep(0.3)
        
        # --- Route 2: Europe PMC full text XML ---
        try:
            epmc_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/PMC{pmcid_num}/fullTextXML"
            resp = self.session.get(epmc_url, timeout=30,
                                    headers={"Accept": "application/xml"})
            if resp.status_code == 200 and len(resp.content) > 5000:
                text_start = resp.content[:200].decode('utf-8', errors='ignore')
                if '<?xml' in text_start or '<article' in text_start:
                    cache_xml.write_bytes(resp.content)
                    self._record_attempt(
                        route="europepmc_xml",
                        url=epmc_url,
                        success=True,
                        access_level="europepmc_xml_fulltext",
                        status_code=resp.status_code,
                        content_type=resp.headers.get("Content-Type", ""),
                    )
                    self._record_source(access_level="europepmc_xml_fulltext", source_url=epmc_url, local_path=cache_xml)
                    return resp.content
                self._record_attempt(route="europepmc_xml", url=epmc_url, success=False, failure_reason="not_xml_content", status_code=resp.status_code, content_type=resp.headers.get("Content-Type", ""))
            else:
                self._record_attempt(route="europepmc_xml", url=epmc_url, success=False, failure_reason="http_error", status_code=resp.status_code, content_type=resp.headers.get("Content-Type", ""))
        except Exception:
            self._record_attempt(route="europepmc_xml", url=epmc_url, success=False, failure_reason="exception")

        return None
    
    def download_from_doi(self, doi: str) -> Optional[bytes]:
        """Download PDF using DOI resolver.
        
        This attempts to find the PDF through various open access routes.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            PDF content as bytes or None
        """
        # Check cache
        self._record_source(access_level="")
        cache_file = self.cache_dir / f"{doi.replace('/', '_')}.pdf"
        if cache_file.exists():
            content = cache_file.read_bytes()
            if content[:4] == b'%PDF':
                self._record_attempt(route="doi_pdf_cache", url=str(cache_file), success=True, access_level="publisher_pdf_fulltext")
                self._record_source(access_level="publisher_pdf_fulltext", local_path=cache_file)
                return content
            cache_file.unlink()
        
        # Try common publisher PDF routes. These are not task-specific; they
        # cover common OA method-paper publishers that expose DOI-addressed PDFs.
        doi_quoted = quote(doi, safe="/:")
        direct_pdf_urls = [
            f"https://link.springer.com/content/pdf/{doi_quoted}.pdf",
            f"https://www.frontiersin.org/articles/{doi_quoted}/pdf",
            f"https://journals.plos.org/plosone/article/file?id={doi_quoted}&type=printable",
        ]
        for pdf_url in direct_pdf_urls:
            content = self._download_pdf_url(
                pdf_url,
                cache_file,
                access_level="publisher_pdf_fulltext",
                route="publisher_pdf_pattern",
            )
            if content:
                return content
            time.sleep(0.3)

        # Try Unpaywall / OpenAlex
        landing_pages = []
        try:
            # First, resolve DOI to get potential PDF links
            headers = {"Accept": "application/json"}
            response = self.session.get(
                f"https://api.unpaywall.org/v2/{doi}?email=paper2skill@example.com",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for best open access location
                best_oa = data.get("best_oa_location")
                if best_oa:
                    pdf_url = best_oa.get("url_for_pdf")
                    if pdf_url:
                        content = self._download_pdf_url(
                            pdf_url,
                            cache_file,
                            access_level="publisher_pdf_fulltext",
                            route="unpaywall_pdf",
                        )
                        if content:
                            return content
                    if best_oa.get("url_for_landing_page"):
                        landing_pages.append(best_oa["url_for_landing_page"])
                for loc in data.get("oa_locations") or []:
                    if loc.get("url_for_pdf"):
                        content = self._download_pdf_url(
                            loc["url_for_pdf"],
                            cache_file,
                            access_level="publisher_pdf_fulltext",
                            route="unpaywall_pdf",
                        )
                        if content:
                            return content
                    if loc.get("url_for_landing_page"):
                        landing_pages.append(loc["url_for_landing_page"])
        except Exception:
            pass

        landing_pages.append(f"https://doi.org/{doi}")
        for landing in landing_pages:
            for pdf_url in self._discover_pdf_links_from_landing(landing):
                content = self._download_pdf_url(
                    pdf_url,
                    cache_file,
                    access_level="publisher_pdf_fulltext",
                    route="landing_page_pdf",
                )
                if content:
                    return content
                time.sleep(0.3)
        
        return None
    
    def extract_text(self, content: bytes) -> str:
        """Extract text from PDF or XML content.
        
        Automatically detects format:
        - XML (from PMC OA): parse with xml.etree or regex
        - PDF: use pdfplumber/PyPDF2
        
        Args:
            content: File content as bytes (PDF or XML)
            
        Returns:
            Extracted plain text
        """
        # Detect content type
        start = content[:200].decode('utf-8', errors='ignore')
        
        if '<?xml' in start or '<OAI-PMH' in start or '<article' in start:
            return self._extract_from_xml(content)
        elif content[:4] == b'%PDF':
            if HAS_PDFPLUMBER:
                return self._extract_with_pdfplumber(content)
            elif HAS_PYPDF2:
                return self._extract_with_pypdf2(content)
            else:
                raise ImportError("Install pdfplumber or PyPDF2 for PDF extraction.")
        else:
            # Unknown format - try PDF first, then XML
            try:
                if HAS_PDFPLUMBER:
                    return self._extract_with_pdfplumber(content)
            except Exception:
                pass
            return self._extract_from_xml(content)
    
    def _extract_from_xml(self, xml_content: bytes) -> str:
        """Extract readable text from PMC OA XML or Europe PMC XML.

        Prefer JATS article body sections over a flat walk of the full OAI
        record. PMC OAI records include headers, front matter, metadata, and
        references before/after the actual article body; flattening everything
        makes downstream section extraction pick up OAI metadata instead of
        paper content.
        """
        import xml.etree.ElementTree as ET
        
        try:
            root = ET.fromstring(xml_content)
            body = self._find_first_tag(root, "body")
            if body is not None:
                section_texts = []
                for sec in self._iter_direct_sections(body):
                    section_text = self._section_to_text(sec)
                    if section_text:
                        section_texts.append(section_text)
                if section_texts:
                    return "\n\n".join(section_texts)

            # Fallback for non-JATS XML: collect all readable text nodes, but
            # skip metadata/reference-heavy tags that poison method summaries.
            skip_tags = {
                'xref', 'ref', 'ref-list', 'label', 'sup', 'sub',
                'front', 'journal-meta', 'article-meta', 'custom-meta-group',
                'back',
            }
            texts = []
            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if tag in skip_tags:
                    continue
                if elem.text and elem.text.strip():
                    texts.append(elem.text.strip())
                if elem.tail and elem.tail.strip():
                    texts.append(elem.tail.strip())

            return "\n".join(texts)
        except ET.ParseError:
            # Fall back to regex-based extraction
            text = xml_content.decode('utf-8', errors='replace')
            # Remove XML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text

    @staticmethod
    def _local_tag(elem) -> str:
        return elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

    def _find_first_tag(self, root, local_tag: str):
        for elem in root.iter():
            if self._local_tag(elem) == local_tag:
                return elem
        return None

    def _iter_direct_sections(self, body):
        sections = [child for child in list(body) if self._local_tag(child) == "sec"]
        return sections or [body]

    def _section_to_text(self, sec, level: int = 2) -> str:
        title = ""
        parts = []
        skip_tags = {'xref', 'ref', 'label', 'sup', 'sub', 'table-wrap-foot'}

        for child in list(sec):
            tag = self._local_tag(child)
            if tag == "title":
                title = self._element_text(child, skip_tags=skip_tags)
                continue
            if tag == "sec":
                nested = self._section_to_text(child, level=level + 1)
                if nested:
                    parts.append(nested)
                continue
            text = self._element_text(child, skip_tags=skip_tags)
            if text:
                parts.append(text)

        heading = f"{'#' * min(level, 6)} {title.strip()}" if title.strip() else ""
        body = "\n".join(p for p in parts if p)
        return "\n".join(p for p in [heading, body] if p).strip()

    def _element_text(self, elem, *, skip_tags: set[str]) -> str:
        chunks = []
        for node in elem.iter():
            tag = self._local_tag(node)
            if tag in skip_tags:
                continue
            if node.text and node.text.strip():
                chunks.append(node.text.strip())
            if node.tail and node.tail.strip():
                chunks.append(node.tail.strip())
        text = " ".join(chunks)
        return re.sub(r"\s+", " ", text).strip()
    
    def _extract_with_pdfplumber(self, pdf_content: bytes) -> str:
        """Extract text using pdfplumber (better quality)."""
        text_parts = []
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception:
                    continue
        
        return "\n\n".join(text_parts)
    
    def _extract_with_pypdf2(self, pdf_content: bytes) -> str:
        """Extract text using PyPDF2 (fallback)."""
        text_parts = []
        
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception:
                    continue
        except Exception:
            pass
        
        return "\n\n".join(text_parts)
    
    def extract_with_page_numbers(self, pdf_content: bytes) -> List[Tuple[int, str]]:
        """Extract text with page numbers.
        
        Returns:
            List of (page_number, text) tuples
        """
        pages = []
        
        if HAS_PDFPLUMBER:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()
                        if text:
                            pages.append((i, text))
                    except Exception:
                        continue
        elif HAS_PYPDF2:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            for i, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    if text:
                        pages.append((i, text))
                except Exception:
                    continue
        
        return pages


class MethodSectionExtractor:
    """Extract Methods section from paper text."""
    
    # Common section headers (in order of preference)
    METHODS_HEADERS = [
        # Standard headers
        r"Methods",
        r"METHODS",
        r"Materials and Methods",
        r"MATERIALS AND METHODS",
        r"Experimental Procedures",
        r"EXPERIMENTAL PROCEDURES",
        r"Protocol",
        r"PROTOCOL",
        r"Implementation",
        r"IMPLEMENTATION",
        # Variants
        r"Methods and Materials",
        r"Detailed Methods",
        r"Statistical Methods",
        r"Computational Methods",
        r"Bioinformatics Methods",
        r"Data Analysis",
        r"Statistical Analysis",
    ]
    
    # Headers that typically end the Methods section
    END_HEADERS = [
        r"Results",
        r"RESULTS",
        r"Discussion",
        r"DISCUSSION",
        r"Findings",
        r"FINDINGS",
        r"Conclusions",
        r"CONCLUSIONS",
        r"Acknowledgments",
        r"ACKNOWLEDGMENTS",
        r"References",
        r"REFERENCES",
        r"Supplementary",
        r"SUPPLEMENTARY",
    ]
    
    def extract(self, text: str) -> str:
        """Extract Methods section from paper text.
        
        Args:
            text: Full text of the paper
            
        Returns:
            Methods section text or empty string if not found
        """
        # Try to find Methods section using patterns
        methods_text = self._extract_by_headers(text)
        
        if methods_text:
            return methods_text
        
        # Fallback: try to identify methods by keywords
        return self._extract_by_keywords(text)
    
    def _extract_by_headers(self, text: str) -> str:
        """Extract methods using section headers."""
        # Create pattern to match Methods header
        methods_pattern = "|".join(self.METHODS_HEADERS)
        
        # Create pattern for section end
        end_pattern = "|".join(self.END_HEADERS)
        
        # Try to find Methods section
        for start_pattern in self.METHODS_HEADERS:
            # Match header at start of line (possibly with number)
            pattern = rf"(?:^|\n)\s*\d*\s*{re.escape(start_pattern)}\.?\s*\n(.*?)(?:\n\s*\d*\s*(?:{end_pattern})\.?\s*\n|$)"
            
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_by_keywords(self, text: str) -> str:
        """Fallback: extract methods-like content using keywords."""
        # Keywords that indicate methods content
        method_keywords = [
            "software", "algorithm", "implemented", "package",
            "statistical analysis", "differential expression",
            "normalized", "RNA-seq", "microarray", "sequencing",
            "DESeq2", "limma", "edgeR", "R version"
        ]
        
        lines = text.split("\n")
        method_lines = []
        
        for line in lines:
            # Check if line contains method keywords
            if any(kw.lower() in line.lower() for kw in method_keywords):
                method_lines.append(line)
        
        # Return a window of context around method mentions
        if method_lines:
            return "\n".join(method_lines[:50])  # Limit to 50 lines
        
        return ""
    
    def extract_subsections(self, methods_text: str) -> Dict[str, str]:
        """Extract subsections within Methods.
        
        Args:
            methods_text: Text of Methods section
            
        Returns:
            Dictionary mapping subsection name to content
        """
        subsections = {}
        
        # Common subsection patterns
        subsection_pattern = r"\n([A-Z][a-zA-Z\s]+)\n"
        
        matches = list(re.finditer(subsection_pattern, methods_text))
        
        for i, match in enumerate(matches):
            name = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(methods_text)
            
            content = methods_text[start:end].strip()
            if len(content) > 50:  # Filter out false positives
                subsections[name] = content
        
        return subsections


class CodeBlockExtractor:
    """Extract code snippets from paper text and PDFs."""
    
    # Patterns for identifying code in text
    CODE_PATTERNS = [
        # R code patterns
        ("R", r"library\([a-zA-Z_]+\)"),
        ("R", r"<-\s+[a-zA-Z_\.\(\)\"\']+"),
        ("R", r"function\s*\([^)]*\)\s*\{"),
        # Python patterns
        ("Python", r"import\s+[a-zA-Z_]+"),
        ("Python", r"def\s+[a-zA-Z_]+\s*\("),
        ("Python", r"print\s*\("),
        # Shell/bash patterns
        ("bash", r"^\$\s+"),
        ("bash", r"#!/bin/bash"),
        ("bash", r"wget\s+http"),
    ]
    
    def extract_from_text(self, text: str) -> List[CodeSnippet]:
        """Extract code snippets from text.
        
        Args:
            text: Text to search for code
            
        Returns:
            List of CodeSnippet objects
        """
        snippets = []
        
        # Look for code blocks (indented or in code sections)
        lines = text.split("\n")
        current_block = []
        current_lang = ""
        
        for i, line in enumerate(lines):
            # Check if line looks like code
            lang = self._identify_language(line)
            
            if lang:
                if not current_block:
                    current_lang = lang
                
                if lang == current_lang or not current_lang:
                    current_block.append(line)
                else:
                    # Language changed, save current block
                    if len(current_block) >= 2:
                        snippets.append(CodeSnippet(
                            language=current_lang,
                            code="\n".join(current_block),
                            context="",  # Could extract surrounding text
                            source="main_text"
                        ))
                    current_block = [line]
                    current_lang = lang
            else:
                # Not a code line
                if len(current_block) >= 2:
                    snippets.append(CodeSnippet(
                        language=current_lang,
                        code="\n".join(current_block),
                        context="",
                        source="main_text"
                    ))
                current_block = []
                current_lang = ""
        
        # Don't forget the last block
        if len(current_block) >= 2:
            snippets.append(CodeSnippet(
                language=current_lang,
                code="\n".join(current_block),
                context="",
                source="main_text"
            ))
        
        return snippets
    
    def _identify_language(self, line: str) -> str:
        """Identify programming language of a code line."""
        for lang, pattern in self.CODE_PATTERNS:
            if re.search(pattern, line):
                return lang
        
        # Heuristic: check for common R patterns
        if re.search(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*<-", line):
            return "R"
        
        return ""
    
    def extract_from_supplementary(self, supplementary_text: str) -> List[CodeSnippet]:
        """Extract code from supplementary materials.
        
        Supplementary materials often contain full scripts.
        
        Args:
            supplementary_text: Text from supplementary file
            
        Returns:
            List of CodeSnippet objects
        """
        snippets = []
        
        # Look for complete script files in supplementary
        # Often marked with file names like "script.R" or "analysis.py"
        
        file_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*\.(R|r|py|sh))\s*\n(.*?)(?=\n[a-zA-Z_][a-zA-Z0-9_]*\.(R|r|py|sh)|$)"
        
        for match in re.finditer(file_pattern, supplementary_text, re.DOTALL):
            filename = match.group(1)
            code = match.group(3).strip()
            
            # Determine language from extension
            if filename.endswith((".R", ".r")):
                lang = "R"
            elif filename.endswith(".py"):
                lang = "Python"
            elif filename.endswith(".sh"):
                lang = "bash"
            else:
                lang = "unknown"
            
            snippets.append(CodeSnippet(
                language=lang,
                code=code,
                context=f"File: {filename}",
                source="supplementary"
            ))
        
        return snippets


class PaperProcessor:
    """High-level processor for extracting skills from papers."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.pdf_extractor = PDFExtractor(cache_dir)
        self.method_extractor = MethodSectionExtractor()
        self.code_extractor = CodeBlockExtractor()
    
    def process_pmc_paper(self, pmcid: str) -> ExtractedContent:
        """Process a paper from PMC.
        
        Fetches full text via XML (preferred) or PDF,
        then extracts methods section and code snippets.
        
        Args:
            pmcid: PMC ID (e.g. "PMC4053721" or "4053721")
            
        Returns:
            ExtractedContent object with full_text, methods_text, code_snippets
        """
        content = ExtractedContent()
        
        # Download content (XML preferred over PDF)
        raw_content = self.pdf_extractor.download_from_pmc(pmcid)
        if not raw_content:
            content.metadata["error"] = f"Could not fetch content for {pmcid}"
            return content
        
        # Extract full text (handles both XML and PDF automatically)
        try:
            full_text = self.pdf_extractor.extract_text(raw_content)
        except Exception as e:
            content.metadata["error"] = f"Text extraction failed: {e}"
            return content
        
        content.full_text = full_text
        content.metadata["text_length"] = len(full_text)
        content.metadata["pmcid"] = pmcid
        
        # Extract Methods section
        methods = self.method_extractor.extract(full_text)
        content.methods_text = methods
        
        # Extract code snippets
        code_snippets = self.code_extractor.extract_from_text(full_text)
        content.code_snippets = [s.code for s in code_snippets]
        
        return content
    
    def process_paper(self, paper_metadata: Dict[str, Any]) -> ExtractedContent:
        """Process a paper given its metadata.
        
        Args:
            paper_metadata: Paper metadata with pmcid, doi, etc.
            
        Returns:
            ExtractedContent object
        """
        pmcid = paper_metadata.get("pmcid")
        doi = paper_metadata.get("doi")
        
        if pmcid:
            return self.process_pmc_paper(pmcid)
        
        # Try DOI if no PMC ID
        if doi:
            content = ExtractedContent()
            pdf_content = self.pdf_extractor.download_from_doi(doi)
            
            if pdf_content:
                full_text = self.pdf_extractor.extract_text(pdf_content)
                content.full_text = full_text
                content.methods_text = self.method_extractor.extract(full_text)
                code_snippets = self.code_extractor.extract_from_text(full_text)
                content.code_snippets = [s.code for s in code_snippets]
            
            return content
        
        return ExtractedContent()


# Convenience functions
def extract_methods_from_pmc(pmcid: str, cache_dir: Optional[Path] = None) -> str:
    """Quick method extraction from PMC paper.
    
    Args:
        pmcid: PMC ID
        cache_dir: Optional cache directory
        
    Returns:
        Methods section text
    """
    processor = PaperProcessor(cache_dir)
    content = processor.process_pmc_paper(pmcid)
    return content.methods_text


def extract_code_from_pmc(pmcid: str, cache_dir: Optional[Path] = None) -> List[str]:
    """Quick code extraction from PMC paper.
    
    Args:
        pmcid: PMC ID
        cache_dir: Optional cache directory
        
    Returns:
        List of code snippets
    """
    processor = PaperProcessor(cache_dir)
    content = processor.process_pmc_paper(pmcid)
    return content.code_snippets


if __name__ == "__main__":
    # Example usage
    print("Testing paper extraction...")
    
    # Test method extraction
    sample_text = """
Introduction
This is the intro.

Methods
We used DESeq2 for differential expression analysis.
library(DESeq2)
data <- read.csv("counts.csv")

Results
We found many DEGs.
    """
    
    extractor = MethodSectionExtractor()
    methods = extractor.extract(sample_text)
    print(f"\nExtracted Methods:\n{methods}")
    
    # Test code extraction
    code_extractor = CodeBlockExtractor()
    code = code_extractor.extract_from_text(sample_text)
    print(f"\nExtracted Code ({len(code)} snippets):")
    for i, snippet in enumerate(code):
        print(f"Snippet {i+1} ({snippet.language}):")
        print(snippet.code[:100] + "..." if len(snippet.code) > 100 else snippet.code)
