#!/usr/bin/env python3
"""Paper Search Module - Retrieve papers from PubMed, Europe PMC, bioRxiv.

This module provides search functionality for academic papers from multiple sources:
- PubMed (NCBI E-utilities)
- Europe PMC
- bioRxiv

Usage:
    searcher = PubMedSearcher()
    results = searcher.search("DESeq2 differential expression RNA-seq", max_results=10)
    
    for paper in results:
        print(f"{paper.title} - PMID:{paper.pmid}")
"""

import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urlencode, quote_plus
import requests


@dataclass
class PaperMetadata:
    """Metadata for a retrieved paper."""
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    title: str = ""
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    year: Optional[int] = None
    citation_count: Optional[int] = None
    has_fulltext: bool = False
    is_open_access: bool = False
    supplementary_urls: List[str] = field(default_factory=list)
    source: str = ""  # pubmed, europepmc, biorxiv
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "doi": self.doi,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "citation_count": self.citation_count,
            "has_fulltext": self.has_fulltext,
            "is_open_access": self.is_open_access,
            "supplementary_urls": self.supplementary_urls,
            "source": self.source,
        }


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_second: float = 3.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
    
    def wait(self):
        """Wait if needed to maintain rate limit."""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


class PubMedSearcher:
    """Search PubMed using NCBI E-utilities API.
    
    NCBI E-utilities documentation:
    https://www.ncbi.nlm.nih.gov/books/NBK25501/
    
    Rate limit: 3 requests per second without API key
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        """Initialize searcher.
        
        Args:
            api_key: NCBI API key for higher rate limits (10 requests/second)
            email: Email for NCBI to contact if issues arise
        """
        self.api_key = api_key
        self.email = email
        self.rate_limiter = RateLimiter(calls_per_second=10.0 if api_key else 3.0)
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> requests.Response:
        """Make rate-limited request to NCBI API."""
        self.rate_limiter.wait()
        
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
            
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response
    
    def search(self, query: str, max_results: int = 10, 
               sort: str = "relevance") -> Dict[str, Any]:
        """Search PubMed for papers matching query.
        
        Args:
            query: PubMed search query (supports full PubMed syntax)
            max_results: Maximum number of results to return
            sort: Sort order ("relevance", "pub_date", etc.)
            
        Returns:
            Dictionary with "count" (total results) and "papers" (list of PaperMetadata)
        """
        # First, perform ESearch to get PMIDs
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": sort,
            "retmode": "json",
        }
        
        response = self._make_request("esearch.fcgi", search_params)
        search_data = response.json()
        
        total_count = int(search_data.get("esearchresult", {}).get("count", 0))
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        
        if not pmids:
            return {"count": 0, "papers": []}
        
        # Then, fetch details for these PMIDs
        papers = self.fetch_papers_by_pmids(pmids)
        
        return {
            "count": total_count,
            "papers": papers
        }
    
    def fetch_papers_by_pmids(self, pmids: List[str]) -> List[PaperMetadata]:
        """Fetch full metadata for a list of PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of PaperMetadata objects
        """
        if not pmids:
            return []
        
        # EFetch to get full details
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        
        response = self._make_request("efetch.fcgi", fetch_params)
        
        # Parse XML response
        root = ET.fromstring(response.content)
        papers = []
        
        for article in root.findall(".//PubmedArticle"):
            paper = self._parse_pubmed_article(article)
            papers.append(paper)
        
        return papers
    
    def _parse_pubmed_article(self, article: ET.Element) -> PaperMetadata:
        """Parse PubMed XML article element."""
        paper = PaperMetadata(source="pubmed")
        
        # PMID
        pmid_elem = article.find(".//PMID")
        if pmid_elem is not None:
            paper.pmid = pmid_elem.text
        
        # DOI
        doi_elem = article.find(".//ArticleId[@IdType='doi']")
        if doi_elem is not None:
            paper.doi = doi_elem.text
        
        # Title
        title_elem = article.find(".//ArticleTitle")
        if title_elem is not None:
            paper.title = "".join(title_elem.itertext()).strip()
        
        # Abstract
        abstract_elems = article.findall(".//AbstractText")
        if abstract_elems:
            paper.abstract = " ".join(
                "".join(elem.itertext()) for elem in abstract_elems
            ).strip()
        
        # Authors
        author_elems = article.findall(".//Author")
        for author in author_elems:
            lastname = author.find("LastName")
            forename = author.find("ForeName")
            if lastname is not None:
                name = lastname.text or ""
                if forename is not None and forename.text:
                    name = f"{forename.text} {name}"
                paper.authors.append(name)
        
        # Journal
        journal_elem = article.find(".//Journal/Title")
        if journal_elem is not None:
            paper.journal = journal_elem.text or ""
        
        # Year
        year_elem = article.find(".//PubDate/Year")
        if year_elem is not None:
            try:
                paper.year = int(year_elem.text)
            except (ValueError, TypeError):
                pass
        
        # Check for PMC ID (has full text)
        pmc_elem = article.find(".//ArticleId[@IdType='pmc']")
        if pmc_elem is not None:
            paper.pmcid = pmc_elem.text
            paper.has_fulltext = True
        
        return paper
    
    def fetch_abstract(self, pmid: str) -> Optional[str]:
        """Fetch abstract for a specific PMID.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            Abstract text or None
        """
        papers = self.fetch_papers_by_pmids([pmid])
        if papers:
            return papers[0].abstract
        return None
    
    def get_pmcid(self, pmid: str) -> Optional[str]:
        """Get PMC ID for a PMID (if available).
        
        Args:
            pmid: PubMed ID
            
        Returns:
            PMC ID or None if not in PMC
        """
        # Use ELink to check PMC
        params = {
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": pmid,
            "retmode": "json",
        }
        
        response = self._make_request("elink.fcgi", params)
        data = response.json()
        
        # Extract PMC ID from response
        linksets = data.get("linksets", [])
        for linkset in linksets:
            for link in linkset.get("linksetdbs", []):
                if link.get("dbto") == "pmc":
                    ids = link.get("ids", [])
                    if ids:
                        return f"PMC{ids[0]}"
        
        return None


class EuropePMCSearcher:
    """Search Europe PMC for papers.
    
    Europe PMC REST API documentation:
    https://www.europepmc.org/RestfulWebService
    
    Europe PMC has better coverage of open access papers and
    provides direct links to PMC PDFs.
    """
    
    BASE_URL = "https://www.europepmc.org/RestfulWebService"
    
    def __init__(self):
        self.rate_limiter = RateLimiter(calls_per_second=10.0)
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> requests.Response:
        """Make rate-limited request to Europe PMC API."""
        self.rate_limiter.wait()
        
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response
    
    def search(self, query: str, max_results: int = 10,
               open_access_only: bool = False) -> List[PaperMetadata]:
        """Search Europe PMC.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            open_access_only: Only return open access papers
            
        Returns:
            List of PaperMetadata objects
        """
        params = {
            "query": query,
            "format": "json",
            "pageSize": min(max_results, 1000),  # Max 1000 per request
        }
        
        if open_access_only:
            params["query"] += " OPEN_ACCESS:y"
        
        response = self._make_request("search", params)
        data = response.json()
        
        results = data.get("resultList", {}).get("result", [])
        
        papers = []
        for item in results[:max_results]:
            paper = self._parse_result(item)
            papers.append(paper)
        
        return papers
    
    def _parse_result(self, item: Dict) -> PaperMetadata:
        """Parse Europe PMC result item."""
        paper = PaperMetadata(source="europepmc")
        
        paper.pmid = item.get("pmid")
        paper.pmcid = item.get("pmcid")
        paper.doi = item.get("doi")
        paper.title = item.get("title", "")
        paper.abstract = item.get("abstractText", "")
        paper.journal = item.get("journalTitle", "")
        paper.citation_count = item.get("citedByCount")
        
        # Authors
        author_string = item.get("authorString", "")
        if author_string:
            paper.authors = [a.strip() for a in author_string.split(",")]
        
        # Year
        year_str = item.get("pubYear")
        if year_str:
            try:
                paper.year = int(year_str)
            except (ValueError, TypeError):
                pass
        
        # Open access / full text
        paper.is_open_access = item.get("isOpenAccess", "N") == "Y"
        paper.has_fulltext = paper.pmcid is not None
        
        return paper
    
    def get_pmc_pdf_url(self, pmcid: str) -> Optional[str]:
        """Get direct PDF URL for PMC article.
        
        Args:
            pmcid: PMC ID (with or without "PMC" prefix)
            
        Returns:
            Direct PDF URL or None
        """
        # Normalize PMC ID
        if pmcid.startswith("PMC"):
            pmcid = pmcid[3:]
        
        # Europe PMC provides direct PDF links
        return f"https://www.europepmc.org/backend/ptpmcrender.fcgi?accid=PMC{pmcid}&blobtype=pdf"


class BioRxivSearcher:
    """Search bioRxiv for preprints.
    
    bioRxiv API documentation:
    https://api.biorxiv.org/
    
    Note: bioRxiv contains preprints that may not be peer-reviewed,
    but often have full text available.
    """
    
    BASE_URL = "https://api.biorxiv.org"
    
    def __init__(self):
        self.rate_limiter = RateLimiter(calls_per_second=1.0)  # Be conservative
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str) -> requests.Response:
        """Make rate-limited request to bioRxiv API."""
        self.rate_limiter.wait()
        
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response
    
    def search(self, query: str, max_results: int = 10) -> List[PaperMetadata]:
        """Search bioRxiv by text query.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of PaperMetadata objects
        """
        # bioRxiv uses URL-encoded query
        encoded_query = quote_plus(query)
        endpoint = f"content/biorxiv/0/100/{encoded_query}"
        
        response = self._make_request(endpoint)
        data = response.json()
        
        results = data.get("results", [])
        
        papers = []
        for item in results[:max_results]:
            paper = self._parse_result(item)
            papers.append(paper)
        
        return papers
    
    def _parse_result(self, item: Dict) -> PaperMetadata:
        """Parse bioRxiv result item."""
        paper = PaperMetadata(source="biorxiv")
        
        paper.doi = item.get("doi")
        paper.title = item.get("title", "")
        
        # Authors
        authors = item.get("authors", "")
        if authors:
            paper.authors = [a.strip() for a in authors.split(";")]
        
        # Year from date
        date_str = item.get("date", "")
        if date_str:
            try:
                paper.year = int(date_str.split("-")[0])
            except (ValueError, IndexError):
                pass
        
        # bioRxiv preprints are open access
        paper.is_open_access = True
        paper.has_fulltext = True
        
        # bioRxiv doesn't have PMIDs/PMCIDs until published
        
        return paper


class PaperSearchAggregator:
    """Aggregate search results from multiple sources."""
    
    def __init__(self, pubmed_api_key: Optional[str] = None, 
                 pubmed_email: Optional[str] = None):
        self.pubmed = PubMedSearcher(api_key=pubmed_api_key, email=pubmed_email)
        self.europepmc = EuropePMCSearcher()
        self.biorxiv = BioRxivSearcher()
    
    def search_all(self, query: str, max_per_source: int = 5,
                   prefer_open_access: bool = True) -> List[PaperMetadata]:
        """Search all available sources and merge results.
        
        Args:
            query: Search query
            max_per_source: Maximum results from each source
            prefer_open_access: Prioritize open access papers
            
        Returns:
            Merged, deduplicated list of papers
        """
        all_papers = []
        
        # Search PubMed
        try:
            pubmed_results = self.pubmed.search(query, max_results=max_per_source)
            all_papers.extend(pubmed_results.get("papers", []))
        except Exception as e:
            print(f"PubMed search failed: {e}")
        
        # Search Europe PMC
        try:
            europepmc_results = self.europepmc.search(
                query, max_results=max_per_source,
                open_access_only=prefer_open_access
            )
            all_papers.extend(europepmc_results)
        except Exception as e:
            print(f"Europe PMC search failed: {e}")
        
        # Search bioRxiv
        try:
            biorxiv_results = self.biorxiv.search(query, max_results=max_per_source)
            all_papers.extend(biorxiv_results)
        except Exception as e:
            print(f"bioRxiv search failed: {e}")
        
        # Deduplicate by DOI or PMID
        seen = set()
        unique_papers = []
        
        for paper in all_papers:
            key = paper.doi or paper.pmid
            if key and key not in seen:
                seen.add(key)
                unique_papers.append(paper)
            elif not key:
                # Keep papers without identifiers (bioRxiv)
                unique_papers.append(paper)
        
        # Sort by relevance (prefer papers with full text)
        unique_papers.sort(
            key=lambda p: (p.has_fulltext, p.is_open_access, p.citation_count or 0),
            reverse=True
        )
        
        return unique_papers
    
    def fetch_pmc_fulltext(self, pmcid: str) -> Optional[bytes]:
        """Fetch full text PDF from PMC.
        
        Args:
            pmcid: PMC ID (with or without PMC prefix)
            
        Returns:
            PDF content as bytes or None
        """
        # Try Europe PMC first
        pdf_url = self.europepmc.get_pmc_pdf_url(pmcid)
        
        if pdf_url:
            try:
                response = requests.get(pdf_url, timeout=30)
                if response.status_code == 200:
                    return response.content
            except Exception:
                pass
        
        # Fallback: Try direct NCBI link
        if pmcid.startswith("PMC"):
            pmcid = pmcid[3:]
        
        ncbi_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/pdf/"
        
        try:
            response = requests.get(ncbi_url, timeout=30)
            if response.status_code == 200:
                return response.content
        except Exception:
            pass
        
        return None


# Convenience function for quick searches
def search_papers(query: str, max_results: int = 10, 
                  sources: List[str] = None) -> List[PaperMetadata]:
    """Quick search across multiple paper sources.
    
    Args:
        query: Search query
        max_results: Maximum results to return
        sources: List of sources to search ("pubmed", "europepmc", "biorxiv")
        
    Returns:
        List of PaperMetadata objects
    """
    if sources is None:
        sources = ["pubmed", "europepmc"]
    
    aggregator = PaperSearchAggregator()
    
    all_papers = []
    
    for source in sources:
        try:
            if source == "pubmed":
                papers = aggregator.pubmed.search(query, max_results)
            elif source == "europepmc":
                papers = aggregator.europepmc.search(query, max_results)
            elif source == "biorxiv":
                papers = aggregator.biorxiv.search(query, max_results)
            else:
                continue
            
            all_papers.extend(papers)
        except Exception as e:
            print(f"Source {source} failed: {e}")
    
    # Deduplicate and sort
    seen = set()
    unique = []
    for p in all_papers:
        key = p.doi or p.pmid
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
    
    return unique[:max_results]


if __name__ == "__main__":
    # Example usage
    print("Testing PubMed search...")
    
    searcher = PubMedSearcher()
    results = searcher.search("DESeq2 differential expression RNA-seq", max_results=5)
    
    for paper in results:
        print(f"\nTitle: {paper.title}")
        print(f"PMID: {paper.pmid}")
        print(f"PMCID: {paper.pmcid}")
        print(f"Year: {paper.year}")
        print(f"Has full text: {paper.has_fulltext}")
