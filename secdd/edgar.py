"""
SEC EDGAR integration for fetching filings.
Uses the EDGAR full-text search API and EFTS for document retrieval.
No API key needed, EDGAR is free and public. Just needs a User-Agent header.
"""

import asyncio
import html.parser
import logging
import os
import re
import httpx
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_ALLOWED_EDGAR_HOSTS = {"www.sec.gov", "data.sec.gov", "efts.sec.gov"}

# SEC requires a User-Agent identifying you. Set EDGAR_USER_AGENT env var.
_default_user_agent = "secdd/0.1.0 (https://github.com/siddvoh/secdd)"
_user_agent = os.getenv("EDGAR_USER_AGENT", _default_user_agent)
if _user_agent == _default_user_agent:
    logger.warning("EDGAR_USER_AGENT not set; using default. SEC recommends setting a contact email.")

EDGAR_HEADERS = {
    "User-Agent": _user_agent,
    "Accept-Encoding": "gzip, deflate",
}

# Rate limit: SEC asks for max 10 requests/second
RATE_LIMIT_SECONDS = 0.12  # ~8 req/s to be safe

_rate_limit_lock = asyncio.Lock()
_last_request_time = 0.0


async def _rate_limit():
    global _last_request_time
    async with _rate_limit_lock:
        now = asyncio.get_event_loop().time()
        wait = RATE_LIMIT_SECONDS - (now - _last_request_time)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_request_time = asyncio.get_event_loop().time()


# Shared httpx client for connection pooling
_shared_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    global _shared_client
    async with _client_lock:
        if _shared_client is None or _shared_client.is_closed:
            _shared_client = httpx.AsyncClient(
                headers=EDGAR_HEADERS,
                follow_redirects=True,
                timeout=30,
            )
    return _shared_client


def _validate_edgar_url(url: str) -> None:
    """Validate that a URL points to an allowed EDGAR host."""
    parsed = urlparse(url)
    if parsed.hostname not in _ALLOWED_EDGAR_HOSTS:
        raise ValueError(f"URL host '{parsed.hostname}' is not an allowed EDGAR host")


@dataclass
class Filing:
    company_name: str
    cik: str
    ticker: str
    form_type: str
    filed_date: str
    accession_number: str
    primary_doc_url: str
    text: str = ""


def normalize_cik(cik: str) -> str:
    """Zero-pad CIK to 10 digits as EDGAR expects."""
    return cik.zfill(10)


def _match_filings_by_year(filings: list[dict], form_type: str, year: int) -> list[dict]:
    """Match filings by fiscal year. For 10-K, tries year+1 first (fiscal year lag)."""
    year_str = str(year)
    if form_type.upper() == "10-K":
        matching = [f for f in filings if f["filing_date"].startswith(str(year + 1))]
        if not matching:
            matching = [f for f in filings if f["filing_date"].startswith(year_str)]
    else:
        matching = [f for f in filings if f["filing_date"].startswith(year_str)]
    return matching


def _build_filing_url(cik: str, accession_number: str, primary_document: str) -> str:
    """Build the EDGAR URL for a filing document."""
    cik_padded = normalize_cik(cik)
    acc_no_dashes = accession_number.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_no_dashes}/{primary_document}"
    _validate_edgar_url(url)
    return url


async def get_cik_from_ticker(ticker: str) -> dict:
    """
    Look up CIK and company name from ticker using EDGAR company tickers JSON.
    Returns {"cik": "0001234567", "name": "Company Name", "ticker": "AAPL"}
    """
    await _rate_limit()
    client = await _get_client()
    resp = await client.get(
        "https://www.sec.gov/files/company_tickers.json",
        timeout=15,
    )
    resp.raise_for_status()

    data = resp.json()
    ticker_upper = ticker.upper().strip()

    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return {
                "cik": normalize_cik(str(entry["cik_str"])),
                "name": entry.get("title", ""),
                "ticker": ticker_upper,
            }

    raise ValueError(f"Ticker '{ticker}' not found in EDGAR")


def _extract_filings_from_section(section: dict, form_type_upper: str, count: int) -> list[dict]:
    """Extract matching filings from a submissions section (recent or older file)."""
    forms = section.get("form", [])
    accession_numbers = section.get("accessionNumber", [])
    filing_dates = section.get("filingDate", [])
    primary_docs = section.get("primaryDocument", [])

    results = []
    for i, form in enumerate(forms):
        if form.upper() == form_type_upper:
            results.append({
                "accession_number": accession_numbers[i],
                "filing_date": filing_dates[i],
                "primary_document": primary_docs[i],
                "form": form,
            })
            if len(results) >= count:
                break
    return results


async def get_recent_filings(
    cik: str,
    form_type: str = "10-K",
    count: int = 3,
) -> list[dict]:
    """
    Fetch filing metadata from EDGAR submissions API.
    Searches the 'recent' section first, then paginates into older filing
    files if not enough results are found (big filers like META have 1000+
    recent entries that only cover ~2 years).
    """
    await _rate_limit()
    cik_padded = normalize_cik(cik)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    client = await _get_client()
    resp = await client.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    form_type_upper = form_type.upper()

    # 1. Try the 'recent' section first
    recent = data.get("filings", {}).get("recent", {})
    results = _extract_filings_from_section(recent, form_type_upper, count)

    if len(results) >= count:
        return results

    # 2. If not enough, fetch older filing files
    older_files = data.get("filings", {}).get("files", [])
    for file_info in older_files:
        if len(results) >= count:
            break
        fname = file_info.get("name", "")
        if not fname:
            continue
        await _rate_limit()
        older_url = f"https://data.sec.gov/submissions/{fname}"
        older_resp = await client.get(older_url, timeout=15)
        older_resp.raise_for_status()
        older_data = older_resp.json()
        remaining = count - len(results)
        results.extend(_extract_filings_from_section(older_data, form_type_upper, remaining))

    return results


def _available_years(filings: list[dict]) -> str:
    """Extract unique years from filings for error messages."""
    years = sorted(set(f["filing_date"][:4] for f in filings), reverse=True)
    return ", ".join(years[:10]) + ("..." if len(years) > 10 else "")


async def fetch_filing_text(cik: str, accession_number: str, primary_document: str) -> str:
    """
    Download the actual filing document text from EDGAR.
    Tries the primary document first, falls back to full submission txt.
    Strips HTML tags for cleaner text processing.
    """
    # Validate primary_document to prevent path traversal / SSRF
    if not re.match(r'^[\w\-\.]+$', primary_document):
        raise ValueError(f"Invalid primary_document name: {primary_document!r}")

    await _rate_limit()
    url = _build_filing_url(cik, accession_number, primary_document)

    client = await _get_client()
    resp = await client.get(url, timeout=30)
    resp.raise_for_status()
    raw = resp.text

    # Basic HTML stripping, SEC filings are often HTML
    text = _strip_html(raw)

    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


class _HTMLTextExtractor(html.parser.HTMLParser):
    """Extract visible text from HTML, skipping script/style/ix:header and display:none elements."""

    _SKIP_TAGS = frozenset({"script", "style", "ix:header"})

    def __init__(self):
        super().__init__()
        self._pieces: list[str] = []
        self._skip_stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()
        if tag_lower in self._SKIP_TAGS:
            self._skip_stack.append(tag_lower)
            return
        # Check for display:none in style attribute
        for attr_name, attr_value in attrs:
            if attr_name == "style" and attr_value and "display" in attr_value:
                if re.search(r'display\s*:\s*none', attr_value, re.IGNORECASE):
                    self._skip_stack.append(tag_lower)
                    return

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if self._skip_stack and self._skip_stack[-1] == tag_lower:
            self._skip_stack.pop()

    def handle_data(self, data):
        if not self._skip_stack:
            self._pieces.append(data)

    def get_text(self) -> str:
        import html as html_mod
        return html_mod.unescape(" ".join(self._pieces))


def _strip_html(html_content: str) -> str:
    """Strip HTML from SEC filings, including iXBRL metadata."""
    if len(html_content) > 20_000_000:
        raise ValueError("HTML content exceeds 20 MB size limit")
    extractor = _HTMLTextExtractor()
    extractor.feed(html_content)
    return extractor.get_text()


async def get_filing(
    ticker: str,
    form_type: str = "10-K",
    index: int = 0,
    year: int | None = None,
) -> Filing:
    """
    High-level: give a ticker and form type, get back a Filing with full text.
    index=0 is most recent, index=1 is second most recent, etc.
    If year is provided, fetches the most recent filing for that calendar year.
    """
    company = await get_cik_from_ticker(ticker)
    # Fetch enough filings to cover 20+ years when filtering by year
    count = 25 if year is not None else index + 1
    filings = await get_recent_filings(company["cik"], form_type, count=count)

    if not filings:
        raise ValueError(f"No {form_type} filings found for {ticker}")

    if year is not None:
        matching = _match_filings_by_year(filings, form_type, year)
        if not matching:
            raise ValueError(
                f"No {form_type} filing found for {ticker} in {year}. "
                f"Available years: {_available_years(filings)}"
            )
        filing_meta = matching[0]
    elif index >= len(filings):
        raise ValueError(f"Only {len(filings)} {form_type} filings found, requested index {index}")
    else:
        filing_meta = filings[index]
    text = await fetch_filing_text(
        company["cik"],
        filing_meta["accession_number"],
        filing_meta["primary_document"],
    )

    return Filing(
        company_name=company["name"],
        cik=company["cik"],
        ticker=company["ticker"],
        form_type=filing_meta["form"],
        filed_date=filing_meta["filing_date"],
        accession_number=filing_meta["accession_number"],
        primary_doc_url=_build_filing_url(
            company["cik"],
            filing_meta["accession_number"],
            filing_meta["primary_document"],
        ),
        text=text,
    )


