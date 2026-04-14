"""Tests for secdd.edgar — pure functions and data logic only."""

import pytest
from secdd.edgar import (
    normalize_cik,
    _strip_html,
    _extract_filings_from_section,
    _match_filings_by_year,
    _build_filing_url,
    _validate_edgar_url,
    _available_years,
    Filing,
)


# ── normalize_cik ──


class TestNormalizeCik:
    def test_short_cik_pads_to_10(self):
        assert normalize_cik("320193") == "0000320193"

    def test_already_10_digits(self):
        assert normalize_cik("0000320193") == "0000320193"

    def test_single_digit(self):
        assert normalize_cik("5") == "0000000005"


# ── _strip_html ──


class TestStripHtml:
    def test_plain_text_unchanged(self):
        assert _strip_html("Hello world") == "Hello world"

    def test_removes_simple_tags(self):
        result = _strip_html("<p>Revenue was <b>$10B</b></p>")
        assert "$10B" in result
        assert "<p>" not in result
        assert "<b>" not in result

    def test_removes_script_blocks(self):
        html = "Before<script>var x = 1;</script>After"
        result = _strip_html(html)
        assert "var x" not in result
        assert "Before" in result
        assert "After" in result

    def test_removes_style_blocks(self):
        html = "Before<style>.foo { color: red; }</style>After"
        result = _strip_html(html)
        assert "color: red" not in result

    def test_removes_ix_header(self):
        html = "Start<ix:header>lots of XBRL taxonomy data here</ix:header>End"
        result = _strip_html(html)
        assert "XBRL" not in result
        assert "Start" in result
        assert "End" in result

    def test_removes_display_none_divs(self):
        html = '<div style="display: none">hidden XBRL</div>Visible text'
        result = _strip_html(html)
        assert "hidden XBRL" not in result
        assert "Visible text" in result

    def test_decodes_html_entities(self):
        html = "AT&amp;T &lt;revenue&gt; was &quot;high&quot; &#39;indeed&#39;"
        result = _strip_html(html)
        assert "AT&T" in result
        assert "<revenue>" in result
        assert '"high"' in result
        assert "'indeed'" in result

    def test_decodes_nbsp(self):
        result = _strip_html("foo&nbsp;bar")
        # html.unescape converts &nbsp; to \xa0 (non-breaking space), not ASCII space
        assert "foo\xa0bar" in result
        assert "&nbsp;" not in result

    def test_numeric_entities_replaced_with_space(self):
        result = _strip_html("item&#160;one&#8226;two")
        assert "&#160;" not in result
        assert "&#8226;" not in result

    def test_real_world_xbrl_snippet(self):
        """A stripped-down version of what modern EDGAR filings look like."""
        html = """
        <ix:header><ix:hidden><ix:nonfraction>12345</ix:nonfraction></ix:hidden></ix:header>
        <div style="display:none"><ix:nonfraction>67890</ix:nonfraction></div>
        <p>Total revenue was <ix:nonfraction>$50,000</ix:nonfraction> million.</p>
        <script type="text/javascript">console.log("debug");</script>
        """
        result = _strip_html(html)
        assert "12345" not in result  # ix:header content removed
        assert "67890" not in result  # display:none content removed
        assert "console.log" not in result  # script removed
        assert "$50,000" in result  # actual content preserved
        assert "million" in result

    def test_rejects_oversized_html(self):
        """20MB limit prevents DoS from enormous filings."""
        with pytest.raises(ValueError, match="20 MB"):
            _strip_html("x" * 20_000_001)

    def test_nested_elements_inside_skip_tag(self):
        """Tags nested inside a skip tag (script/style) should still be suppressed."""
        html = '<script><div>leaked code</div><p>more code</p></script>Visible'
        result = _strip_html(html)
        assert "leaked code" not in result
        assert "more code" not in result
        assert "Visible" in result

    def test_display_none_with_extra_styles(self):
        """display:none buried among other CSS properties."""
        html = '<div style="color: red; display: none; font-size: 12px">hidden</div>shown'
        result = _strip_html(html)
        assert "hidden" not in result
        assert "shown" in result


# ── _validate_edgar_url ──


class TestValidateEdgarUrl:
    def test_allows_www_sec_gov(self):
        _validate_edgar_url("https://www.sec.gov/Archives/edgar/data/0001/doc.htm")

    def test_allows_data_sec_gov(self):
        _validate_edgar_url("https://data.sec.gov/submissions/CIK0001.json")

    def test_allows_efts_sec_gov(self):
        _validate_edgar_url("https://efts.sec.gov/LATEST/search-index")

    def test_rejects_non_edgar_host(self):
        with pytest.raises(ValueError, match="not an allowed EDGAR host"):
            _validate_edgar_url("https://evil.com/Archives/edgar/data/0001/doc.htm")

    def test_rejects_similar_hostname(self):
        with pytest.raises(ValueError, match="not an allowed EDGAR host"):
            _validate_edgar_url("https://www.sec.gov.evil.com/doc.htm")


# ── _build_filing_url ──


class TestBuildFilingUrl:
    def test_constructs_correct_url(self):
        url = _build_filing_url("320193", "0000320193-25-000001", "filing.htm")
        assert url == (
            "https://www.sec.gov/Archives/edgar/data/0000320193/"
            "000032019325000001/filing.htm"
        )

    def test_pads_short_cik(self):
        url = _build_filing_url("5", "0000000005-25-000001", "doc.htm")
        assert "/0000000005/" in url

    def test_strips_dashes_from_accession(self):
        url = _build_filing_url("320193", "0000320193-25-000001", "doc.htm")
        assert "0000320193-25-000001" not in url
        assert "000032019325000001" in url


# ── _match_filings_by_year ──


class TestMatchFilingsByYear:
    """_match_filings_by_year only selects the year-matching strategy based on
    form_type. It does NOT filter by form_type since callers pre-filter the
    list via get_recent_filings. Fixtures reflect this real-world usage."""

    def test_10k_matches_year_plus_one(self):
        """FY2024 10-K is filed in early 2025, so year+1 is tried first."""
        filings = [
            {"filing_date": "2026-02-15"},  # FY2025
            {"filing_date": "2025-02-15"},  # FY2024
            {"filing_date": "2024-02-15"},  # FY2023
        ]
        result = _match_filings_by_year(filings, "10-K", 2024)
        assert len(result) == 1
        assert result[0]["filing_date"] == "2025-02-15"

    def test_10k_falls_back_to_same_year(self):
        """If no year+1 match, fall back to filings in the same year."""
        filings = [{"filing_date": "2024-10-30"}]
        result = _match_filings_by_year(filings, "10-K", 2024)
        assert len(result) == 1

    def test_10q_matches_same_year(self):
        """10-Q filings are filed in the same calendar year."""
        filings = [
            {"filing_date": "2025-11-01"},
            {"filing_date": "2025-08-01"},
            {"filing_date": "2025-05-01"},
            {"filing_date": "2024-11-01"},
        ]
        result = _match_filings_by_year(filings, "10-Q", 2025)
        assert len(result) == 3

    def test_no_match_returns_empty(self):
        filings = [{"filing_date": "2025-02-15"}, {"filing_date": "2024-02-15"}]
        result = _match_filings_by_year(filings, "10-K", 2020)
        assert result == []

    def test_case_insensitive_form_type(self):
        """form_type.upper() is used for the 10-K check, so '10-k' works."""
        filings = [{"filing_date": "2025-02-15"}, {"filing_date": "2024-02-15"}]
        result = _match_filings_by_year(filings, "10-k", 2024)
        assert len(result) == 1
        assert result[0]["filing_date"] == "2025-02-15"


# ── _extract_filings_from_section ──


class TestExtractFilingsFromSection:
    @pytest.fixture
    def sample_section(self):
        return {
            "form": ["10-K", "10-Q", "8-K", "10-K", "10-Q"],
            "accessionNumber": ["acc-001", "acc-002", "acc-003", "acc-004", "acc-005"],
            "filingDate": ["2025-02-15", "2024-11-01", "2024-09-01", "2024-02-15", "2023-11-01"],
            "primaryDocument": ["doc1.htm", "doc2.htm", "doc3.htm", "doc4.htm", "doc5.htm"],
        }

    def test_extracts_matching_form_type(self, sample_section):
        results = _extract_filings_from_section(sample_section, "10-K", count=10)
        assert len(results) == 2
        assert all(r["form"] == "10-K" for r in results)

    def test_respects_count_limit(self, sample_section):
        results = _extract_filings_from_section(sample_section, "10-K", count=1)
        assert len(results) == 1
        assert results[0]["accession_number"] == "acc-001"

    def test_lowercase_form_type_upper_does_not_match(self, sample_section):
        # form_type_upper param is NOT uppercased by the function, so "10-k" won't
        # match data values "10-K" (even though form.upper() == "10-K", "10-K" != "10-k")
        results = _extract_filings_from_section(sample_section, "10-k", count=10)
        assert len(results) == 0

    def test_no_matching_forms(self, sample_section):
        results = _extract_filings_from_section(sample_section, "DEF 14A", count=10)
        assert results == []

    def test_empty_section(self):
        results = _extract_filings_from_section({}, "10-K", count=5)
        assert results == []

    def test_preserves_filing_metadata(self, sample_section):
        results = _extract_filings_from_section(sample_section, "8-K", count=1)
        assert len(results) == 1
        r = results[0]
        assert r["accession_number"] == "acc-003"
        assert r["filing_date"] == "2024-09-01"
        assert r["primary_document"] == "doc3.htm"


# ── _available_years ──


class TestAvailableYears:
    def test_extracts_years_sorted_descending(self):
        filings = [
            {"filing_date": "2023-03-01"},
            {"filing_date": "2025-02-15"},
            {"filing_date": "2024-11-01"},
        ]
        result = _available_years(filings)
        assert result == "2025, 2024, 2023"

    def test_deduplicates_years(self):
        filings = [
            {"filing_date": "2024-03-01"},
            {"filing_date": "2024-11-01"},
            {"filing_date": "2024-06-15"},
        ]
        result = _available_years(filings)
        assert result == "2024"

    def test_truncates_after_10_with_ellipsis(self):
        filings = [{"filing_date": f"{y}-01-01"} for y in range(2010, 2026)]
        result = _available_years(filings)
        years = result.replace("...", "").split(", ")
        assert len(years) == 10
        assert result.endswith("...")

    def test_no_ellipsis_under_10_years(self):
        filings = [{"filing_date": f"{y}-01-01"} for y in range(2020, 2025)]
        result = _available_years(filings)
        assert "..." not in result


# ── Filing dataclass ──


class TestFilingDataclass:
    def test_filing_construction(self):
        filing = Filing(
            company_name="Apple Inc",
            cik="0000320193",
            ticker="AAPL",
            form_type="10-K",
            filed_date="2025-02-15",
            accession_number="0000320193-25-000001",
            primary_doc_url="https://www.sec.gov/Archives/edgar/data/0000320193/doc.htm",
            text="Full filing text here",
        )
        assert filing.ticker == "AAPL"
        assert filing.text == "Full filing text here"
