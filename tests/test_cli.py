"""Tests for secdd.cli — file reading."""

import pytest
from secdd.cli import read_file_content


# ── read_file_content ──


class TestReadFileContent:
    def test_reads_txt_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, SEC filing content here.", encoding="utf-8")
        result = read_file_content(str(f))
        assert result == "Hello, SEC filing content here."

    def test_reads_html_file_utf8(self, tmp_path):
        f = tmp_path / "filing.html"
        f.write_text("<html><body>Revenue: $50M</body></html>", encoding="utf-8")
        result = read_file_content(str(f))
        assert "<html>" in result  # read_file_content does NOT strip HTML
        assert "Revenue: $50M" in result

    def test_reads_htm_file_with_encoding_fallback(self, tmp_path):
        f = tmp_path / "filing.htm"
        # Write with cp1252 encoding (common in older EDGAR filings)
        content = "Revenue was \u20ac100M"  # euro sign
        f.write_bytes(content.encode("cp1252"))
        result = read_file_content(str(f))
        assert "100M" in result

    def test_reads_htm_invalid_utf8_falls_to_cp1252_or_latin1(self, tmp_path):
        """Bytes invalid in utf-8 (0x81) are read via cp1252 or latin-1 fallback.
        latin-1 accepts all byte values, so the errors='replace' path is
        unreachable in practice. This test verifies the fallback chain works."""
        f = tmp_path / "filing.htm"
        f.write_bytes(b"Hello\x81World")
        result = read_file_content(str(f))
        assert "Hello" in result
        assert "World" in result

    def test_reads_generic_file_as_utf8(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\n1,2", encoding="utf-8")
        result = read_file_content(str(f))
        assert "col1,col2" in result

    def test_pdf_no_text_fallback(self, tmp_path):
        """A minimal valid PDF with no extractable text."""
        pypdf = pytest.importorskip("pypdf")
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=612, height=792)
        pdf_path = tmp_path / "blank.pdf"
        with open(pdf_path, "wb") as f:
            writer.write(f)
        result = read_file_content(str(pdf_path))
        assert result == "(No text extracted from PDF)"

    def test_pdf_with_text(self, tmp_path):
        """Create a PDF with actual text content and verify extraction."""
        pytest.importorskip("pypdf")
        reportlab_canvas = pytest.importorskip("reportlab.pdfgen.canvas")
        import io

        buf = io.BytesIO()
        c = reportlab_canvas.Canvas(buf)
        c.drawString(72, 700, "Annual Revenue Report FY2024")
        c.save()
        buf.seek(0)
        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_bytes(buf.read())
        result = read_file_content(str(pdf_path))
        assert "Annual Revenue Report" in result
