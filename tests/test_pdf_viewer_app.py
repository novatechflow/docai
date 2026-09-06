import pytest

pytest.importorskip("langchain")
pytest.importorskip("tkinter")
pypdf = pytest.importorskip("pypdf")
PdfReader = pypdf.PdfReader

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from pdf_viewer_app import PDFViewerApp

pytestmark = pytest.mark.skipif(not HAS_REPORTLAB, reason="reportlab not installed")


def test_write_pdf_round_trip(tmp_path):
    """Ensure text written with _write_pdf can be read back."""
    content = "Hello world\nSecond line"
    pdf_path = tmp_path / "out.pdf"

    app = PDFViewerApp.__new__(PDFViewerApp)  # bypass Tk setup for a pure logic test
    app._write_pdf(str(pdf_path), content, canvas, letter)

    assert pdf_path.exists()

    reader = PdfReader(str(pdf_path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)

    assert "Hello world" in text
    assert "Second line" in text
