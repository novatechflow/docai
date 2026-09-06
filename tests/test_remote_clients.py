import pytest

pytest.importorskip("langchain")

from docai_toolkit.ocr.clients import RemoteOcrClient


class DummyHttpClient:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def post(self, *_, **__):
        self.calls += 1
        return self.response


def test_remote_ocr_parses_common_shapes(tmp_path):
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 dummy")

    scenarios = [
        "plain text",
        {"text": "text in dict"},
        {"pages": [{"text": "page text"}]},
        ["page 1", "page 2"],
    ]

    for resp in scenarios:
        client = RemoteOcrClient(api_key=None, endpoint="https://example.com")
        client.client = DummyHttpClient(resp)  # type: ignore[attr-defined]
        pages = client.recognize(pdf_path)
        assert pages, f"Empty pages for response {resp}"
        assert isinstance(pages[0].text, str)
