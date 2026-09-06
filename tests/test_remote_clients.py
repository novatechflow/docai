import pytest

pytest.importorskip("langchain")

from docai_toolkit.ocr.clients import RemoteOcrClient
from docai_toolkit.rag.index import RemoteEmbeddings
from docai_toolkit import rag


class DummyClient:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def post_json(self, *_, **__):
        self.calls += 1
        return self.response


def test_remote_ocr_parses_common_shapes(tmp_path, monkeypatch):
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
        client.client = DummyClient(resp)  # type: ignore[attr-defined]
        pages = client.recognize(pdf_path)
        assert pages, f"Empty pages for response {resp}"
        assert isinstance(pages[0].text, str)


def test_remote_embeddings_handles_list_response(monkeypatch):
    resp = [[0.1, 0.2, 0.3]]

    class DummyEmbClient(DummyClient):
        pass

    dummy = DummyEmbClient(resp)
    # Patch the HF client at source to avoid network calls.
    monkeypatch.setattr("docai_toolkit.rag.index.HuggingFaceClient", lambda *a, **k: dummy)

    emb = RemoteEmbeddings(endpoint="https://example.com")
    vec = emb.embed_query("hi")
    assert vec == resp[0]
    assert dummy.calls == 1
