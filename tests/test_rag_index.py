import pytest

pytest.importorskip("langchain")

from docai_toolkit.rag.index import RemoteEmbeddings, load_index


class DummyClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def post_json(self, *_, **__):
        self.calls += 1
        return self.responses.pop(0)


def _embeddings(monkeypatch, responses):
    dummy = DummyClient(responses)
    monkeypatch.setattr("docai_toolkit.rag.index.HuggingFaceClient", lambda *a, **k: dummy)
    return RemoteEmbeddings(endpoint="https://example.com"), dummy


def test_embed_documents_uses_batch_response(monkeypatch):
    emb, dummy = _embeddings(monkeypatch, [[[0.1], [0.2]]])
    assert emb.embed_documents(["a", "b"]) == [[0.1], [0.2]]
    assert dummy.calls == 1


def test_embed_documents_falls_back_when_batch_length_mismatches(monkeypatch):
    emb, dummy = _embeddings(monkeypatch, [[[0.1]], [[0.1]], [[0.2]]])
    assert emb.embed_documents(["a", "b"]) == [[0.1], [0.2]]
    assert dummy.calls == 3


def test_embed_documents_propagates_transport_errors(monkeypatch):
    class FailingClient:
        def post_json(self, *_, **__):
            raise RuntimeError("endpoint unreachable")

    monkeypatch.setattr("docai_toolkit.rag.index.HuggingFaceClient", lambda *a, **k: FailingClient())
    emb = RemoteEmbeddings(endpoint="https://example.com")
    with pytest.raises(RuntimeError):
        emb.embed_documents(["a", "b"])


def test_load_index_requires_explicit_opt_in(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    with pytest.raises(ValueError, match="allow_dangerous_deserialization"):
        load_index(index_dir, embeddings=None)
