import pytest

pytest.importorskip("langchain")

from docai_toolkit.rag.index import RemoteEmbeddings, load_index


class DummyOpenAIClient:
    def __init__(self, *_, **__):
        self.embed_calls = []

    def embed(self, inputs):
        self.embed_calls.append(list(inputs))
        return [[float(len(text))] for text in inputs]


def test_remote_embeddings_documents(monkeypatch):
    dummy = DummyOpenAIClient()
    monkeypatch.setattr("docai_toolkit.rag.index.OpenAIClient", lambda *a, **k: dummy)
    emb = RemoteEmbeddings(endpoint="https://example.com/v1", model="text-embed")
    assert emb.embed_documents(["a", "bb"]) == [[1.0], [2.0]]
    assert dummy.embed_calls == [["a", "bb"]]


def test_remote_embeddings_query(monkeypatch):
    dummy = DummyOpenAIClient()
    monkeypatch.setattr("docai_toolkit.rag.index.OpenAIClient", lambda *a, **k: dummy)
    emb = RemoteEmbeddings(endpoint="https://example.com/v1", model="text-embed")
    assert emb.embed_query("abc") == [3.0]


def test_remote_embeddings_propagates_transport_errors(monkeypatch):
    class FailingClient:
        def __init__(self, *_, **__):
            pass

        def embed(self, _inputs):
            raise RuntimeError("endpoint unreachable")

    monkeypatch.setattr("docai_toolkit.rag.index.OpenAIClient", FailingClient)
    emb = RemoteEmbeddings(endpoint="https://example.com/v1", model="text-embed")
    with pytest.raises(RuntimeError):
        emb.embed_documents(["a", "b"])


def test_load_index_requires_explicit_opt_in(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    with pytest.raises(ValueError, match="allow_dangerous_deserialization"):
        load_index(index_dir, embeddings=None)
