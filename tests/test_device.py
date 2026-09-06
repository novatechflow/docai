import pytest

from docai_toolkit.rag import index as index_mod
from docai_toolkit.rag.index import resolve_device


def test_explicit_device_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:0") == "cuda:0"
    assert resolve_device("mps") == "mps"


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch):
    import types

    def fake_torch(cuda, mps):
        mod = types.SimpleNamespace()
        mod.cuda = types.SimpleNamespace(is_available=lambda: cuda)
        mod.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
        return mod

    import builtins

    real_import = builtins.__import__

    def make_import(torch_mod):
        def _import(name, *args, **kwargs):
            if name == "torch":
                return torch_mod
            return real_import(name, *args, **kwargs)
        return _import

    monkeypatch.setattr(builtins, "__import__", make_import(fake_torch(True, False)))
    assert resolve_device("auto") == "cuda"
    monkeypatch.setattr(builtins, "__import__", make_import(fake_torch(False, True)))
    assert resolve_device("auto") == "mps"
    monkeypatch.setattr(builtins, "__import__", make_import(fake_torch(False, False)))
    assert resolve_device("auto") == "cpu"


def test_auto_without_torch_returns_none(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    assert resolve_device("auto") is None


def test_embeddings_forwards_resolved_device(monkeypatch):
    captured = {}

    class FakeST:
        def __init__(self, model_name, device=None):
            captured["model"] = model_name
            captured["device"] = device

        def encode(self, x):  # pragma: no cover - not exercised here
            return []

    monkeypatch.setattr(index_mod, "SentenceTransformer", FakeST)
    index_mod.SentenceTransformerEmbeddings("all-mpnet-base-v2", device="cpu")
    assert captured == {"model": "all-mpnet-base-v2", "device": "cpu"}
