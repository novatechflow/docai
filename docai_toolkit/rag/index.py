from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

try:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:  # langchain < 0.2 kept the splitters in the core package
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    try:
        from langchain_core.embeddings import Embeddings
    except ImportError:
        from langchain.embeddings.base import Embeddings
    from langchain_community.vectorstores import FAISS
    from docai_toolkit.http_client import OpenAIClient
except ImportError as _langchain_exc:  # pragma: no cover - optional dependency
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]
    FAISS = None  # type: ignore[assignment]
    Embeddings = object  # type: ignore[assignment]
    OpenAIClient = None  # type: ignore[assignment]
    _LANGCHAIN_IMPORT_ERROR = _langchain_exc
else:
    _LANGCHAIN_IMPORT_ERROR = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[misc,assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class SentenceTransformerEmbeddings(Embeddings):
    """Adapt a local SentenceTransformer to the langchain Embeddings interface."""

    def __init__(self, model_name: str):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed") from _IMPORT_ERROR
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(value) for value in vector] for vector in self.model.encode(texts)]

    def embed_query(self, text: str) -> List[float]:
        return [float(value) for value in self.model.encode(text)]


class RemoteEmbeddings(Embeddings):
    """Embeddings from an OpenAI-compatible server (`/v1/embeddings`)."""

    def __init__(self, endpoint: str, api_key: Optional[str] = None, model: Optional[str] = None):
        self.client = OpenAIClient(endpoint, api_key=api_key, model=model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self.client.embed([text])[0]


def build_index_from_markdown(
    markdown_files: Iterable[Path],
    embedding_model: str = "all-mpnet-base-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    persist_path: Optional[Path] = None,
    embedding_endpoint: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
):
    if RecursiveCharacterTextSplitter is None or FAISS is None:
        raise RuntimeError("langchain is required for RAG. Install langchain and langchain-community.")

    texts: List[str] = []
    metadatas: List[dict] = []

    for path in markdown_files:
        texts.append(path.read_text(encoding="utf-8"))
        metadatas.append({"source": str(path)})

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents(texts, metadatas=metadatas)

    if embedding_endpoint:
        embeddings = RemoteEmbeddings(embedding_endpoint, api_key=embedding_api_key, model=embedding_model)
    else:
        embeddings = SentenceTransformerEmbeddings(embedding_model)

    db = FAISS.from_documents(docs, embeddings)
    if persist_path:
        persist_path.parent.mkdir(parents=True, exist_ok=True)
        db.save_local(str(persist_path))
    return db


def load_index(persist_path: Path, embeddings, allow_dangerous_deserialization: bool = False):
    if FAISS is None:
        raise RuntimeError("langchain-community is required to load indexes.")
    if not persist_path.exists():
        raise FileNotFoundError(f"Persisted index not found at {persist_path}")
    if not allow_dangerous_deserialization:
        raise ValueError(
            "Loading a FAISS index unpickles its docstore, which executes arbitrary code if the "
            "files were tampered with. Pass allow_dangerous_deserialization=True only for indexes "
            "you created yourself."
        )
    return FAISS.load_local(str(persist_path), embeddings, allow_dangerous_deserialization=True)
