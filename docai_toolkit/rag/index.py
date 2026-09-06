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
    from docai_toolkit.hf_client import HuggingFaceClient
except ImportError as _langchain_exc:  # pragma: no cover - optional dependency
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]
    FAISS = None  # type: ignore[assignment]
    Embeddings = object  # type: ignore[assignment]
    HuggingFaceClient = None  # type: ignore[assignment]
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
    """Call a remote embedding endpoint that accepts JSON {\"inputs\": text}."""

    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        self.client = HuggingFaceClient(api_key, default_endpoint=endpoint)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self._extract_batch(self.client.post_json({"inputs": texts}), len(texts))
        except ValueError:
            return [self._extract_vector(self.client.post_json({"inputs": text})) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._extract_vector(self.client.post_json({"inputs": text}))

    @staticmethod
    def _extract_vector(response):
        if isinstance(response, list) and response and isinstance(response[0], list):
            return response[0]
        if isinstance(response, list):
            return response
        raise ValueError("Unexpected embedding response format.")

    @staticmethod
    def _extract_batch(response, expected: int) -> List[List[float]]:
        if not isinstance(response, list) or len(response) != expected:
            raise ValueError("Unexpected batch embedding response format.")
        if response and isinstance(response[0], list):
            return response
        if response and isinstance(response[0], dict) and "embedding" in response[0]:
            return [item["embedding"] for item in response]
        raise ValueError("Unexpected batch embedding response format.")


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
        embeddings = RemoteEmbeddings(embedding_endpoint, api_key=embedding_api_key)
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
