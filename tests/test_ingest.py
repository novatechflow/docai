import pytest

pytest.importorskip("langchain")

from langchain_core.embeddings import Embeddings

from docai_toolkit.ingest import batch_ocr, build_index_incremental, LEDGER_NAME
from docai_toolkit.ocr.clients import OcrClient, PageResult


class FakeOcrClient(OcrClient):
    def __init__(self, fail_on=None):
        self.fail_on = fail_on or set()
        self.seen = []

    def recognize(self, pdf_path):
        self.seen.append(pdf_path.name)
        if pdf_path.name in self.fail_on:
            raise RuntimeError("boom")
        return [PageResult(page_number=1, text=f"text of {pdf_path.stem}")]


class FakeEmbeddings(Embeddings):
    def __init__(self, dim=8):
        self.dim = dim
        self.calls = 0

    def _vec(self, text):
        return [float((len(text) + i) % 7) for i in range(self.dim)]

    def embed_documents(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _make_pdfs(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"doc{i}.pdf"
        p.write_bytes(b"%PDF-1.4 x")
        paths.append(p)
    return paths


def test_batch_ocr_writes_markdown_and_isolates_failures(tmp_path):
    pdfs = _make_pdfs(tmp_path, 3)
    out = tmp_path / "md"
    client = FakeOcrClient(fail_on={"doc1.pdf"})

    summary = batch_ocr(pdfs, out, client, workers=3)

    assert len(summary.markdown_paths) == 2
    assert len(summary.failures) == 1
    assert summary.failures[0].source.endswith("doc1.pdf")
    assert (out / "doc0.md").read_text().strip().endswith("text of doc0")


def test_batch_ocr_skips_existing(tmp_path):
    pdfs = _make_pdfs(tmp_path, 2)
    out = tmp_path / "md"
    first = batch_ocr(pdfs, out, FakeOcrClient(), workers=2)
    assert all(not o.skipped for o in first.outcomes)

    client2 = FakeOcrClient()
    second = batch_ocr(pdfs, out, client2, workers=2)
    assert all(o.skipped for o in second.outcomes)
    assert client2.seen == []  # nothing re-OCR'd


def _make_markdown(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"m{i}.md"
        p.write_text(f"# Page 1\n\nContent number {i} " + ("word " * 50))
        paths.append(p)
    return paths


def test_incremental_build_and_persist(tmp_path):
    mds = _make_markdown(tmp_path, 4)
    index = tmp_path / "faiss"
    emb = FakeEmbeddings()

    db = build_index_incremental(mds, emb, index, chunk_size=100, chunk_overlap=10, batch_size=2, checkpoint_every=1)

    assert db is not None
    assert db.index.ntotal > 0
    assert (index / "index.faiss").exists()
    assert (index / LEDGER_NAME).exists()


def test_incremental_resume_skips_processed(tmp_path):
    mds = _make_markdown(tmp_path, 3)
    index = tmp_path / "faiss"

    emb1 = FakeEmbeddings()
    db1 = build_index_incremental(mds, emb1, index, chunk_size=100, chunk_overlap=10)
    total_after_first = db1.index.ntotal

    emb2 = FakeEmbeddings()
    db2 = build_index_incremental(
        mds, emb2, index, chunk_size=100, chunk_overlap=10, allow_dangerous_deserialization=True
    )
    assert emb2.calls == 0  # everything already in the ledger
    assert db2.index.ntotal == total_after_first


def test_incremental_resume_indexes_only_new_files(tmp_path):
    mds = _make_markdown(tmp_path, 2)
    index = tmp_path / "faiss"
    db1 = build_index_incremental(mds, FakeEmbeddings(), index, chunk_size=100, chunk_overlap=10)
    first_total = db1.index.ntotal

    extra = tmp_path / "new.md"
    extra.write_text("# Page 1\n\nBrand new content " + ("word " * 50))
    emb = FakeEmbeddings()
    db2 = build_index_incremental(
        mds + [extra], emb, index, chunk_size=100, chunk_overlap=10, allow_dangerous_deserialization=True
    )
    assert emb.calls > 0
    assert db2.index.ntotal > first_total


def test_resume_without_optin_raises(tmp_path):
    mds = _make_markdown(tmp_path, 1)
    index = tmp_path / "faiss"
    build_index_incremental(mds, FakeEmbeddings(), index, chunk_size=100, chunk_overlap=10)

    extra = _make_markdown(tmp_path, 1)  # different dir? same names -> use new file
    new = tmp_path / "another.md"
    new.write_text("# Page 1\n\nmore " + ("word " * 50))
    with pytest.raises(ValueError, match="allow_dangerous_deserialization"):
        build_index_incremental([new], FakeEmbeddings(), index, chunk_size=100, chunk_overlap=10)
