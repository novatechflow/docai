"""Corpus-scale ingestion: parallel OCR and checkpointed, resumable indexing.

``build_index_from_markdown`` reads every file and embeds every chunk in one
pass, which does not survive a large corpus or an interrupted run. This module
OCRs PDFs concurrently, then adds their chunks to a FAISS index in batches,
saving the index and a ledger periodically so a re-run resumes where it stopped.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from docai_toolkit.ocr.clients import OcrClient

try:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
except ImportError as _exc:  # pragma: no cover - optional dependency
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]
    FAISS = None  # type: ignore[assignment]

LEDGER_NAME = "ingest_ledger.json"


@dataclass
class OcrOutcome:
    source: str
    markdown: Optional[str] = None
    error: Optional[str] = None
    skipped: bool = False


@dataclass
class OcrSummary:
    outcomes: List[OcrOutcome] = field(default_factory=list)

    @property
    def markdown_paths(self) -> List[Path]:
        return [Path(o.markdown) for o in self.outcomes if o.markdown and not o.error]

    @property
    def failures(self) -> List[OcrOutcome]:
        return [o for o in self.outcomes if o.error]


def _write_markdown(md_path: Path, pages) -> None:
    lines: List[str] = []
    for page in pages:
        lines.append(f"# Page {page.page_number}")
        lines.append("")
        lines.append(page.text.strip())
        lines.append("\n")
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _ocr_one(pdf_path: Path, output_dir: Path, client: OcrClient, skip_existing: bool) -> OcrOutcome:
    md_path = output_dir / f"{pdf_path.stem}.md"
    if skip_existing and md_path.exists():
        return OcrOutcome(source=str(pdf_path), markdown=str(md_path), skipped=True)
    try:
        pages = client.recognize(pdf_path)
        _write_markdown(md_path, pages)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
        return OcrOutcome(source=str(pdf_path), error=str(exc))
    return OcrOutcome(source=str(pdf_path), markdown=str(md_path))


def batch_ocr(
    pdf_paths: Iterable[Path],
    output_dir: Path,
    client: OcrClient,
    workers: int = 4,
    skip_existing: bool = True,
) -> OcrSummary:
    """OCR many PDFs to Markdown concurrently; a failure on one is recorded, not raised."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = list(pdf_paths)
    summary = OcrSummary()
    if not paths:
        return summary

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_ocr_one, path, output_dir, client, skip_existing): path for path in paths
        }
        for future in as_completed(futures):
            summary.outcomes.append(future.result())

    summary.outcomes.sort(key=lambda o: o.source)
    return summary


def _load_ledger(persist_path: Path) -> set:
    ledger_file = persist_path / LEDGER_NAME
    if not ledger_file.exists():
        return set()
    try:
        return set(json.loads(ledger_file.read_text(encoding="utf-8")).get("sources", []))
    except (OSError, ValueError):
        return set()


def _save_ledger(persist_path: Path, sources: set) -> None:
    persist_path.mkdir(parents=True, exist_ok=True)
    (persist_path / LEDGER_NAME).write_text(
        json.dumps({"sources": sorted(sources)}, indent=2), encoding="utf-8"
    )


def build_index_incremental(
    markdown_files: Iterable[Path],
    embeddings,
    persist_path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 256,
    checkpoint_every: int = 8,
    allow_dangerous_deserialization: bool = False,
) -> Optional["FAISS"]:
    """Add markdown files to a FAISS index in batches, checkpointing to disk.

    Already-indexed sources (tracked in a ledger beside the index) are skipped,
    so an interrupted run resumes without re-embedding. ``persist_path`` is a
    directory; resuming an existing index unpickles it, so
    ``allow_dangerous_deserialization`` must be set for indexes you trust.
    """
    if RecursiveCharacterTextSplitter is None or FAISS is None:
        raise RuntimeError("langchain is required for ingestion. Install langchain and langchain-community.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    processed = _load_ledger(persist_path)

    db: Optional[FAISS] = None
    index_file = persist_path / "index.faiss"
    if index_file.exists():
        if not allow_dangerous_deserialization:
            raise ValueError(
                "Resuming loads a pickled FAISS index; pass allow_dangerous_deserialization=True "
                "for an index you created yourself."
            )
        db = FAISS.load_local(str(persist_path), embeddings, allow_dangerous_deserialization=True)

    pending_texts: List[str] = []
    pending_meta: List[dict] = []
    files_since_checkpoint = 0

    def flush() -> None:
        nonlocal db, pending_texts, pending_meta
        if not pending_texts:
            return
        docs = splitter.create_documents(pending_texts, metadatas=pending_meta)
        if docs:
            if db is None:
                db = FAISS.from_documents(docs, embeddings)
            else:
                db.add_documents(docs)
        pending_texts = []
        pending_meta = []

    def checkpoint() -> None:
        if db is not None:
            db.save_local(str(persist_path))
        _save_ledger(persist_path, processed)

    for md_path in markdown_files:
        source = str(md_path)
        if source in processed:
            continue
        pending_texts.append(md_path.read_text(encoding="utf-8"))
        pending_meta.append({"source": source})
        processed.add(source)
        files_since_checkpoint += 1

        if len(pending_texts) >= batch_size:
            flush()
        if files_since_checkpoint >= checkpoint_every:
            flush()
            checkpoint()
            files_since_checkpoint = 0

    flush()
    checkpoint()
    return db


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    from docai_toolkit.config import AppConfig
    from docai_toolkit.ocr import RemoteOcrClient, TesseractOcrClient
    from docai_toolkit.rag.index import RemoteEmbeddings, SentenceTransformerEmbeddings

    parser = argparse.ArgumentParser(description="OCR a set of PDFs and build a resumable FAISS index.")
    parser.add_argument("--docs", type=Path, help="File of PDF paths, one per line (e.g. docai-classify --doc-list).")
    parser.add_argument("pdfs", nargs="*", type=Path, help="PDF paths (alternative to --docs).")
    parser.add_argument("--markdown-dir", type=Path, required=True, help="Directory for OCR'd Markdown.")
    parser.add_argument("--index", type=Path, required=True, help="Directory for the FAISS index.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=8)
    parser.add_argument(
        "--allow-dangerous-deserialization",
        action="store_true",
        help="Required to resume an existing index (unpickles it).",
    )
    args = parser.parse_args(argv)

    pdf_paths: List[Path] = list(args.pdfs)
    if args.docs:
        pdf_paths += [Path(line) for line in args.docs.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not pdf_paths:
        parser.error("no PDFs given (use --docs or positional paths)")

    config = AppConfig.from_env()
    if config.ocr.endpoint:
        ocr_client: OcrClient = RemoteOcrClient(
            api_key=config.ocr.api_key or config.llm.api_key, endpoint=config.ocr.endpoint
        )
    else:
        ocr_client = TesseractOcrClient()

    if config.embeddings.endpoint:
        embeddings = RemoteEmbeddings(
            config.embeddings.endpoint,
            api_key=config.embeddings.api_key or config.llm.api_key,
            model=config.embeddings.model,
        )
    else:
        embeddings = SentenceTransformerEmbeddings(config.embeddings.model)

    ocr = batch_ocr(pdf_paths, args.markdown_dir, ocr_client, workers=args.workers)
    print(f"OCR: {len(ocr.markdown_paths)} ok, {len(ocr.failures)} failed")
    for failure in ocr.failures:
        print(f"  FAILED {failure.source}: {failure.error}")

    build_index_incremental(
        ocr.markdown_paths,
        embeddings,
        args.index,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        allow_dangerous_deserialization=args.allow_dangerous_deserialization,
    )
    print(f"index -> {args.index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
