# DocAI Toolkit

Local OCR + Markdown + RAG with optional Hugging Face/custom endpoints. Renamed to avoid PyPI name collisions (`docai-toolkit` package import is `docai_toolkit`).

- `pdf_viewer_app.py`: Tkinter UI to open PDFs, run OCR → Markdown, and “chat” via retrieval + generation.
- `docai_toolkit/`: library for OCR (local Tesseract or remote endpoint), embedding/indexing (local or remote), and simple chat over FAISS.
- Status: under active development; APIs and defaults may change as the AI ecosystem moves quickly.

## Requirements

- Python 3.11+ (tested on 3.11, 3.12 and 3.13)
- Dependencies are split into extras so the viewer does not drag in the ML stack:
  - core (always installed): `pypdf`, `reportlab`
  - `ocr`: `pytesseract`, `pdf2image` (plus the `tesseract-ocr` and `poppler` system packages)
  - `rag`: `langchain`, `langchain-community`, `sentence-transformers`, `faiss-cpu`, `huggingface-hub`
  - `llm`: `transformers`, `accelerate`, `bitsandbytes` (only for local generation)

```bash
# viewer only
pip install .

# everything
pip install ".[all]"

# development (extras + pytest)
pip install -e ".[dev]"

# exact pinned set used by the Docker image
pip install -r requirements.txt
```

## Usage

### GUI Viewer

```bash
python pdf_viewer_app.py
```

- Open: loads all pages of a PDF into the text area.
- Save As: renders the text area content into a new PDF (requires `reportlab`).
- OCR → Markdown: run OCR on a PDF and save Markdown to the configured output directory (local Tesseract or remote OCR endpoint via HF/custom).
- Chat: build a quick FAISS index over a chosen Markdown file and query it with a selected HF model (remote endpoint or local HF pipeline).
- Settings: set HF token, optional custom endpoints (OCR/embeddings/LLM), model choices, and output directory. Settings persist to `~/.docai/config.json`. Env vars (`HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, `DOC_AI_OUTPUT_DIR`) are auto-read.

### Triaging a mixed file tree

Before OCR/indexing a large, untrusted pile of files, separate real documents
from media, archives, and renamed binaries. Classification checks each file two
ways — extension *and* detected content type — and only keeps a file when both
agree; anything that disagrees, is empty, unreadable, or unexpectedly large is
quarantined for review rather than fed downstream.

```bash
docai-classify /path/to/tree --manifest manifest.json --doc-list docs.txt
```

- `--manifest`: JSON report with per-file verdicts, reasons, and byte/count totals.
- `--doc-list`: newline-separated paths of the kept documents, ready to pipe into OCR.

Kept types: `.pdf .doc .docx .rtf .odt .txt .md`. Content detection uses
`python-magic` (`pip install ".[classify]"`) when available, then the system
`file` command, then a built-in signature sniff — so it runs with no extra deps
but is more precise with libmagic installed.

### Ingesting a corpus (OCR + index at scale)

`build_index_from_markdown` loads and embeds everything in one pass, which does
not survive a large or interrupted run. `docai-ingest` OCRs PDFs concurrently,
then adds their chunks to a FAISS index in batches, checkpointing the index and
a ledger so a re-run resumes where it stopped (already-indexed files are
skipped, not re-embedded).

```bash
docai-ingest --docs docs.txt --markdown-dir out/md --index out/faiss --workers 8
```

- `--docs`: a file of PDF paths, one per line — e.g. the `--doc-list` from `docai-classify`.
- OCR client and embeddings come from `~/.docai/config.json` / env, same as the viewer.
- `--allow-dangerous-deserialization`: required only to **resume** an existing index (it unpickles it); use it for indexes you created yourself.

Pipeline end to end: `docai-classify` a tree → feed its `--doc-list` to
`docai-ingest` → query the index with `chat_over_corpus`.

### Generation and embeddings: OpenAI-compatible endpoints

Remote generation and embeddings speak the **OpenAI REST API**
(`/v1/chat/completions`, `/v1/embeddings`). Any server that implements it works
through the same client — a local runner or a hosted gateway. Set the endpoint
to the server's `/v1` base URL and the model to the served model name.

**Local on a laptop (Ollama)** — cross-platform, CUDA and Apple-Metal:

```bash
ollama serve
ollama pull llama3.1
ollama pull nomic-embed-text
```
- LLM Endpoint: `http://localhost:11434/v1`, LLM Model: `llama3.1`
- Embedding Endpoint: `http://localhost:11434/v1`, Embedding Model: `nomic-embed-text`

**Linux + NVIDIA, or corpus-scale batch (vLLM)** — high throughput:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3   # serves an OpenAI API on :8000
```
- LLM Endpoint: `http://localhost:8000/v1`, LLM Model: `mistralai/Mistral-7B-Instruct-v0.3`

**Local embeddings on Apple Silicon / NVIDIA** — with no embedding endpoint set, `sentence-transformers` runs in-process and `EmbeddingConfig.device` (`auto` by default) picks the device: CUDA if present, else Apple Metal (MPS), else CPU. Set it explicitly (`cpu`, `cuda`, `cuda:0`, `mps`) to override.

**Hosted API** — set the endpoint to the provider's `/v1` base URL and put the
key in the token field. Leaving the endpoint blank uses in-process
`sentence-transformers` for embeddings and a local `transformers` pipeline for
generation, no server required.

Endpoints must be `https`, except `http` is allowed for `localhost` — which is
exactly where Ollama (`:11434`) and vLLM (`:8000`) listen, so the local token is
never sent in cleartext. `~/.docai/config.json` is written `0600`, and a token
picked up from the environment is not persisted to it.

Environment variables:
- `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` / `DOC_AI_HF_TOKEN`: auth token (auto-loads into LLM + embeddings).
- `DOC_AI_OUTPUT_DIR`: default output directory for OCR/Markdown.

### Docker

Build:
```bash
docker build -t docai-toolkit .
```

Run (GUI requires X/Wayland forwarding; for headless tasks, override CMD):
```bash
docker run --rm -v $PWD:/data docai-toolkit python -m pytest -q
# or override to run OCR in batch using the library CLI you add
```

macOS GUI via XQuartz:
1) Install/start XQuartz (`brew install --cask xquartz`; enable “Allow connections from network clients” in prefs and restart).
2) Allow local clients: `xhost +localhost`
3) Run:
```bash
docker run --rm -it \
  -e DISPLAY=host.docker.internal:0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  docai-toolkit
```
For day-to-day use, running natively is simpler; use the container when you need an isolated, reproducible environment.

## Tests

Basic round-trip test for the viewer’s PDF writer:

```bash
pytest
```

`reportlab` must be installed for the test to run.

## OCR + RAG (docai_toolkit/)

- OCR: pluggable clients (`RemoteOcrClient` for a custom OCR endpoint, `TesseractOcrClient` local fallback) that turn PDFs into Markdown (`ocr/pipeline.py`).
- RAG: build a FAISS index from Markdown (`rag/index.py`), then chat over it (`rag/chat.py`).
- Transport: generation and embeddings use the OpenAI REST API (`docai_toolkit/http_client.py`), so Ollama, vLLM, llama.cpp, TGI, and hosted APIs are interchangeable.
- Config: lightweight dataclasses in `docai_toolkit/config.py`; saved at `~/.docai/config.json`.
- Local retrieval: FAISS runs in-process; embeddings default to local `sentence-transformers` when no endpoint is set.

To experiment locally:

```bash
# OCR to Markdown (Tesseract fallback requires pytesseract + pdf2image installed)
python - <<'PY'
from pathlib import Path
from docai_toolkit.ocr import TesseractOcrClient, run_ocr_to_markdown
client = TesseractOcrClient()
md_path = run_ocr_to_markdown(Path("your.pdf"), Path("outputs"), client)
print("Saved:", md_path)
PY

# Build index + chat
# Local default needs sentence-transformers (+ transformers for local generation);
# or point at an OpenAI-compatible server (Ollama/vLLM/hosted) via the endpoint args.
python - <<'PY'
from pathlib import Path
from docai_toolkit.rag import (
    build_index_from_markdown,
    chat_over_corpus,
    load_index,
    SentenceTransformerEmbeddings,
)
index_path = Path("outputs/faiss_index")

# Served embeddings + generation (Ollama shown; drop the endpoint args for local):
db = build_index_from_markdown(
    [Path("outputs/your.md")],
    embedding_model="nomic-embed-text",
    embedding_endpoint="http://localhost:11434/v1",
    persist_path=index_path,
)
print(chat_over_corpus(
    db,
    "What is this document about?",
    model_id="llama3.1",
    endpoint="http://localhost:11434/v1",
))
# Later, for an index you created yourself (loading unpickles the docstore):
# db = load_index(index_path, SentenceTransformerEmbeddings("all-mpnet-base-v2"),
#                 allow_dangerous_deserialization=True)
PY
```

## Case Study

This toolkit was architected as part of a consulting engagement to solve common document-centric workflow challenges: extraction, segmentation, text cleaning, OCR, and embedding preparation for AI models.

**Full case study:** [Document Processing Pipeline Using docAI Toolkit](https://www.novatechflow.com/p/document-processing-pipeline-using.html)

### The Problem

Teams building RAG pipelines or document analytics typically stitch together custom scripts, ad-hoc processing, or partial library support—resulting in inconsistent, brittle pipelines.

### The Solution

docAI provides a clean, modular toolkit for building AI-ready document processing flows:

- **Document loading** — PDF, DOCX, Markdown
- **Page and text splitting** — Configurable chunking
- **Preprocessing** — Cleaning, normalization
- **OCR** — Local (Tesseract) or remote (Hugging Face/custom endpoints)
- **Embedding preparation** — Ready for FAISS, vector DBs, or ML pipelines

### Example Workflow
```
Source documents (PDF, DOCX, MD)
        ↓
   docAI splits & preprocesses
        ↓
   Cleaned chunks → embedding model
        ↓
   Indexed for search / RAG / analytics
```

### Benefits

- Reduces bespoke glue code
- Predictable interface for common document tasks
- Easy integration into production systems

---

**Need help building document pipelines or RAG systems?**

→ [Consulting Services](https://www.novatechflow.com/p/consulting-services.html)  
→ [Book a call](https://cal.com/alexanderalten)


## License

CC BY-NC-SA 4.0 (see `LICENSE`).
