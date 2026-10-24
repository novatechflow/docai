# DocAI Toolkit

Local OCR, Markdown and RAG with optional OpenAI-compatible endpoints for generation and embeddings, and an optional custom OCR endpoint. The PyPI package is `docai-toolkit` and the import name is `docai_toolkit`.

- `pdf_viewer_app.py`: Tkinter UI to open PDFs, run OCR → Markdown, and “chat” via retrieval + generation.
- `docai_toolkit/`: library for OCR (local Tesseract or remote endpoint), embedding/indexing (local or remote), and simple chat over FAISS.
- Status: under active development; APIs and defaults may change as the AI ecosystem moves quickly.

## Requirements

- Python 3.11+ (CI tests 3.11, 3.12, 3.13 and 3.14; the Docker image uses 3.14)
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
- OCR → Markdown: run OCR on a PDF and save Markdown to the configured output directory (local Tesseract, or a custom OCR endpoint that takes the raw PDF).
- Chat: build a quick FAISS index over a chosen Markdown file and query it with the configured LLM model (an OpenAI-compatible endpoint, or a local `transformers` pipeline when no endpoint is set).
- Settings: set HF token, optional custom endpoints (OCR/embeddings/LLM), model choices, and output directory. Settings persist to `~/.docai/config.json`. Env vars (`HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, `DOC_AI_OUTPUT_DIR`) are auto-read.

### Triaging a mixed file tree

Before OCR/indexing a large, untrusted pile of files, separate real documents
from media, archives, and renamed binaries. Classification checks each file two
ways, by extension *and* by detected content type, and only keeps a file when both
agree; anything that disagrees, is empty, unreadable, or unexpectedly large is
quarantined for review rather than fed downstream.

```bash
docai-classify /path/to/tree --manifest manifest.json --doc-list docs.txt
```

- `--manifest`: JSON report with per-file verdicts, reasons, and byte/count totals.
- `--doc-list`: newline-separated paths of the kept documents. It lists every kept type, not only PDFs.

Kept types: `.pdf .doc .docx .rtf .odt .txt .md`. Content detection uses
`python-magic` (`pip install ".[classify]"`) when available, then the system
`file` command, then a built-in signature sniff. The signature sniff knows PDF,
RTF, legacy `.doc` and zip-based `.docx`/`.odt`. When no detector identifies a
`.txt` or `.md` file, it is kept as `text/plain` if it has no NUL bytes in its
first 8 KB. Without `python-magic` or `file`, other mismatches are reported as
`content type undetermined` instead of `extension and content type disagree`.

### Ingesting a corpus (OCR + index at scale)

`build_index_from_markdown` loads and embeds everything in one pass, which does
not survive a large or interrupted run. `docai-ingest` OCRs PDFs concurrently,
then adds their chunks to a FAISS index in batches, checkpointing the index and
a ledger so a re-run resumes where it stopped (already-indexed files are
skipped, not re-embedded).

```bash
docai-ingest --docs docs.txt --markdown-dir out/md --index out/faiss --workers 8
```

- `--docs`: a file of PDF paths, one per line. `docai-ingest` OCRs every entry as a PDF, so other file types fail and are reported as `FAILED`.
- OCR client and embeddings come from `~/.docai/config.json` / env, same as the viewer.
- `--allow-dangerous-deserialization`: required only to **resume** an existing index (it unpickles it). Use it only for indexes you created yourself.

Pipeline end to end: run `docai-classify` on a tree, keep only the PDFs from its
`--doc-list`, pass that file to `docai-ingest`, then query the index with
`chat_over_corpus`.

```bash
docai-classify /path/to/tree --doc-list docs.txt
grep -i '\.pdf$' docs.txt > pdfs.txt
docai-ingest --docs pdfs.txt --markdown-dir out/md --index out/faiss --workers 8
```

### Generation and embeddings: OpenAI-compatible endpoints

Remote generation and embeddings speak the **OpenAI REST API**
(`/v1/chat/completions`, `/v1/embeddings`). Any server that implements it works
through the same client, whether it is a local runner or a hosted gateway. Set the endpoint
to the server's `/v1` base URL and the model to the served model name.

**Local on a laptop (Ollama)**, on CUDA or Apple Metal:

```bash
ollama serve
ollama pull llama3.1
ollama pull nomic-embed-text
```
- LLM Endpoint: `http://localhost:11434/v1`, LLM Model: `llama3.1`
- Embedding Endpoint: `http://localhost:11434/v1`, Embedding Model: `nomic-embed-text`

**Linux with NVIDIA, or large batch runs (vLLM)**:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3   # serves an OpenAI API on :8000
```
- LLM Endpoint: `http://localhost:8000/v1`, LLM Model: `mistralai/Mistral-7B-Instruct-v0.3`

**Local embeddings on Apple Silicon or NVIDIA**: with no embedding endpoint set, `sentence-transformers` runs in-process and `EmbeddingConfig.device` (`auto` by default) picks the device: CUDA if present, else Apple Metal (MPS), else CPU. Set it explicitly (`cpu`, `cuda`, `cuda:0`, `mps`) to override.

**Hosted API**: set the endpoint to the provider's `/v1` base URL and put the
key in the token field. Leaving the endpoint blank uses in-process
`sentence-transformers` for embeddings and a local `transformers` pipeline for
generation, no server required.

Endpoints must be `https`, except `http` is allowed for `localhost`. Ollama
(`:11434`) and vLLM (`:8000`) listen there, so a token is not sent in cleartext
over the network. `~/.docai/config.json` is written `0600`, and a token
picked up from the environment is not persisted to it.

Environment variables:
- `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` / `DOC_AI_HF_TOKEN`: auth token (auto-loads into LLM + embeddings).
- `DOC_AI_OUTPUT_DIR`: default output directory for OCR/Markdown.

### Docker

Build:
```bash
docker build -t docai-toolkit .
```

The default command starts the GUI, which needs X11 forwarding. For headless
tasks, override the command. The image installs `requirements.txt` and copies
the source to `/app`. It does not install the package, so the `docai-*` console
scripts are not on the `PATH`. Use `python -m` instead. This classifies the
current directory, mounted read-only at `/data`, and prints the counts:

```bash
docker run --rm -v "$PWD:/data:ro" docai-toolkit python -m docai_toolkit.classify /data
```

The image does not include `pytest`. Run the tests outside the container, as
shown in [Run the tests](#run-the-tests).

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

## Run the tests

Install the package with the `dev` extra, then run `pytest`. CI runs the same
commands on Python 3.11, 3.12, 3.13 and 3.14 with the `tesseract-ocr` and
`poppler-utils` system packages installed.

```bash
pip install -e ".[dev]"
python -m pytest -q
```

The tests cover classification, config loading, device selection, the HTTP
client, ingestion, the viewer's PDF writer, index building and the remote OCR
and embedding clients. Tests that need `langchain` are skipped when it is not
installed. Some classification tests expect the `file` command or
`python-magic` to be available.

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

## Background

The toolkit came out of consulting work on document pipelines. See
[Beyond Ctrl+F: Use LLMs for PDF Analysis](https://www.novatechflow.com/2024/05/beyond-ctrlf-use-llms-for-pdf-analysis.html).

What it does:

- Load PDFs: the viewer reads PDF text with `pypdf`, and OCR turns PDFs into Markdown.
- OCR: local Tesseract, or a custom OCR endpoint that takes the raw PDF.
- Triage: `docai-classify` sorts a file tree into documents, non-documents and quarantine.
- Chunking: Markdown is split with `RecursiveCharacterTextSplitter` (`chunk_size` and `chunk_overlap` are parameters).
- Embeddings and search: chunks are embedded locally with `sentence-transformers` or through an OpenAI-compatible endpoint, and stored in a FAISS index.
- Chat: retrieval over the index plus generation through an OpenAI-compatible endpoint or a local `transformers` pipeline.

```
PDFs
  ↓  OCR (Tesseract or custom endpoint)
Markdown
  ↓  chunking
Chunks
  ↓  embeddings
FAISS index
  ↓  retrieval and generation
Answers
```

For help with document pipelines or RAG systems, see
[Consulting Services](https://www.novatechflow.com/p/consulting-services.html).

## License

CC BY-NC-SA 4.0 (see `LICENSE`).
