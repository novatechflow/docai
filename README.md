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

### Hugging Face onboarding (fast path)

1. Create a Hugging Face access token: https://huggingface.co/settings/tokens (choose “Read” or “Write” as needed).
2. Export it so the app can auto-load it:
   ```bash
   export HF_TOKEN=your_token_here
   # or HUGGINGFACEHUB_API_TOKEN=your_token_here
   ```
3. Pick models (examples):
   - OCR: point the OCR endpoint at a hosted OCR model (HF Inference API URL).
   - Embeddings: e.g., `sentence-transformers/all-mpnet-base-v2` via Inference Endpoints (text-embeddings task) or local.
   - LLM: e.g., `mistralai/Mistral-7B-Instruct-v0.1` via Inference Endpoints or local HF pipeline.
4. Start the app, open Settings, and paste endpoints/models if you didn’t set env vars. Output dir can be set there as well.

Endpoints must be `https` (plain `http` is accepted only for `localhost`), so the access token is never
sent in cleartext. The config file `~/.docai/config.json` is written with `0600` permissions, and a token
picked up from the environment is not written to it.

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

- OCR: pluggable clients (`RemoteOcrClient` for HF/custom endpoints, `TesseractOcrClient` local fallback) that turn PDFs into Markdown (`ocr/pipeline.py`).
- RAG: build a FAISS index from Markdown (`rag/index.py`), then chat using a chosen HF model (`rag/chat.py`).
- Config: lightweight dataclasses in `docai_toolkit/config.py` for selecting providers/models; saved at `~/.docai/config.json`.
- Remote-friendly: use HF token + model ids by default; configs allow custom OCR/embedding/generation endpoints. FAISS runs locally for fast retrieval.

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

# Build index + chat (requires sentence_transformers + transformers)
python - <<'PY'
from pathlib import Path
from docai_toolkit.rag import (
    build_index_from_markdown,
    chat_over_corpus,
    load_index,
    SentenceTransformerEmbeddings,
)
index_path = Path("outputs/faiss_index")
db = build_index_from_markdown([Path("outputs/your.md")], persist_path=index_path)
print(chat_over_corpus(db, "What is this document about?", model_id="mistralai/Mistral-7B-Instruct-v0.1"))
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
