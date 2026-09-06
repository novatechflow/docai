# Changelog

## Unreleased
### Added
- `docai_toolkit.classify` and the `docai-classify` CLI: recursively triage a file tree into documents / non-documents / quarantine, verifying each file by both extension and detected content type so mislabelled or renamed binaries do not slip through. Emits a JSON manifest and a kept-document list. Content detection prefers `python-magic` (new `classify` extra), then the `file` command, then a built-in signature sniff.

### Fixed
- Chat with Docs is wired up again: the question is asked before the worker starts, and the worker now retrieves, generates, and shows the answer instead of discarding the index.
- RAG imports work with langchain >= 0.2 (`langchain_text_splitters` / `langchain_core.embeddings`); indexing previously failed with "langchain is required" even when langchain was installed.
- Local embeddings go through a `SentenceTransformerEmbeddings` adapter, so the default (endpoint-less) index path no longer hands a raw `SentenceTransformer` to FAISS.
- `load_index` takes the embeddings it needs and an explicit `allow_dangerous_deserialization` opt-in; it previously raised unconditionally.
- Worker threads report status through `root.after` instead of touching Tk directly.
- Default OCR provider is `tesseract`; the unimplemented `deepseek` default made OCR fail out of the box.
- Malformed or outdated `~/.docai/config.json` falls back to defaults instead of crashing at startup.
- Remote embedding batches are length-checked, and the per-text fallback no longer swallows transport errors.
- `pyproject.toml` version matches the changelog.

### Security
- Endpoints are restricted to `https` (and `http` on localhost); arbitrary schemes such as `file://` previously turned an endpoint call into a local file read, and `http://` sent the bearer token in cleartext.
- `~/.docai/config.json` is written `0600` inside a `0700` directory, and a token sourced from the environment is no longer persisted to it. The Settings token field is masked.

### Removed
- `DeepSeekOcrClient` placeholder and the unused `OcrConfig.model` / OCR Model setting.

### Dependencies
- Split dependencies into `ocr`, `rag`, `llm`, `all` and `dev` extras so the viewer no longer pulls in torch, bitsandbytes and faiss; `pytest` moved out of the runtime dependency set.
- `requirements.txt` is now an exact-pinned set used by the Docker image, and version ranges in `pyproject.toml` are bounded.
- Replaced the retired `PyPDF2` with its maintained successor `pypdf`.
- Raised the floor to Python 3.11 and added a `pytest` matrix over 3.11/3.12/3.13; the Docker base image moves to `python:3.13-slim`, pinned by digest.
- Added a `[build-system]` table and PEP 639 license metadata.
- The `docai-viewer` console script works from an installed wheel: `pdf_viewer_app` was never packaged, so the entry point could only fail with `ModuleNotFoundError`.

### CI
- Added Dependabot for pip, GitHub Actions and Docker, with auto-merge for non-major updates and a comment on major ones.
- Pinned every third-party action to a commit SHA.
- Release notes now contain only the section for the tag being released instead of the entire changelog.

## 0.1.3.1
### Changed
- pypi push fixed
- project URL fixed

## 0.1.3
### Added
- Docker workflow now pushes images only on tags and publishes to GHCR (optional Docker Hub) with tag-based naming.
- PyPI workflow now only runs on tags, syncs version from tag, and skips existing uploads.
- GitHub Release automation on tagged pushes (uses changelog as body).

### Changed
- Fixed YAML indentation in workflows to satisfy GitHub Actions parser.

## 0.1.2
### Added
- GHCR publishing in Docker workflow and tag-only image pushes; PyPI workflow syncs version from tag and skips existing uploads.
- Docker image runs as non-root, includes Tk/ocr deps, and README documents macOS XQuartz steps for GUI in Docker.

### Changed
- Default Docker CMD launches the viewer; README clarifies active development, HF onboarding, and Docker usage.

## 0.1.0
### Added
- New `docai_toolkit` package with OCR (local Tesseract or remote endpoint), embeddings/indexing (local or remote), and chat over FAISS.
- GUI updates: OCR → Markdown flow, chat with docs, settings persistence to `~/.docai/config.json`, background threads for OCR/chat, and HF/custom endpoint support.
- Dockerfile (non-root) and GitHub Actions workflows for Docker build and PyPI trusted publishing.
- `pyproject.toml` packaging with console entrypoint `docai-viewer`.
- Tests for PDF writer, remote OCR parsing, and remote embeddings.

### Changed
- Renamed from `docai` to `docai_toolkit` to avoid PyPI naming conflicts.
- Hardened HF client (timeouts/errors), safer FAISS load (no dangerous deserialization), and remote embedding batching.
- README refresh with install, HF onboarding, Docker/XQuartz instructions, and macOS GUI notes.

### Removed
- Legacy training scripts and legacy `pdf_ui.py` entrypoint.
