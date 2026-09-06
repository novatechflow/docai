"""Classify files in a tree as documents, non-documents, or ambiguous.

A byte-for-byte match on extension is not enough on an untrusted dump: files
are mislabelled, extensionless, or renamed binaries. Every candidate is checked
two ways — its extension and its detected content type — and only kept when both
agree it is a document. Anything that disagrees or cannot be identified is
quarantined for review rather than fed downstream.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

DOCUMENT_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".rtf", ".odt", ".txt", ".md",
}

DOCUMENT_MIME_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/rtf",
    "text/rtf",
    "application/vnd.oasis.opendocument.text",
    "application/x-tika-ooxml",  # some libmagic builds report OOXML generically
    "text/plain",
    "text/markdown",
}

ZIP_BACKED_MIME = "application/zip"  # docx/odt are zip containers; magic may say so
ZIP_BACKED_EXTENSIONS = {".docx", ".odt"}

MAX_REASONABLE_BYTES = 512 * 1024 * 1024  # documents past this are suspect on a dump

_SIGNATURES = (
    (b"%PDF-", "application/pdf"),
    (b"{\\rtf", "application/rtf"),
    (b"PK\x03\x04", ZIP_BACKED_MIME),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "application/msword"),  # OLE2 (legacy .doc)
)


class Verdict(str, Enum):
    DOCUMENT = "document"
    NOT_DOCUMENT = "not_document"
    QUARANTINE = "quarantine"


@dataclass
class FileReport:
    path: str
    size: int
    extension: str
    detected_mime: Optional[str]
    verdict: Verdict
    reason: str


@dataclass
class ClassificationSummary:
    root: str
    counts: Dict[str, int] = field(default_factory=dict)
    bytes_by_verdict: Dict[str, int] = field(default_factory=dict)
    documents: List[FileReport] = field(default_factory=list)
    quarantined: List[FileReport] = field(default_factory=list)


MimeDetector = Callable[[Path], Optional[str]]


def _detect_with_python_magic() -> Optional[MimeDetector]:
    try:
        import magic  # type: ignore
    except ImportError:
        return None
    detector = magic.Magic(mime=True)

    def detect(path: Path) -> Optional[str]:
        try:
            return detector.from_file(str(path)) or None
        except OSError:
            return None

    return detect


def _detect_with_file_cli() -> Optional[MimeDetector]:
    from shutil import which

    if which("file") is None:
        return None

    def detect(path: Path) -> Optional[str]:
        try:
            result = subprocess.run(
                ["file", "--brief", "--mime-type", str(path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        mime = result.stdout.strip()
        return mime or None

    return detect


def _detect_with_signatures() -> MimeDetector:
    def detect(path: Path) -> Optional[str]:
        try:
            with open(path, "rb") as handle:
                head = handle.read(8)
        except OSError:
            return None
        for signature, mime in _SIGNATURES:
            if head.startswith(signature):
                return mime
        return None

    return detect


def default_mime_detector() -> MimeDetector:
    """Best available detector: python-magic, then `file`, then signatures."""
    return _detect_with_python_magic() or _detect_with_file_cli() or _detect_with_signatures()


PLAIN_TEXT_EXTENSIONS = {".txt", ".md"}


def _looks_like_text(path: Path) -> bool:
    """Sniff a prefix for text. libmagic reports octet-stream on tiny inputs."""
    try:
        with open(path, "rb") as handle:
            chunk = handle.read(8192)
    except OSError:
        return False
    if b"\x00" in chunk:
        return False
    try:
        chunk.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return chunk.decode("latin-1", errors="strict") is not None


def _mime_agrees_with_extension(extension: str, mime: str, path: Path) -> bool:
    if extension in PLAIN_TEXT_EXTENSIONS:
        if mime.startswith("text/"):
            return True
        return _looks_like_text(path)
    if mime in DOCUMENT_MIME_TYPES:
        return True
    if mime == ZIP_BACKED_MIME and extension in ZIP_BACKED_EXTENSIONS:
        return True
    return False


def classify_file(path: Path, detect_mime: MimeDetector) -> FileReport:
    extension = path.suffix.lower()
    try:
        size = path.stat().st_size
    except OSError as exc:
        return FileReport(str(path), 0, extension, None, Verdict.QUARANTINE, f"stat failed: {exc}")

    if size == 0:
        return FileReport(str(path), 0, extension, None, Verdict.QUARANTINE, "empty file")

    if extension not in DOCUMENT_EXTENSIONS:
        return FileReport(str(path), size, extension, None, Verdict.NOT_DOCUMENT, "extension not a document type")

    mime = detect_mime(path)
    if mime is None:
        return FileReport(str(path), size, extension, None, Verdict.QUARANTINE, "content type undetermined")

    if not _mime_agrees_with_extension(extension, mime, path):
        return FileReport(str(path), size, extension, mime, Verdict.QUARANTINE, "extension and content type disagree")

    if size > MAX_REASONABLE_BYTES:
        return FileReport(str(path), size, extension, mime, Verdict.QUARANTINE, "document larger than expected")

    return FileReport(str(path), size, extension, mime, Verdict.DOCUMENT, "extension and content type agree")


def iter_files(root: Path, follow_symlinks: bool = False) -> Iterator[Path]:
    for entry in sorted(root.rglob("*")):
        if entry.is_symlink() and not follow_symlinks:
            continue
        if entry.is_file():
            yield entry


def classify_tree(
    root: Path,
    detect_mime: Optional[MimeDetector] = None,
    follow_symlinks: bool = False,
) -> ClassificationSummary:
    detect_mime = detect_mime or default_mime_detector()
    summary = ClassificationSummary(
        root=str(root),
        counts={v.value: 0 for v in Verdict},
        bytes_by_verdict={v.value: 0 for v in Verdict},
    )
    for path in iter_files(root, follow_symlinks=follow_symlinks):
        report = classify_file(path, detect_mime)
        summary.counts[report.verdict.value] += 1
        summary.bytes_by_verdict[report.verdict.value] += report.size
        if report.verdict is Verdict.DOCUMENT:
            summary.documents.append(report)
        elif report.verdict is Verdict.QUARANTINE:
            summary.quarantined.append(report)
    return summary


def write_manifest(summary: ClassificationSummary, path: Path) -> None:
    payload = {
        "root": summary.root,
        "counts": summary.counts,
        "bytes_by_verdict": summary.bytes_by_verdict,
        "documents": [asdict(r) for r in summary.documents],
        "quarantined": [asdict(r) for r in summary.quarantined],
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Classify a file tree into documents / non-documents / quarantine.")
    parser.add_argument("root", type=Path, help="Directory to scan recursively.")
    parser.add_argument("--manifest", type=Path, help="Write a JSON manifest to this path.")
    parser.add_argument("--doc-list", type=Path, help="Write kept document paths (one per line) to this path.")
    parser.add_argument("--follow-symlinks", action="store_true", help="Descend into symlinked files/dirs.")
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        parser.error(f"{args.root} is not a directory")

    summary = classify_tree(args.root, follow_symlinks=args.follow_symlinks)

    for verdict in Verdict:
        key = verdict.value
        print(f"{key:13} {summary.counts[key]:>10}  {_human_bytes(summary.bytes_by_verdict[key])}")

    if args.manifest:
        write_manifest(summary, args.manifest)
        print(f"manifest -> {args.manifest}")
    if args.doc_list:
        args.doc_list.write_text("\n".join(r.path for r in summary.documents) + "\n", encoding="utf-8")
        print(f"doc-list -> {args.doc_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
