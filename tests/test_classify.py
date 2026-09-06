import struct
import zipfile

import pytest

from docai_toolkit.classify import (
    Verdict,
    classify_file,
    classify_tree,
    default_mime_detector,
    write_manifest,
)


@pytest.fixture
def detector():
    return default_mime_detector()


def _write_pdf(path):
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")


def _write_docx(path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", "<document/>")


def _write_png(path):
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)


def test_real_pdf_is_kept(tmp_path, detector):
    p = tmp_path / "report.pdf"
    _write_pdf(p)
    assert classify_file(p, detector).verdict is Verdict.DOCUMENT


def test_real_docx_is_kept(tmp_path, detector):
    p = tmp_path / "letter.docx"
    _write_docx(p)
    report = classify_file(p, detector)
    assert report.verdict is Verdict.DOCUMENT


def test_plain_text_and_markdown_kept(tmp_path, detector):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    md = tmp_path / "readme.md"
    md.write_text("# hi")
    assert classify_file(txt, detector).verdict is Verdict.DOCUMENT
    assert classify_file(md, detector).verdict is Verdict.DOCUMENT


def test_media_file_is_dropped(tmp_path, detector):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16)
    assert classify_file(p, detector).verdict is Verdict.NOT_DOCUMENT


def test_binary_wearing_a_document_extension_is_quarantined(tmp_path, detector):
    p = tmp_path / "invoice.pdf"
    _write_png(p)
    report = classify_file(p, detector)
    assert report.verdict is Verdict.QUARANTINE
    assert "disagree" in report.reason


def test_empty_file_is_quarantined(tmp_path, detector):
    p = tmp_path / "empty.pdf"
    p.touch()
    report = classify_file(p, detector)
    assert report.verdict is Verdict.QUARANTINE
    assert report.reason == "empty file"


def test_oversized_document_is_quarantined(tmp_path, detector, monkeypatch):
    monkeypatch.setattr("docai_toolkit.classify.MAX_REASONABLE_BYTES", 4)
    p = tmp_path / "big.pdf"
    _write_pdf(p)
    report = classify_file(p, detector)
    assert report.verdict is Verdict.QUARANTINE
    assert "larger than expected" in report.reason


def test_undetermined_content_is_quarantined(tmp_path):
    p = tmp_path / "mystery.txt"
    p.write_bytes(b"\x00\x01\x02\x03")

    def blind_detector(_path):
        return None

    report = classify_file(p, blind_detector)
    assert report.verdict is Verdict.QUARANTINE
    assert report.reason == "content type undetermined"


def test_classify_tree_counts_and_manifest(tmp_path, detector):
    _write_pdf(tmp_path / "a.pdf")
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "c.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    _write_png(tmp_path / "d.pdf")  # renamed binary -> quarantine
    nested = tmp_path / "sub"
    nested.mkdir()
    _write_pdf(nested / "e.pdf")

    summary = classify_tree(tmp_path, detect_mime=detector)
    assert summary.counts[Verdict.DOCUMENT.value] == 3
    assert summary.counts[Verdict.NOT_DOCUMENT.value] == 1
    assert summary.counts[Verdict.QUARANTINE.value] == 1
    assert sum(summary.bytes_by_verdict.values()) > 0

    manifest = tmp_path / "manifest.json"
    write_manifest(summary, manifest)
    import json

    data = json.loads(manifest.read_text())
    assert len(data["documents"]) == 3
    assert len(data["quarantined"]) == 1


def test_symlinks_are_skipped_by_default(tmp_path, detector):
    real = tmp_path / "real.pdf"
    _write_pdf(real)
    link = tmp_path / "link.pdf"
    link.symlink_to(real)

    summary = classify_tree(tmp_path, detect_mime=detector)
    assert summary.counts[Verdict.DOCUMENT.value] == 1


def test_tiny_text_file_kept_despite_octet_stream(tmp_path, detector):
    p = tmp_path / "one.txt"
    p.write_bytes(b"x")
    assert classify_file(p, detector).verdict is Verdict.DOCUMENT


def test_binary_renamed_to_txt_is_quarantined(tmp_path, detector):
    p = tmp_path / "payload.txt"
    p.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 64)
    assert classify_file(p, detector).verdict is Verdict.QUARANTINE
