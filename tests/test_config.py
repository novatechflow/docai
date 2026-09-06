import json
import os
import stat

from docai_toolkit.config import AppConfig, OcrConfig


def test_save_writes_owner_only_permissions(tmp_path):
    config = AppConfig()
    config.llm.api_key = "hf_typed_by_user"
    path = tmp_path / ".docai" / "config.json"

    config.save(path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert json.loads(path.read_text())["llm"]["api_key"] == "hf_typed_by_user"


def test_save_omits_token_sourced_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    monkeypatch.setattr("docai_toolkit.config.CONFIG_PATH", tmp_path / "missing.json")

    config = AppConfig.from_env()
    assert config.llm.api_key == "hf_from_env"

    path = tmp_path / "config.json"
    config.save(path)

    saved = json.loads(path.read_text())
    assert saved["llm"]["api_key"] is None
    assert saved["embeddings"]["api_key"] is None
    assert "hf_from_env" not in path.read_text()


def test_load_from_file_tolerates_bad_content(tmp_path):
    path = tmp_path / "config.json"

    path.write_text("{ not json")
    assert AppConfig.load_from_file(path) is None

    path.write_text(json.dumps({"ocr": {"provider": "tesseract", "removed_key": 1}}))
    assert AppConfig.load_from_file(path).ocr == OcrConfig(provider="tesseract")


def test_default_ocr_provider_is_local():
    assert AppConfig().ocr.provider == "tesseract"
