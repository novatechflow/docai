from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_PATH = Path.home() / ".docai" / "config.json"


@dataclass
class OcrConfig:
    provider: str = "tesseract"
    api_key: Optional[str] = None
    endpoint: Optional[str] = None  # custom OCR endpoint (raw PDF POST)


@dataclass
class EmbeddingConfig:
    backend: str = "sentence-transformers"  # local sentence-transformers, or set endpoint for a served model
    model: str = "all-mpnet-base-v2"
    device: str = "auto"
    endpoint: Optional[str] = None  # OpenAI-compatible /v1 base URL
    api_key: Optional[str] = None


@dataclass
class LlmConfig:
    backend: str = "local"  # local transformers, or set endpoint for an OpenAI-compatible server
    model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    api_key: Optional[str] = None
    max_new_tokens: int = 256
    endpoint: Optional[str] = None  # OpenAI-compatible /v1 base URL


def _known_fields(cls, data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    allowed = {f.name for f in fields(cls)}
    return {key: value for key, value in data.items() if key in allowed}


@dataclass
class AppConfig:
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    ocr: OcrConfig = field(default_factory=OcrConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    env_token: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> "AppConfig":
        cfg = cls.load_from_file(CONFIG_PATH) or cls()
        hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("DOC_AI_HF_TOKEN")
        )
        if hf_token:
            cfg.env_token = hf_token
            cfg.llm.api_key = hf_token
            cfg.embeddings.api_key = hf_token

        output_dir_env = os.getenv("DOC_AI_OUTPUT_DIR")
        if output_dir_env:
            cfg.output_dir = Path(output_dir_env)
        return cfg

    @classmethod
    def load_from_file(cls, path: Optional[Path]) -> Optional["AppConfig"]:
        if not path or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        return cls(
            output_dir=Path(data.get("output_dir") or "./outputs"),
            ocr=OcrConfig(**_known_fields(OcrConfig, data.get("ocr"))),
            embeddings=EmbeddingConfig(**_known_fields(EmbeddingConfig, data.get("embeddings"))),
            llm=LlmConfig(**_known_fields(LlmConfig, data.get("llm"))),
        )

    def save(self, path: Optional[Path] = None) -> None:
        path = path or CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)

        data: Dict[str, Any] = {
            "output_dir": str(self.output_dir),
            "ocr": asdict(self.ocr),
            "embeddings": asdict(self.embeddings),
            "llm": asdict(self.llm),
        }
        for section in ("ocr", "embeddings", "llm"):
            if self.env_token and data[section].get("api_key") == self.env_token:
                data[section]["api_key"] = None

        handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(data, output, indent=2)
        os.chmod(path, 0o600)
