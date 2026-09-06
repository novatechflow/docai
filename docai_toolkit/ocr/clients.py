from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from docai_toolkit.hf_client import HuggingFaceClient


@dataclass
class PageResult:
    page_number: int
    text: str


class OcrClient(ABC):
    @abstractmethod
    def recognize(self, pdf_path: Path) -> List[PageResult]:
        raise NotImplementedError


class RemoteOcrClient(OcrClient):
    """Generic OCR via a Hugging Face Inference or custom endpoint."""

    def __init__(self, api_key: Optional[str], endpoint: str) -> None:
        self.client = HuggingFaceClient(api_key, default_endpoint=endpoint)

    def recognize(self, pdf_path: Path) -> List[PageResult]:
        pdf_bytes = pdf_path.read_bytes()
        response = self.client.post_json(pdf_bytes, content_type="application/pdf")

        if isinstance(response, str):
            pages = [response]
        elif isinstance(response, list):
            pages = [str(item) for item in response]
        elif isinstance(response, dict):
            if "pages" in response:
                pages = [p.get("text", "") if isinstance(p, dict) else str(p) for p in response["pages"]]
            elif "text" in response:
                pages = [str(response["text"])]
            else:
                pages = [str(response)]
        else:
            pages = [""]

        return [PageResult(page_number=index + 1, text=text) for index, text in enumerate(pages)]


class TesseractOcrClient(OcrClient):
    """Local fallback using pytesseract + pdf2image."""

    def __init__(self) -> None:
        try:
            import pytesseract  # type: ignore
            from pdf2image import convert_from_path  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError("pytesseract and pdf2image are required for TesseractOcrClient.") from exc

        self._pytesseract = pytesseract
        self._convert_from_path = convert_from_path

    def recognize(self, pdf_path: Path) -> List[PageResult]:
        images = self._convert_from_path(str(pdf_path))
        return [
            PageResult(page_number=index + 1, text=self._pytesseract.image_to_string(image))
            for index, image in enumerate(images)
        ]
