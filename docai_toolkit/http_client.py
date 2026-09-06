"""HTTP transport: a scheme-guarded JSON client and an OpenAI-compatible client.

Remote generation and embeddings speak the OpenAI REST API
(`/v1/chat/completions`, `/v1/embeddings`), so any runner that implements it —
Ollama, vLLM, llama.cpp's server, TGI, or a hosted gateway — works through the
same client. Point ``base_url`` at the server's ``/v1`` root.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def validate_endpoint(url: str) -> str:
    """Reject schemes that would turn an endpoint call into a local file or intranet read."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "https" and parsed.hostname:
        return url
    if parsed.scheme == "http" and parsed.hostname in LOOPBACK_HOSTS:
        return url
    raise ValueError(
        f"Refusing to call endpoint {url!r}: only https endpoints are allowed "
        "(http is permitted for localhost only)."
    )


class HttpClient:
    """Minimal POST-JSON client with endpoint-scheme validation."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 60.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def post(
        self,
        url: str,
        payload: Dict[str, Any] | bytes,
        content_type: str = "application/json",
    ) -> Any:
        url = validate_endpoint(url)
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data)
        req.add_header("Content-Type", content_type)
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310 - scheme validated above
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Request to {url} failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Request to {url} failed: {exc}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body


class OpenAIClient:
    """Client for an OpenAI-compatible server (Ollama, vLLM, llama.cpp, hosted APIs)."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = HttpClient(api_key=api_key, timeout=timeout)

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        resp = self._http.post(self._url("chat/completions"), payload)
        try:
            return resp["choices"][0]["message"]["content"]
        except (TypeError, KeyError, IndexError) as exc:
            raise ValueError(f"Unexpected chat completion response: {resp!r}") from exc

    def embed(self, inputs: List[str]) -> List[List[float]]:
        resp = self._http.post(self._url("embeddings"), {"model": self.model, "input": inputs})
        try:
            data = resp["data"]
        except (TypeError, KeyError) as exc:
            raise ValueError(f"Unexpected embeddings response: {resp!r}") from exc
        if len(data) != len(inputs):
            raise ValueError(f"Expected {len(inputs)} embeddings, got {len(data)}.")
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in ordered]
