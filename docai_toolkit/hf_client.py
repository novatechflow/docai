"""Minimal Hugging Face / custom endpoint client."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

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


class HuggingFaceClient:
    def __init__(self, api_token: Optional[str], default_endpoint: Optional[str] = None, timeout: float = 30.0) -> None:
        self.api_token = api_token
        self.default_endpoint = default_endpoint
        self.timeout = timeout

    def post_json(
        self,
        payload: Dict[str, Any] | bytes,
        endpoint: Optional[str] = None,
        content_type: str = "application/json",
    ) -> Dict[str, Any] | Any:
        url = endpoint or self.default_endpoint
        if not url:
            raise ValueError("Endpoint is required for HuggingFaceClient.")
        url = validate_endpoint(url)

        if isinstance(payload, bytes):
            data = payload
        else:
            data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data)
        req.add_header("Content-Type", content_type)
        if self.api_token:
            req.add_header("Authorization", f"Bearer {self.api_token}")

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
