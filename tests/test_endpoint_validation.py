import pytest

from docai_toolkit.hf_client import HuggingFaceClient, validate_endpoint


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "http://example.com/infer",
        "https://",
    ],
)
def test_validate_endpoint_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        validate_endpoint(url)


@pytest.mark.parametrize("url", ["https://example.com/infer", "http://localhost:8080/infer"])
def test_validate_endpoint_accepts_safe_urls(url):
    assert validate_endpoint(url) == url


def test_post_json_refuses_file_scheme(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("token-material")

    client = HuggingFaceClient("tok", default_endpoint=f"file://{secret}")
    with pytest.raises(ValueError):
        client.post_json({"inputs": "x"})
