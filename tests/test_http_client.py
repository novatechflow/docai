import pytest

from docai_toolkit.http_client import HttpClient, OpenAIClient, validate_endpoint


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com/x", "http://example.com/infer", "https://"],
)
def test_validate_endpoint_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        validate_endpoint(url)


@pytest.mark.parametrize("url", ["https://example.com/infer", "http://localhost:8080/infer"])
def test_validate_endpoint_accepts_safe_urls(url):
    assert validate_endpoint(url) == url


def test_http_post_refuses_file_scheme(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("token-material")
    client = HttpClient(api_key="tok")
    with pytest.raises(ValueError):
        client.post(f"file://{secret}", {"input": "x"})


def _client_with_captured_post(response):
    client = OpenAIClient("https://api.example.com/v1", api_key="k", model="m")
    captured = {}

    def fake_post(url, payload, content_type="application/json"):
        captured["url"] = url
        captured["payload"] = payload
        return response

    client._http.post = fake_post  # type: ignore[assignment]
    return client, captured


def test_chat_builds_openai_request_and_parses_content():
    response = {"choices": [{"message": {"role": "assistant", "content": "42"}}]}
    client, captured = _client_with_captured_post(response)

    answer = client.chat([{"role": "user", "content": "q"}], max_tokens=16)

    assert answer == "42"
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["payload"]["model"] == "m"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "q"}]
    assert captured["payload"]["max_tokens"] == 16


def test_chat_raises_on_unexpected_shape():
    client, _ = _client_with_captured_post({"unexpected": True})
    with pytest.raises(ValueError):
        client.chat([{"role": "user", "content": "q"}])


def test_embed_orders_by_index_and_returns_vectors():
    response = {
        "data": [
            {"index": 1, "embedding": [0.3, 0.4]},
            {"index": 0, "embedding": [0.1, 0.2]},
        ]
    }
    client, captured = _client_with_captured_post(response)

    vectors = client.embed(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["url"] == "https://api.example.com/v1/embeddings"
    assert captured["payload"] == {"model": "m", "input": ["a", "b"]}


def test_embed_rejects_count_mismatch():
    response = {"data": [{"index": 0, "embedding": [0.1]}]}
    client, _ = _client_with_captured_post(response)
    with pytest.raises(ValueError):
        client.embed(["a", "b"])
