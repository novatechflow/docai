from typing import List, Optional

from docai_toolkit.http_client import OpenAIClient

try:
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline
except ImportError as _chat_import_error:  # pragma: no cover - optional dependency
    HuggingFacePipeline = None  # type: ignore[assignment]
    pipeline = None  # type: ignore[assignment]
    _CHAT_IMPORT_ERROR = _chat_import_error
else:
    _CHAT_IMPORT_ERROR = None

SYSTEM_PROMPT = "Answer the question using only the provided context."


def chat_over_corpus(
    db,
    query: str,
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.1",
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    max_new_tokens: int = 256,
    k: int = 4,
) -> str:
    """Retrieve-then-generate over a FAISS db.

    With ``endpoint`` set, generation goes to an OpenAI-compatible server
    (Ollama, vLLM, llama.cpp, a hosted API); ``endpoint`` is the server's
    ``/v1`` base URL and ``model_id`` is the served model name. Without it,
    a local transformers pipeline runs ``model_id``.
    """
    docs = db.similarity_search(query, k=k)
    context_blocks: List[str] = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        context_blocks.append(f"Source: {src}\n{doc.page_content}")
    context = "\n\n".join(context_blocks)

    user_prompt = f"Context:\n{context}\n\nQuestion:\n{query}"

    if endpoint:
        client = OpenAIClient(endpoint, api_key=api_key, model=model_id)
        return client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_new_tokens,
        )

    if pipeline is None or HuggingFacePipeline is None:
        raise RuntimeError("transformers/langchain-community required for local HF generation.")

    prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nAnswer:"
    pipe = pipeline("text-generation", model=model_id, device_map="auto")
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm.invoke(prompt)
