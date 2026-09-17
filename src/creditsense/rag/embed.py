"""Embedding pipeline (P4.3).

text-embedding-3-small, 1536 dimensions — matches the existing Vector(1536) column on
Chunk (src/creditsense/db/models.py), so no further migration is needed.

Caches to disk keyed by a hash of the text, so re-ingesting the corpus during
development does not re-bill or re-wait on the API for text that hasn't changed.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path

from openai import APIError, APIStatusError, OpenAI, RateLimitError

from creditsense.config import get_settings

DEFAULT_CACHE_PATH = Path("src/creditsense/data/raw_corpus/.embedding_cache.jsonl")
MAX_BATCH_SIZE = 128
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_settings().openai_api_key)
    return _client


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """A flat JSONL file: one {"key": ..., "embedding": [...]} row per unique text."""

    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH) -> None:
        self.path = Path(path)
        self._entries: dict[str, list[float]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self._entries[row["key"]] = row["embedding"]

    def get(self, text: str) -> list[float] | None:
        return self._entries.get(_text_key(text))

    def put(self, text: str, embedding: list[float]) -> None:
        key = _text_key(text)
        if key in self._entries:
            return
        self._entries[key] = embedding
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "embedding": embedding}) + "\n")


def _embed_batch_with_retry(
    client: OpenAI, texts: list[str], model: str
) -> list[list[float]]:
    backoff = INITIAL_BACKOFF_SECONDS
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]
        except RateLimitError as exc:
            last_error = exc
        except APIStatusError as exc:
            if exc.status_code < 500:
                raise
            last_error = exc
        except APIError as exc:
            last_error = exc
        if attempt < MAX_RETRIES - 1:
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(
        f"Embedding batch failed after {MAX_RETRIES} attempts"
    ) from last_error


def embed_texts(
    texts: Sequence[str],
    *,
    batch_size: int = MAX_BATCH_SIZE,
    cache: EmbeddingCache | None = None,
    model: str | None = None,
) -> list[list[float]]:
    """Embed `texts`, in order, using the cache to skip anything already embedded."""
    settings = get_settings()
    model = model or settings.embedding_model
    cache = cache if cache is not None else EmbeddingCache()
    client = _get_client()

    results: list[list[float] | None] = [None] * len(texts)
    to_embed_indices: list[int] = []
    to_embed_texts: list[str] = []

    for i, text in enumerate(texts):
        cached = cache.get(text)
        if cached is not None:
            results[i] = cached
        else:
            to_embed_indices.append(i)
            to_embed_texts.append(text)

    for start in range(0, len(to_embed_texts), batch_size):
        batch = to_embed_texts[start : start + batch_size]
        batch_indices = to_embed_indices[start : start + batch_size]
        embeddings = _embed_batch_with_retry(client, batch, model)
        for idx, text, embedding in zip(batch_indices, batch, embeddings):
            results[idx] = embedding
            cache.put(text, embedding)

    return results  # type: ignore[return-value]
