"""P8.1 gap: rag/embed.py had zero test references -- including its retry logic, which
is the module that spends money. Its retryable-vs-fatal split (retry a 5xx / rate
limit, re-raise a 4xx immediately rather than burning five attempts on a request that
will never succeed) is exactly the kind of thing that is wrong silently, so it is
worth proving.

The disk cache matters for the same reason: a broken cache means re-paying OpenAI for
text that hasn't changed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import APIStatusError, RateLimitError

from creditsense.rag.embed import (
    MAX_RETRIES,
    EmbeddingCache,
    _embed_batch_with_retry,
    _text_key,
    embed_texts,
)


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code=status_code, request=httpx.Request("POST", "https://api.openai.com/v1")
    )


def _api_status_error(status_code: int) -> APIStatusError:
    return APIStatusError("boom", response=_response(status_code), body=None)


def _rate_limit_error() -> RateLimitError:
    return RateLimitError("slow down", response=_response(429), body=None)


class _FakeEmbeddings:
    def __init__(self, *, results=None, errors=None):
        self._results = results
        self._errors = list(errors or [])
        self.call_count = 0

    def create(self, model, input):
        self.call_count += 1
        if self._errors:
            error = self._errors.pop(0)
            if error is not None:
                raise error
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in input])


class _FakeClient:
    def __init__(self, embeddings):
        self.embeddings = embeddings


# --- cache -------------------------------------------------------------------

def test_cache_round_trips_through_disk(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.jsonl")
    cache.put("hello", [0.1, 0.2])

    reloaded = EmbeddingCache(tmp_path / "cache.jsonl")
    assert reloaded.get("hello") == [0.1, 0.2]


def test_cache_misses_return_none(tmp_path):
    assert EmbeddingCache(tmp_path / "cache.jsonl").get("never seen") is None


def test_cache_does_not_write_the_same_text_twice(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = EmbeddingCache(path)
    cache.put("hello", [0.1])
    cache.put("hello", [0.9])

    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert cache.get("hello") == [0.1]


def test_cache_keys_are_content_addressed():
    assert _text_key("a") == _text_key("a")
    assert _text_key("a") != _text_key("b")


# --- retry -------------------------------------------------------------------

def test_a_successful_call_does_not_retry():
    embeddings = _FakeEmbeddings()
    result = _embed_batch_with_retry(_FakeClient(embeddings), ["one", "two"], "model")

    assert len(result) == 2
    assert embeddings.call_count == 1


def test_a_rate_limit_is_retried_then_succeeds():
    embeddings = _FakeEmbeddings(errors=[_rate_limit_error(), None])

    with patch("creditsense.rag.embed.time.sleep"):
        result = _embed_batch_with_retry(_FakeClient(embeddings), ["one"], "model")

    assert len(result) == 1
    assert embeddings.call_count == 2


def test_a_5xx_is_retried():
    embeddings = _FakeEmbeddings(errors=[_api_status_error(503), None])

    with patch("creditsense.rag.embed.time.sleep"):
        _embed_batch_with_retry(_FakeClient(embeddings), ["one"], "model")

    assert embeddings.call_count == 2


def test_a_4xx_is_raised_immediately_without_burning_retries():
    """A malformed request will never succeed -- retrying it five times just wastes
    time and money."""
    embeddings = _FakeEmbeddings(errors=[_api_status_error(400)])

    with patch("creditsense.rag.embed.time.sleep"):
        with pytest.raises(APIStatusError):
            _embed_batch_with_retry(_FakeClient(embeddings), ["one"], "model")

    assert embeddings.call_count == 1


def test_exhausting_every_retry_raises_a_runtime_error():
    """Unlike llm.py (which degrades), embed.py RAISES on exhaustion -- ingestion
    must fail loudly rather than silently writing a corpus with missing vectors."""
    embeddings = _FakeEmbeddings(errors=[_rate_limit_error()] * MAX_RETRIES)

    with patch("creditsense.rag.embed.time.sleep"):
        with pytest.raises(RuntimeError, match="failed after"):
            _embed_batch_with_retry(_FakeClient(embeddings), ["one"], "model")

    assert embeddings.call_count == MAX_RETRIES


# --- embed_texts -------------------------------------------------------------

def test_cached_texts_are_not_sent_to_the_api(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.jsonl")
    cache.put("already known", [0.5, 0.5])
    embeddings = _FakeEmbeddings()

    with patch("creditsense.rag.embed._get_client", return_value=_FakeClient(embeddings)):
        result = embed_texts(["already known"], cache=cache, model="model")

    assert result == [[0.5, 0.5]]
    assert embeddings.call_count == 0


def test_results_stay_in_input_order_when_only_some_are_cached(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.jsonl")
    cache.put("cached", [0.9, 0.9])
    embeddings = _FakeEmbeddings()

    with patch("creditsense.rag.embed._get_client", return_value=_FakeClient(embeddings)):
        result = embed_texts(["cached", "fresh"], cache=cache, model="model")

    assert result[0] == [0.9, 0.9]
    assert result[1] == [0.1, 0.2]
    assert cache.get("fresh") == [0.1, 0.2]


def test_texts_are_batched(tmp_path):
    embeddings = _FakeEmbeddings()

    with patch("creditsense.rag.embed._get_client", return_value=_FakeClient(embeddings)):
        result = embed_texts(
            ["a", "b", "c", "d", "e"],
            batch_size=2,
            cache=EmbeddingCache(tmp_path / "cache.jsonl"),
            model="model",
        )

    assert len(result) == 5
    assert embeddings.call_count == 3  # 2 + 2 + 1
