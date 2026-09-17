"""Shared LLM helper for the agents layer (Phase 5).

One place that talks to the model, so every agent goes through the same client and the
same failure behavior instead of five copies of try/except-around-an-API-call.

Two conventions carried over from `rag/rerank.py` on purpose:
  - a module-level `_client` singleton behind `_get_client()`, so tests can patch
    `creditsense.agents.llm._get_client` exactly the way
    `tests/test_retrieval.py` patches `creditsense.rag.rerank._get_client`.
  - degrade, never raise. rerank.py returns the fused order unchanged on failure rather
    than taking retrieval down; the analogous move here is returning `fallback` (None if
    the caller didn't give one) so the calling agent turns that into an escalation
    instead of a stack trace or a fabricated answer.

This uses LangChain (`langchain-core` + `langchain-openai`, per the Section 3 lock and
the P5.1 decision) rather than the raw OpenAI SDK rerank.py uses -- `with_structured_output`
is LangChain's equivalent of rerank.py's "ask for JSON, parse it," but validated against a
Pydantic schema instead of bare `json.loads`.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from creditsense.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client: ChatOpenAI | None = None


def _get_client() -> ChatOpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = ChatOpenAI(
            model=settings.agent_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    return _client


def complete_structured(
    system: str,
    user: str,
    schema: type[T],
    *,
    fallback: T | None = None,
) -> T | None:
    """One LLM call, parsed into `schema` via LangChain's structured-output mode.

    On ANY failure -- network error, schema-validation failure, refusal, timeout -- logs
    a warning with exc_info and returns `fallback` (None if not given). Never raises.
    Callers must treat a None return as "the LLM step produced nothing usable" and
    react the way the rest of this system already reacts to that: escalate, don't guess.
    """
    try:
        structured_client = _get_client().with_structured_output(schema)
        result = structured_client.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        if not isinstance(result, schema):
            raise TypeError(f"Expected {schema.__name__}, got {type(result).__name__}")
        return result
    except Exception:
        logger.warning(
            "LLM structured call failed (schema=%s); returning fallback",
            schema.__name__,
            exc_info=True,
        )
        return fallback
