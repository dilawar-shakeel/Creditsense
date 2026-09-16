"""LLM reranker (P4.6).

Scores all fused candidates in a single call rather than one call per candidate —
cheaper, faster, and lets the model compare candidates against each other instead of
scoring each in isolation.

Degrades instead of failing: if the API errors or the response can't be parsed, the
fused order is returned unchanged and the failure is logged. A reranker outage must
not take retrieval down — this is half the answer to "what happens when the LLM
misbehaves" (tracker interview question #4).
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from creditsense.config import get_settings
from creditsense.rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

_RERANK_SYSTEM_PROMPT = """You are scoring candidate regulation/policy clauses for \
relevance to a user's question, for an SME credit compliance system. Score each \
candidate 0-10 (10 = directly and completely answers the question, 0 = irrelevant). \
Respond with ONLY a JSON array of objects: [{"id": "<chunk_id>", "score": <0-10>}, ...] \
— one entry per candidate, in any order. No prose, no markdown fences."""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_settings().openai_api_key)
    return _client


def _build_user_prompt(query: str, candidates: list[RetrievedChunk]) -> str:
    lines = [f"Question: {query}", "", "Candidates:"]
    for candidate in candidates:
        lines.append(
            f"- id: {candidate.chunk_id} | {candidate.regulation_number}: "
            f"{candidate.clause_title}\n  {candidate.content[:500]}"
        )
    return "\n".join(lines)


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    *,
    top_k: int = 5,
    model: str | None = None,
) -> list[RetrievedChunk]:
    if not candidates:
        return []

    settings = get_settings()
    model = model or settings.rerank_model

    try:
        response = _get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(query, candidates)},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content or "[]"
        scores = {row["id"]: float(row["score"]) for row in json.loads(raw)}
    except Exception:
        logger.warning(
            "Reranker failed; returning fused order unchanged", exc_info=True
        )
        return candidates[:top_k]

    by_id = {c.chunk_id: c for c in candidates}
    ranked_ids = sorted(
        (cid for cid in scores if cid in by_id), key=lambda cid: scores[cid], reverse=True
    )
    # Anything the model didn't score (a malformed/partial response) still appears,
    # after the scored ones, in its original fused order — never silently dropped.
    unscored_ids = [c.chunk_id for c in candidates if c.chunk_id not in scores]

    reranked = []
    for cid in ranked_ids + unscored_ids:
        original = by_id[cid]
        reranked.append(
            RetrievedChunk(
                chunk_id=original.chunk_id,
                regulation_number=original.regulation_number,
                clause_title=original.clause_title,
                section_path=original.section_path,
                content=original.content,
                source_type=original.source_type,
                cross_references=original.cross_references,
                score=scores.get(cid, original.score),
                source="reranked",
            )
        )
    return reranked[:top_k]
