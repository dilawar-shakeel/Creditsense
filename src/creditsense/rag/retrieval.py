"""Hybrid retrieval (P4.4, P4.5): keyword (BM25-style) + dense, fused with Reciprocal
Rank Fusion, with an optional LLM reranker layered on top (rerank.py).

Keyword search uses ts_rank_cd against the generated content_tsv column (migration
0002). Fix (2026-09-16): the query is matched with OR semantics between terms, not
AND. websearch_to_tsquery's default — every term in the query must be present — was
tried first and returned ZERO results for all 41 eval queries: a natural-language
question has 8-12 words, this corpus's vocabulary is narrow (87 chunks), and it only
takes one non-matching word (a different tense, "per-party" vs. the corpus's
"Per Party", etc.) to fail the whole AND and return nothing. OR semantics plus
ts_rank_cd ranking documents higher for more overlapping terms is a much closer
approximation of real BM25 behavior anyway — BM25 doesn't require full keyword
coverage, it weights by term overlap. The trade-off: websearch_to_tsquery's quoted-
phrase and "-exclusion" syntax is lost; not needed for the plain-question queries
this system actually receives.

Dense search uses pgvector cosine distance (`<=>`).

Fusion follows the standard RRF formula: score = sum(1 / (k + rank)) across the ranked
lists a chunk appears in, k=60 (the constant from the original Cormack et al. RRF
paper — large enough that no single list's rank-1 result automatically dominates).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import text
from sqlalchemy.orm import Session

from creditsense.config import get_settings
from creditsense.rag.embed import embed_texts

DEFAULT_RRF_K = 60
DEFAULT_FUSION_LIMIT = 20


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    regulation_number: str
    clause_title: str
    section_path: str
    content: str
    source_type: str
    cross_references: list[str]
    score: float
    source: Literal["bm25", "dense", "fused", "reranked"]


_BM25_SQL = text(
    """
    SELECT id, regulation_number, clause_title, section_path, content, source_type,
           cross_references,
           ts_rank_cd(content_tsv, query) AS rank
    FROM chunks,
         to_tsquery(
             'english',
             replace(plainto_tsquery('english', :query)::text, ' & ', ' | ')
         ) AS query
    WHERE content_tsv @@ query
    ORDER BY rank DESC
    LIMIT :limit
    """
)

_DENSE_SQL = text(
    """
    SELECT id, regulation_number, clause_title, section_path, content, source_type,
           cross_references,
           1 - (embedding <=> (:query_vec)::vector) AS similarity
    FROM chunks
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> (:query_vec)::vector
    LIMIT :limit
    """
)

_BY_REGULATION_SQL = text(
    """
    SELECT id, regulation_number, clause_title, section_path, content, source_type,
           cross_references
    FROM chunks
    WHERE regulation_number = :regulation_number
    ORDER BY chunk_index
    """
)


def _row_to_chunk(row, score: float, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row.id,
        regulation_number=row.regulation_number or "",
        clause_title=row.clause_title or "",
        section_path=row.section_path or "",
        content=row.content,
        source_type=row.source_type or "",
        cross_references=list(row.cross_references or []),
        score=score,
        source=source,  # type: ignore[arg-type]
    )


def bm25_search(session: Session, query: str, *, limit: int = DEFAULT_FUSION_LIMIT) -> list[RetrievedChunk]:
    rows = session.execute(_BM25_SQL, {"query": query, "limit": limit}).all()
    return [_row_to_chunk(row, float(row.rank), "bm25") for row in rows]


def dense_search(session: Session, query: str, *, limit: int = DEFAULT_FUSION_LIMIT) -> list[RetrievedChunk]:
    embedding = embed_texts([query])[0]
    vector_literal = "[" + ",".join(str(x) for x in embedding) + "]"
    rows = session.execute(
        _DENSE_SQL, {"query_vec": vector_literal, "limit": limit}
    ).all()
    return [_row_to_chunk(row, float(row.similarity), "dense") for row in rows]


def lookup_by_regulation_number(session: Session, regulation_number: str) -> list[RetrievedChunk]:
    """A regulation_number filter short-circuits to a direct lookup — asking for R-5
    by name should never depend on retrieval luck."""
    rows = session.execute(
        _BY_REGULATION_SQL, {"regulation_number": regulation_number}
    ).all()
    return [_row_to_chunk(row, 1.0, "bm25") for row in rows]


def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedChunk], k: int = DEFAULT_RRF_K, limit: int = DEFAULT_FUSION_LIMIT
) -> list[RetrievedChunk]:
    """score = sum(1 / (k + rank)) over every list a chunk appears in (rank is
    1-indexed). Pure function — no database needed, so it's fully unit-testable
    against hand-built rank lists."""
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunks_by_id.setdefault(chunk.chunk_id, chunk)

    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:limit]
    return [
        RetrievedChunk(
            chunk_id=chunks_by_id[cid].chunk_id,
            regulation_number=chunks_by_id[cid].regulation_number,
            clause_title=chunks_by_id[cid].clause_title,
            section_path=chunks_by_id[cid].section_path,
            content=chunks_by_id[cid].content,
            source_type=chunks_by_id[cid].source_type,
            cross_references=chunks_by_id[cid].cross_references,
            score=scores[cid],
            source="fused",
        )
        for cid in ordered_ids
    ]


def _bm25_search_with_own_connection(engine, query: str, limit: int) -> list[RetrievedChunk]:
    with engine.connect() as connection:
        rows = connection.execute(_BM25_SQL, {"query": query, "limit": limit}).all()
        return [_row_to_chunk(row, float(row.rank), "bm25") for row in rows]


def _dense_search_with_own_connection(engine, query: str, limit: int) -> list[RetrievedChunk]:
    embedding = embed_texts([query])[0]
    vector_literal = "[" + ",".join(str(x) for x in embedding) + "]"
    with engine.connect() as connection:
        rows = connection.execute(
            _DENSE_SQL, {"query_vec": vector_literal, "limit": limit}
        ).all()
        return [_row_to_chunk(row, float(row.similarity), "dense") for row in rows]


async def _run_bm25_and_dense_concurrently(
    session: Session, query: str, limit: int
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    # A SQLAlchemy Session is not safe for concurrent use from two threads at once,
    # so the concurrent path does NOT share `session` across the two calls — each
    # gets its own short-lived Connection from the same Engine (session.get_bind()),
    # which *is* safe to use from multiple threads/tasks simultaneously. The public
    # bm25_search()/dense_search() functions above keep taking a Session for the
    # simple single-call and test-facing case.
    engine = session.get_bind()
    bm25_task = asyncio.to_thread(_bm25_search_with_own_connection, engine, query, limit)
    dense_task = asyncio.to_thread(_dense_search_with_own_connection, engine, query, limit)
    return await asyncio.gather(bm25_task, dense_task)


def hybrid_search(
    session: Session,
    query: str,
    *,
    regulation_number: str | None = None,
    top_k: int | None = None,
    use_reranker: bool = True,
) -> list[RetrievedChunk]:
    if regulation_number is not None:
        return lookup_by_regulation_number(session, regulation_number)

    settings = get_settings()
    top_k = top_k or settings.rag_top_k

    try:
        bm25_results, dense_results = asyncio.run(
            _run_bm25_and_dense_concurrently(session, query, DEFAULT_FUSION_LIMIT)
        )
    except RuntimeError:
        # Already inside an event loop (e.g. called from an async FastAPI route) —
        # fall back to sequential calls rather than nesting asyncio.run().
        bm25_results = bm25_search(session, query, limit=DEFAULT_FUSION_LIMIT)
        dense_results = dense_search(session, query, limit=DEFAULT_FUSION_LIMIT)

    fused = reciprocal_rank_fusion(
        bm25_results, dense_results, k=settings.rrf_k, limit=DEFAULT_FUSION_LIMIT
    )

    if not use_reranker:
        return fused[:top_k]

    from creditsense.rag.rerank import rerank  # local import: keeps reranker/openai
    # out of the hot path for tests that pass use_reranker=False and never touch the
    # network.

    return rerank(query, fused, top_k=top_k)
