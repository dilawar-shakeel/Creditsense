from unittest.mock import patch

from creditsense.rag.retrieval import RetrievedChunk, reciprocal_rank_fusion


def _chunk(chunk_id: str, source: str = "bm25") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        regulation_number=f"R-{chunk_id}",
        clause_title=f"Title {chunk_id}",
        section_path="Part-II",
        content=f"Content {chunk_id}",
        source_type="sbp_regulation",
        cross_references=[],
        score=1.0,
        source=source,  # type: ignore[arg-type]
    )


def test_rrf_matches_hand_computed_scores():
    # bm25 order: A, B, C   dense order: B, A, D
    bm25 = [_chunk("A", "bm25"), _chunk("B", "bm25"), _chunk("C", "bm25")]
    dense = [_chunk("B", "dense"), _chunk("A", "dense"), _chunk("D", "dense")]

    fused = reciprocal_rank_fusion(bm25, dense, k=60)

    expected = {
        "A": 1 / 61 + 1 / 62,  # rank 1 in bm25, rank 2 in dense
        "B": 1 / 62 + 1 / 61,  # rank 2 in bm25, rank 1 in dense
        "C": 1 / 63,  # rank 3 in bm25 only
        "D": 1 / 63,  # rank 3 in dense only
    }
    scores_by_id = {c.chunk_id: c.score for c in fused}
    for chunk_id, expected_score in expected.items():
        assert scores_by_id[chunk_id] == expected_score

    # A and B tie exactly (symmetric ranks 1+2 vs 2+1), both above C and D
    assert set(c.chunk_id for c in fused[:2]) == {"A", "B"}


def test_rrf_respects_limit():
    bm25 = [_chunk(str(i)) for i in range(10)]
    fused = reciprocal_rank_fusion(bm25, limit=3)
    assert len(fused) == 3


def test_rrf_deduplicates_chunks_appearing_in_both_lists():
    bm25 = [_chunk("A")]
    dense = [_chunk("A")]
    fused = reciprocal_rank_fusion(bm25, dense)
    assert len(fused) == 1


def test_rrf_marks_results_as_fused():
    fused = reciprocal_rank_fusion([_chunk("A")])
    assert all(c.source == "fused" for c in fused)


def test_rrf_with_no_lists_returns_empty():
    assert reciprocal_rank_fusion() == []


def test_rerank_degrades_to_fused_order_on_api_failure():
    from creditsense.rag.rerank import rerank

    candidates = [_chunk("A"), _chunk("B"), _chunk("C")]

    with patch("creditsense.rag.rerank._get_client", side_effect=RuntimeError("boom")):
        result = rerank("some query", candidates, top_k=2)

    # Falls back to the fused order, truncated to top_k, not an exception.
    assert [c.chunk_id for c in result] == ["A", "B"]


def test_rerank_empty_candidates_returns_empty():
    from creditsense.rag.rerank import rerank

    assert rerank("query", []) == []
