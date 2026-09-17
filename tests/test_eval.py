import math

from creditsense.rag.eval import _metrics_for_query
from creditsense.rag.retrieval import RetrievedChunk


def _chunk(regulation_number: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=regulation_number,
        regulation_number=regulation_number,
        clause_title="",
        section_path="",
        content="",
        source_type="sbp_regulation",
        cross_references=[],
        score=1.0,
        source="fused",
    )


def test_precision_at_5_counts_relevant_in_top_5():
    retrieved = [_chunk(r) for r in ["R-1", "R-5", "R-2", "R-3", "R-4", "R-6"]]
    metrics = _metrics_for_query(retrieved, relevant={"R-5", "R-6"})
    # R-5 is in top 5, R-6 is not (rank 6) -> 1/5
    assert metrics.precision_at_5 == 1 / 5


def test_recall_at_10_counts_relevant_found_in_top_10():
    retrieved = [_chunk(f"R-{i}") for i in range(1, 12)]
    metrics = _metrics_for_query(retrieved, relevant={"R-1", "R-9", "R-11"})
    # R-1 (rank1) and R-9 (rank9) are in top 10; R-11 (rank11) is not
    assert metrics.recall_at_10 == 2 / 3


def test_mrr_is_reciprocal_of_first_relevant_rank():
    retrieved = [_chunk(r) for r in ["R-9", "R-8", "R-5"]]
    metrics = _metrics_for_query(retrieved, relevant={"R-5"})
    assert metrics.reciprocal_rank == 1 / 3


def test_mrr_zero_when_nothing_relevant_retrieved():
    retrieved = [_chunk(r) for r in ["R-9", "R-8"]]
    metrics = _metrics_for_query(retrieved, relevant={"R-5"})
    assert metrics.reciprocal_rank == 0.0


def test_ndcg_at_5_is_one_for_perfect_ranking():
    retrieved = [_chunk(r) for r in ["R-1", "R-2"]]
    metrics = _metrics_for_query(retrieved, relevant={"R-1", "R-2"})
    assert metrics.ndcg_at_5 == 1.0


def test_ndcg_at_5_penalizes_relevant_result_ranked_lower():
    # relevant result at rank 2 instead of rank 1
    retrieved = [_chunk(r) for r in ["R-9", "R-1"]]
    metrics = _metrics_for_query(retrieved, relevant={"R-1"})
    expected_dcg = 1.0 / math.log2(3)  # rank 2 -> log2(2+1)
    expected_idcg = 1.0 / math.log2(2)  # ideal: rank 1 -> log2(1+1)
    assert metrics.ndcg_at_5 == expected_dcg / expected_idcg


def test_out_of_scope_query_with_no_relevant_ids_returns_zeroed_metrics():
    retrieved = [_chunk("R-1")]
    metrics = _metrics_for_query(retrieved, relevant=set())
    assert metrics.precision_at_5 == 0.0
    assert metrics.recall_at_10 == 0.0
    assert metrics.reciprocal_rank == 0.0
    assert metrics.ndcg_at_5 == 0.0
