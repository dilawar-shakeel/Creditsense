"""Retrieval evaluation harness (P4.7) — "what are your retrieval numbers?"

Reports precision@5, recall@10, MRR, and nDCG@5 across four configurations: keyword
only, dense only, fused (RRF), and fused + LLM reranked. The comparison table is the
deliverable, not any single number — it shows whether hybrid retrieval and reranking
actually earned their complexity.

Refuses to report anything until every row in eval_queries.jsonl is verified: true
(see eval_review.py) — an unverified, LLM-drafted label set would just measure the
corpus echoing its own phrasing back at itself.

Run: python -m creditsense.rag.eval
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from creditsense.db.session import session_scope
from creditsense.rag.rerank import rerank
from creditsense.rag.retrieval import (
    RetrievedChunk,
    bm25_search,
    dense_search,
    reciprocal_rank_fusion,
)

DEFAULT_QUERIES_PATH = Path("src/creditsense/data/raw_corpus/eval_queries.jsonl")
DEFAULT_REPORT_PATH = Path("reports/rag/retrieval_eval.md")

CONFIGURATIONS = ["keyword_only", "dense_only", "fused", "fused_reranked"]


@dataclass(frozen=True)
class QueryMetrics:
    precision_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    ndcg_at_5: float


def _load_verified_queries(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unverified = [r["id"] for r in rows if not r.get("verified")]
    if unverified:
        raise RuntimeError(
            f"{len(unverified)} eval row(s) are not verified: {unverified}. "
            "Run `python -m creditsense.rag.eval_review` first — headline numbers "
            "are refused until every label has been human-checked (see eval.py "
            "module docstring)."
        )
    return rows


def _metrics_for_query(
    retrieved: list[RetrievedChunk], relevant: set[str]
) -> QueryMetrics:
    if not relevant:
        # Out-of-scope query: "correct" means nothing was retrieved with confidence,
        # which we treat here as: no chunk in the top result. Handled separately by
        # the caller (out_of_scope_no_hit_rate), not folded into precision/recall.
        return QueryMetrics(0.0, 0.0, 0.0, 0.0)

    retrieved_ids = [c.regulation_number for c in retrieved]

    top5 = retrieved_ids[:5]
    precision_at_5 = sum(1 for rid in top5 if rid in relevant) / 5

    top10 = retrieved_ids[:10]
    recall_at_10 = sum(1 for rid in relevant if rid in top10) / len(relevant)

    reciprocal_rank = 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        (1.0 if rid in relevant else 0.0) / math.log2(rank + 1)
        for rank, rid in enumerate(top5, start=1)
    )
    ideal_hits = min(len(relevant), 5)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg_at_5 = dcg / idcg if idcg > 0 else 0.0

    return QueryMetrics(precision_at_5, recall_at_10, reciprocal_rank, ndcg_at_5)


def _retrieve_for_configuration(
    session: Session, query: str, configuration: str
) -> list[RetrievedChunk]:
    if configuration == "keyword_only":
        return bm25_search(session, query, limit=10)
    if configuration == "dense_only":
        return dense_search(session, query, limit=10)
    if configuration == "fused":
        bm25 = bm25_search(session, query, limit=20)
        dense = dense_search(session, query, limit=20)
        return reciprocal_rank_fusion(bm25, dense, limit=10)
    if configuration == "fused_reranked":
        bm25 = bm25_search(session, query, limit=20)
        dense = dense_search(session, query, limit=20)
        fused = reciprocal_rank_fusion(bm25, dense, limit=20)
        return rerank(query, fused, top_k=10)
    raise ValueError(f"Unknown configuration: {configuration}")


def run_eval(
    queries_path: Path = DEFAULT_QUERIES_PATH,
) -> dict[str, dict]:
    """Returns {configuration: {"precision_at_5": ..., "recall_at_10": ..., "mrr": ...,
    "ndcg_at_5": ..., "out_of_scope_no_hit_rate": ...}}."""
    rows = _load_verified_queries(queries_path)
    in_scope_rows = [r for r in rows if r["relevant_regulation_numbers"]]
    out_of_scope_rows = [r for r in rows if not r["relevant_regulation_numbers"]]

    # A confidence floor below which "no confident hit" is declared for an
    # out-of-scope query. RRF scores are small (~1/60 per contributing list), so a
    # low fixed floor is appropriate; tune once real query volume is observed.
    OUT_OF_SCOPE_SCORE_FLOOR = 0.01

    results: dict[str, dict] = {}
    with session_scope() as session:
        for configuration in CONFIGURATIONS:
            per_query: list[QueryMetrics] = []
            for row in in_scope_rows:
                retrieved = _retrieve_for_configuration(
                    session, row["query"], configuration
                )
                per_query.append(
                    _metrics_for_query(
                        retrieved, set(row["relevant_regulation_numbers"])
                    )
                )

            no_hit_count = 0
            for row in out_of_scope_rows:
                retrieved = _retrieve_for_configuration(
                    session, row["query"], configuration
                )
                top_score = retrieved[0].score if retrieved else 0.0
                if top_score < OUT_OF_SCOPE_SCORE_FLOOR:
                    no_hit_count += 1

            n = len(per_query) or 1
            results[configuration] = {
                "precision_at_5": sum(m.precision_at_5 for m in per_query) / n,
                "recall_at_10": sum(m.recall_at_10 for m in per_query) / n,
                "mrr": sum(m.reciprocal_rank for m in per_query) / n,
                "ndcg_at_5": sum(m.ndcg_at_5 for m in per_query) / n,
                "out_of_scope_no_hit_rate": (
                    no_hit_count / len(out_of_scope_rows) if out_of_scope_rows else None
                ),
            }
    return results


def build_report(results: dict[str, dict]) -> str:
    lines = ["# Retrieval Evaluation (WO-P4.7)", ""]
    lines.append(
        "| Configuration | precision@5 | recall@10 | MRR | nDCG@5 | out-of-scope no-hit rate |"
    )
    lines.append("|---|---|---|---|---|---|")
    for configuration in CONFIGURATIONS:
        r = results[configuration]
        no_hit = "n/a" if r["out_of_scope_no_hit_rate"] is None else f"{r['out_of_scope_no_hit_rate']:.2f}"
        lines.append(
            f"| {configuration} | {r['precision_at_5']:.3f} | {r['recall_at_10']:.3f} "
            f"| {r['mrr']:.3f} | {r['ndcg_at_5']:.3f} | {no_hit} |"
        )
    lines.append("")
    lines.append(
        "precision@5/recall@10/MRR/nDCG@5 are computed over the in-scope queries only "
        "(single-clause + multi-hop). out-of-scope no-hit rate is computed separately "
        "over the out-of-scope queries: the fraction where nothing was retrieved above "
        "the confidence floor — the behavior that lets ComplianceAgent say 'insufficient "
        "data, escalate to human' instead of inventing a rule."
    )
    return "\n".join(lines) + "\n"


def main(
    queries_path: Path = DEFAULT_QUERIES_PATH, report_path: Path = DEFAULT_REPORT_PATH
) -> None:
    results = run_eval(queries_path)
    report = build_report(results)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
