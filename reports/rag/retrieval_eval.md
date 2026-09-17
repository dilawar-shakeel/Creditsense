# Retrieval Evaluation (WO-P4.7)

| Configuration | precision@5 | recall@10 | MRR | nDCG@5 | out-of-scope no-hit rate |
|---|---|---|---|---|---|
| keyword_only | 0.250 | 0.903 | 0.690 | 0.755 | 0.00 |
| dense_only | 0.278 | 0.972 | 0.883 | 0.933 | 0.00 |
| fused | 0.261 | 0.972 | 0.865 | 0.888 | 0.00 |
| fused_reranked | 0.267 | 0.972 | 0.943 | 0.953 | 0.80 |

precision@5/recall@10/MRR/nDCG@5 are computed over the in-scope queries only (single-clause + multi-hop). out-of-scope no-hit rate is computed separately over the out-of-scope queries: the fraction where nothing was retrieved above the confidence floor — the behavior that lets ComplianceAgent say 'insufficient data, escalate to human' instead of inventing a rule.
