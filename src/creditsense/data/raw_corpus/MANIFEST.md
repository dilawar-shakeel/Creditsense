# RAG Corpus Manifest

This corpus backs `ComplianceAgent`'s retrieval. Every file is either **real** SBP
regulation text or **simulated** content written for this project. A demo built on this
corpus must never imply the system is quoting live law from a simulated file.

| File | Provenance | Source |
|---|---|---|
| `sbp_prudential_sme_regulations.md` | **Real** | State Bank of Pakistan, "Prudential Regulations for Small & Medium Enterprise Financing," SME, Housing & Sustainable Finance Department, updated July 16, 2026. Retrieved 2026-09-16 from https://www.sbp.org.pk/assets/documents/circulars/SH_SFD-Annexure-I-Updated_Prudential_Regulations_for_SME_Financing.pdf. Text extracted with `pypdf`; only PDF-artifact whitespace/hyphenation was cleaned — no wording was changed. |
| `internal_credit_policy.md` | **Simulated** | Authored for this project to model a bank/DFI's internal implementation of the real regulations above. Not a real institution's policy. |
| `loan_structuring_templates.md` | **Simulated** | Authored for this project. Not real bank documentation. |

## Real vs. simulated in the retrieval layer

Every chunk stores `source_type` (`sbp_regulation`, `internal_policy`, or `template`).
`ComplianceAgent` must surface this alongside any citation, so "per Regulation R-5" and
"per internal Policy P-28" are never presented as equivalently authoritative.

## Cross-references and the multi-hop test case

The simulated files cite real SBP regulation numbers throughout (e.g. Policy P-1 cites
Regulation R-4, Template T-14 cites Regulation R-5). This is deliberate: a question like
"what's the aggregate exposure rule for group companies?" should correctly retrieve
**both** the internal policy clause (Policy P-29) **and** the SBP regulation it
implements (Regulation R-5) — that pair is the multi-hop retrieval test case used by
`eval_queries.jsonl` (WO-P4.7).

## Finding: the data generator's enterprise-tier turnover breakpoints are stale

Fetching the real regulation surfaced a discrepancy worth recording rather than fixing
silently under a Phase 4 task.

`src/creditsense/data/generators/portfolio.py`'s `SBP_TIER_TURNOVER_BREAKS` currently
reads:

| Tier | Generator (current) | SBP PR (July 16, 2026, Part-I) |
|---|---|---|
| Micro | PKR 0 – 30M | PKR 0 – 30M ✓ matches |
| Small (SE) | PKR 30M – 150M | PKR 30M – **400M** ✗ stale |
| Medium (ME) | PKR 150M – 800M | PKR 400M – **2,000M** ✗ stale |

The generator's per-party exposure ceilings (`SBP_TIER_SECURED_CEILING`: Micro/SE
PKR 100M, ME PKR 500M) and clean-lending cap (`SBP_CLEAN_LENDING_CAP`: PKR 50M) **do
match** Regulation R-5 and R-9 exactly — only the turnover breakpoints used to *assign*
an enterprise its tier are out of date, most likely because they were written against
an earlier circular (the generator's own comment cites 06-Nov-2025; this manifest's
source is dated 16-Jul-2026, a later update).

**Impact:** the synthetic dataset's `sbp_enterprise_tier` column and the per-obligor cap
selection derived from it are computed against the old breakpoints. This does not
invalidate the ML work already done (the model learned whatever tier labels it was
given, consistently), but it means the dataset's tier mix does not reflect the current
regulation. Left as a flagged, not-yet-actioned finding — fixing it means regenerating
the 50,000-row dataset and retraining, which is a Phase 1/3 decision, not a Phase 4 one.

## Corpus size vs. the 200+ chunk target

The tracker's P4.1 target is 200+ clause-level chunks. The real regulation alone is 19
regulations (`R-1`–`R-19`) plus Part-I definitions and 4 annexure sub-sections — roughly
24 chunks; the actual SBP SME regulation is simply not a 200-clause document. Combined
with the simulated internal policy (41 clauses) and structuring templates (23
templates), the corpus as written totals **87 chunks** — verified by
`parse_corpus_dir()`: 23 `sbp_regulation`, 41 `internal_policy`, 23 `template`, zero
undefined cross-references, no chunk over the 800-token ceiling.

This is short of 200+. Recorded here rather than padded to hit the number: every
clause in this corpus is substantive and either real or a genuine simulated
implementation of something real, not filler written to inflate a count. If 200+ is a
hard requirement, the honest way to get there is more of the same — more sector
overlays, more per-tier documentation variants, more restructuring/trade-finance
templates — not shorter, thinner clauses. Flagged for a follow-up pass rather than
done under time pressure in this one.
