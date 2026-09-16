# CreditSense — SME Credit Risk & SBP Regulatory Compliance Assistant
### Master Project Context & Progress Tracker

> **This file is the single source of truth for this project.** It is meant to be pasted or uploaded into any AI coding assistant (Claude, ChatGPT, Gemini, Cursor, etc.) at the start of a session so the assistant has full context — what the system is, what's been built, how it was built, and what's still open — without the human needing to re-explain progress verbally every time.

---

## 0. Instructions for the AI Assistant Reading This File

Read this whole file before answering. Specifically:

1. **Treat the status flags in Section 4 as ground truth.** Do not assume a task is incomplete just because it "sounds hard," and do not re-explain or redo anything flagged `[x] DONE` unless the user explicitly asks for a change/refactor.
2. **Read the "Notes" line under each task before helping with it.** That line records the actual implementation choice made (library, pattern, schema decision, gotcha). Stay consistent with it instead of proposing a different approach from scratch — unless the current approach is genuinely broken, in which case flag the concern before changing course.
3. **Section 3 ("Fixed Technical Decisions") is binding.** Don't suggest swapping the agreed stack (e.g., "use OpenAI instead of Claude," "use Chroma instead of pgvector") unless asked to reconsider it.
4. **Work on the next `[ ]` or `[~]` item in the current phase** unless the user directs you elsewhere. Don't jump ahead to later phases out of enthusiasm.
5. **When you finish helping with a task, remind the user to update its status flag and Notes line** — you cannot edit this file yourself in most tool contexts, so hand that back to the user explicitly.
6. If something in this doc looks stale or contradicted by the actual code the user shares, say so — don't silently trust the doc over the ground truth of the repo.

---

## 1. Project Concept (Context, Not a To-Do)

**Title:** CreditSense — Automated SME Underwriting & Prudential Regulation Assistant

**Business problem:** Pakistani fintechs and digital lending arms of banks serving SME clients need to underwrite loans quickly while staying compliant with State Bank of Pakistan (SBP) prudential regulations. Credit officers currently rely on spreadsheet-based scoring and manual cross-checks of loan structuring against SBP circulars — slow, error-prone, and a hard cap on how many SME applications can be processed per week.

**What the system does:** Scores SME credit risk with a gradient-boosted model trained on financial-statement and behavioral features, while a compliance agent checks proposed loan terms against SBP prudential regulations via hybrid RAG retrieval — flagging violations before they reach a human approver.

**Interview pitch (for reference, not implementation):** "We built an underwriting copilot that scores SME credit risk with a gradient-boosted model trained on financial-statement and behavioral features, while a compliance agent checks proposed loan terms against SBP prudential regulations using hybrid RAG retrieval — flagging violations before they reach a human approver. For a digital lender processing 500+ SME applications a month, this cuts credit-officer review time significantly and reduces regulatory exposure — a P&L and risk-committee concern, not just a UX nicety."

### System Flow
Applicant documents/financials uploaded → FastAPI ingestion → `FinancialAnalystAgent` normalizes data → XGBoost model scores default risk → `ComplianceAgent` retrieves relevant SBP clauses for the proposed loan structure → `UnderwritingSupervisorAgent` applies a hard guardrail (approve only if risk score in range **and** zero unresolved compliance flags) → decision + audit trail returned via MCP-exposed tools.

---

## 2. Architecture Reference (Context, Not a To-Do)

### 2.1 Classical ML Layer — XGBoost
- Task: binary classification `default_probability_12m` + regression head `recommended_credit_limit`.
- Features: financial ratios (current ratio, debt-to-equity, revenue growth YoY), sector risk code, years in business, bank statement volatility (std dev of monthly inflows), existing loan exposure, late-payment history count, collateral coverage ratio.
- **Monotonic constraints** on features like debt-to-equity (risk must not decrease as leverage increases) — a real underwriting-model detail, not optional polish.
- **Platt scaling** to calibrate raw XGBoost scores into usable probabilities.

### 2.2 Hybrid RAG Layer
- Corpus: SBP Prudential Regulations for SMEs (public circulars), simulated internal credit policy docs, loan structuring templates.
- Storage: PostgreSQL 16 + `pgvector`, with a `tsvector` column for BM25 alongside the `vector` column for dense embeddings.
- Retrieval: BM25 (`ts_rank_cd`) + dense cosine similarity run in parallel, merged via **Reciprocal Rank Fusion (RRF)**.
- **Chunking by regulation clause number** (e.g., "Regulation R-5: Exposure Limits") — not fixed token windows — so retrieval returns citable, auditable clauses.

### 2.3 AI Agents Layer
- `FinancialAnalystAgent` — extracts/normalizes financial statement data from uploaded documents (must tolerate messy/OCR-style input).
- `RiskScoringAgent` — calls the ML model, translates the score into underwriting narrative.
- `ComplianceAgent` — retrieves relevant SBP regulation clauses for the proposed loan structure and flags breaches (e.g., single-borrower exposure limits). **Must not state a duty/limit without citing a retrieved source chunk** — if it can't find one, it returns "insufficient data, escalate to human."
- `UnderwritingSupervisorAgent` — final decision node. Hard guardrail: approve **only if** risk score is within threshold **and** compliance agent reports zero unresolved flags. This is a hard rule, not a soft LLM suggestion — enforce it in code, not just in a prompt.

### 2.4 Custom MCP Server Layer
Tools to expose:
- `get_applicant_financials(applicant_id)` — scoped read.
- `search_sbp_regulations(topic, regulation_number)` — wraps hybrid RAG retrieval.
- `get_portfolio_exposure(sector)` — used by `ComplianceAgent` to check aggregate sector exposure limits.
- `submit_underwriting_decision(applicant_id, decision, rationale)` — write-scoped, fully audited.

Security requirement: **field-level data masking** in MCP tool responses (e.g., masked account numbers) — a genuine fintech data-security expectation, and a strong interview talking point.

### 2.5 Data Strategy Reference
- ~10,000–20,000 synthetic SME applicant records, with financial-ratio distributions **conditioned on sector** (textile SMEs vs. tech startups vs. trading firms have genuinely different leverage/margin norms — encode the correlation, don't randomize uniformly).
- Injected messiness: ~10% of records simulate OCR-extracted bank statements with digit-transposition noise and inconsistent currency formatting (PKR / Rs. / numeric-only); ~5% missing collateral fields.
- RAG corpus: 200+ chunked clauses from real/simulated SBP-style prudential regulation text, with intentional cross-references between clauses (e.g., Regulation X references Regulation Y) to test multi-hop retrieval.
- Multilingual noise: credit-officer notes with Urdu-English code-switching (e.g., "client ka cash flow seasonal hai, Eid se pehle spike hota hai") to stress-test `FinancialAnalystAgent`'s text parsing.

---

## 3. Fixed Technical Decisions (Binding — Confirm Before Deviating)

> Fill these in as you decide. Once filled, treat as locked unless you explicitly revisit them — this prevents different AI sessions from nudging you toward a different stack each time.

| Decision | Choice | Date Locked | Notes |
|---|---|---|---|
| LLM provider (agents) | OpenAI SDK | 2026-09-16 | Used today for embeddings and the reranker; Phase 5 agents will use the same provider. |
| Agent framework | LangChain | 2026-09-16 | Chosen over LangGraph — Phase 5 not built yet, but this is locked so nobody proposes LangGraph again. |
| Embedding model | `text-embedding-3-small` (1536-d) | 2026-09-16 | Matches the `Vector(1536)` column already in the database — no extra migration needed. In real use in the RAG pipeline. |
| Reranker | LLM call via OpenAI (`gpt-4o-mini`, configurable) | 2026-09-16 | Chosen over `bge-reranker-base` to avoid installing `torch`/`sentence-transformers` — this machine already lost time to a Windows security policy blocking a much smaller ML package's file, so a 2-3GB native dependency was judged not worth the risk. See Decision Log. |
| Database hosting | Local Docker Postgres (`pgvector/pgvector:pg16`) | 2026-09-16 | Confirmed working via `docker-compose.yml`. |
| Deployment target | _TBD_ | | Local demo / cloud VM / other |
| Repo structure convention | Matches Section 6's proposed layout | 2026-09-16 | `data/generators/`, `rag/`, `db/` are now real, populated folders, not placeholders. |
| Package/dependency manager | `uv` | 2026-09-14 | Existing `pyproject.toml` and `uv.lock`; the project also keeps `requirements.txt` for container/CI installs. |
| Testing framework | `pytest` | | From tech stack baseline |
| Auth approach for API/MCP | _TBD_ | | API key vs. JWT vs. other |

---

## 4. Implementation Task Tracker

**Status legend** — use these exact tags so any AI tool parses them consistently:
- `[ ] TODO` — not started
- `[~] IN_PROGRESS` — started, not complete
- `[x] DONE` — complete and working
- `[!] BLOCKED` — stuck; reason must go in Notes
- `[s] SKIPPED` — deliberately descoped; reason must go in Notes

Each task has a **Notes** line — record the actual library/pattern used, key decisions, or gotchas once you touch it. Leave placeholders (`—`) until then.

### Phase 0 — Project Scaffolding & Environment
- [x] DONE **P0.1** — Initialize repo structure (see Section 6 proposed layout)
  - Notes: `src/creditsense` package layout created with `api`, `agents`, `data`, `db`, `ml`, `rag`, and `mcp_server` areas; empty future folders are retained with `.gitkeep` files.
- [x] DONE **P0.2** — Set up `.env` + config management, `.env.example` committed
  - Notes: `pydantic-settings` `Settings` class reads `.env`; `.env.example` documents local and Docker values; `.env` is ignored. **2026-09-16 fix:** the example file used to default to the Docker-only database hostname, so running the app directly on your own machine failed with no obvious reason — it now defaults to `localhost` and shows the Docker value as a comment. `OPENAI_API_KEY` was also added, needed from Phase 4 onward.
- [x] DONE **P0.3** — Docker Compose: Postgres (+ pgvector extension), app service, agent service
  - Notes: `docker-compose.yml` uses `pgvector/pgvector:pg16`, a persistent volume, database health checks, and separate `app` and `agent` services built from `Dockerfile`. **2026-09-16 fix:** the `agent` service ran a 10-line placeholder script that printed something and immediately exited, so it showed up as a crashed container on every `docker compose up` — it's commented out until Phase 5 actually builds an agent worth running as a service. The `app` service now also mounts the trained model folder from your machine instead of expecting it baked into the image (the model file is intentionally not committed to git).
- [x] DONE **P0.4** — Base FastAPI app skeleton (health check endpoint, Pydantic v2 settings)
  - Notes: `creditsense.api.main` exposes `GET /health`; settings use Pydantic v2 `BaseSettings`; `tests/test_health.py` verifies the endpoint.
- [x] DONE **P0.5** — CI scaffold (GitHub Actions: lint + test on push)
  - Notes: `.github/workflows/ci.yml` installs `requirements.txt`, runs Ruff, and runs pytest on pushes and pull requests. **2026-09-16 fix:** CI was installing with plain `pip` from `requirements.txt`, while the project's real dependency manager is `uv` with a lockfile — CI could pass on a dependency set `uv` would never actually produce. Switched CI to `uv sync` + `uv run`.

### Phase 1 — Synthetic Data Generation
- [x] DONE **P1.1** — Design applicant schema (fields, types, sector categories)
  - Notes: 42-column schema is documented in `src/creditsense/data/raw/creditsense_synthetic_portfolio.metadata.json`; the dataset contains 11 sector categories and typed numeric, categorical, and binary fields.
- [x] DONE **P1.2** — Build sector-conditioned generator (financial ratios correlate realistically per sector, not uniform random)
  - Notes: Deterministic generator uses sector-conditioned risk, energy, and foreign-import patterns; the dedicated suite passed all 74 distribution and economic-realism tests.
- [s] SKIPPED **P1.3** — Inject messiness: OCR-style digit noise (~10%), inconsistent currency formatting, missing collateral fields (~5%)
  - Notes: Deliberately descoped by project decision; the main 50,000-row CSV keeps clean numeric fields and does not simulate OCR/currency-format noise or missing collateral values. **2026-09-16:** a small separate 60-record file was added instead — see P1.8 below — so this messiness still exists somewhere in the project and can be used to test the upcoming `FinancialAnalystAgent`.
- [s] SKIPPED **P1.4** — Add Urdu-English code-switched free-text notes field
  - Notes: Deliberately descoped by project decision; no free-text notes field is included in the main dataset. **2026-09-16:** covered instead by the same small file added for P1.3 — see P1.8.
- [x] DONE **P1.5** — Generate final dataset (target: 10,000–20,000 rows), save with schema/version metadata
  - Notes: 50,000-row CSV saved at `src/creditsense/data/raw/creditsense_synthetic_portfolio.csv`, intentionally above the original target; schema, version, seed, validation results, and scope decisions saved in `src/creditsense/data/raw/creditsense_synthetic_portfolio.metadata.json`. **2026-09-16:** the generator moved from `tests/synthetic_data/creditsense_generator.py` to `src/creditsense/data/generators/portfolio.py` (run with `python -m creditsense.data.generators.portfolio`) so that regenerating real data doesn't mean importing code out of the test folder. Every row now also gets a unique `applicant_id` (e.g. `SME-000001`) — the database needs this to know which applicant a row belongs to, and there was previously no way to tell rows apart.
- [x] DONE **P1.6** — Sanity-check distributions (per-sector plots/summary stats) to confirm realism
  - Notes: `python -m creditsense.data.generators.profile` writes `src/creditsense/data/raw/data_profile.md` — a plain-English table of default rate and key ratio quartiles for each of the 11 sectors, readable without running any code. Backed by the 74 automated checks already in `tests/synthetic_data/`.
- [x] DONE **P1.7** (new, 2026-09-16) — Give every applicant row a unique ID
  - Notes: The 50,000-row dataset had no column that identified a single applicant, which meant the database (which needs a unique ID per applicant) had nothing to load. Fixed as part of P1.5 above — see that entry.
- [x] DONE **P1.8** (new, 2026-09-16) — Small messy-data file to replace what P1.3/P1.4 skipped
  - Notes: `tests/fixtures/messy_applications.json` — 60 sample applications built from real rows with realistic problems mixed in on purpose: numbers with two digits accidentally swapped (like a bad scan would produce), amounts written three different ways ("Rs. 1,234,567" / "PKR 1234567" / plain numbers), a few applications missing their collateral info, and a handful of notes written in mixed Urdu-English (e.g. "client ka cash flow seasonal hai, Eid se pehle spike hota hai"). Regenerate with `python tests/fixtures/build_messy_applications.py`. Explained further in `tests/fixtures/README.md`.

### Phase 2 — Database & Infra
- [x] DONE **P2.1** — Postgres schema: applicants table, documents table (RAG corpus), chunks table (`tsvector` + `vector` columns), audit_log table
  - Notes: All four tables exist via the two migrations under `src/creditsense/db/migrations/versions/`. **2026-09-16:** the schema had three real problems that would have blocked Phase 4 entirely, fixed in a second migration (`20260916_0002_rag_corpus_schema.py`): (1) a regulation/policy document couldn't be saved at all, because `documents` required an applicant and a regulation doesn't have one — fixed by allowing that field to be empty; (2) the keyword-search column existed but nothing ever filled it in, so keyword search would have silently returned nothing forever — it's now a column Postgres fills in automatically and can never forget to update; (3) a chunk of regulation text had no way to record which regulation number, title, or related regulations it belonged to — added, and needed for the citation feature. There was also no code anywhere that could actually open a database connection — added `src/creditsense/db/session.py`.
- [x] DONE **P2.2** — SQLAlchemy 2.0 models + migrations (Alembic or equivalent)
  - Notes: SQLAlchemy models are defined in `src/creditsense/db/models.py`; Alembic manages the initial revision in `src/creditsense/db/migrations/versions/20260915_0001_initial_schema.py`, including the pgvector extension and the four Phase 2 tables. See P2.1 for the follow-up migration.
- [x] DONE **P2.4** — SQL views for portfolio exposure aggregation (sector-level exposure)
  - Notes: `src/creditsense/db/migrations/versions/20260916_0003_portfolio_exposure_view.py` adds `portfolio_sector_exposure` (sector, applicant count, total exposure, % of book) — what `get_portfolio_exposure(sector)` (P6.4) and the 25% concentration check in policy P-1 will read from. Verified against the real 50k-row portfolio: percentages sum to 100.01% (rounding), largest sector is Retail/Trade at 21.80% — no sector currently breaches the 25% cap.
- [x] DONE **P2.3** — Seed script to load synthetic applicant data into Postgres
  - Notes: `src/creditsense/db/seed.py`; upserts on `applicant_id` in batches of 1000 (safe to re-run), `--limit N` for a fast dev subset. Verified: all 50,000 applicants loaded (`SELECT count(*) FROM applicants` = 50000).

### Phase 3 — Classical ML Layer (XGBoost)
- [x] DONE **P3.1** — EDA + feature engineering pipeline
  - Notes: `Stage1Preprocessor` implements training-only thresholds, financial stress count, score-adjusted risk, one-hot documentation tiers, and native categorical sector handling in `src/creditsense/ml/features.py`.
- [x] DONE **P3.2** — Train/test split, baseline XGBoost model
  - Notes: `train_bundle()` and `tune_stage1()` use a stratified 80/20 split and XGBoost classification plus the Stage 2 credit-limit regressor; Stage 2 now trains on calibrated out-of-fold Stage 1 probabilities to match serving behavior.
- [x] DONE **P3.3** — Apply monotonic constraints (e.g., debt-to-equity → risk direction)
  - Notes: Business directions are declared in `MONOTONE_DIRECTIONS` and converted to column-aligned XGBoost constraint tuples for both stages.
- [x] DONE **P3.4** — Platt scaling for probability calibration
  - Notes: `fit_stage1_calibrated()` trains the tree model on one training slice and sigmoid-calibrates it on a separate training-only slice using `CalibratedClassifierCV`.
- [x] DONE **P3.5** — Cross-validation + feature importance / SHAP explainability
  - Notes: Stage 1 tuning uses stratified cross-validation and PR-AUC; `explain_decision()` provides verified SHAP reason codes; `tests/test_ml_pipeline.py` covers the explanation output.
- [x] DONE **P3.6** — Model versioning (MLflow or pickle + metadata JSON)
  - Notes: `CreditSenseBundle` saves the fitted preprocessor, calibrated classifier, Stage 2 encoder/model, and training metadata with joblib plus companion JSON files; `creditsense_bundle.metrics.json` records classification, calibration, confusion-matrix, and regression metrics; the training entry point is `creditsense.ml.train.main()`.
- [x] DONE **P3.7** — `/predict/credit-risk` FastAPI endpoint (Pydantic-validated request/response, SHAP output included)
  - Notes: `creditsense.api.predict` validates all required Stage 1 and Stage 2 inputs, lazily loads the configured joblib bundle, returns calibrated risk, decision cutoff, recommendation, and SHAP explanation, and returns HTTP 503 when no trained bundle is available; covered by `tests/test_prediction_endpoint.py`.
- [x] DONE **P3.8** — Unit tests for scoring pipeline
  - Notes: `tests/test_ml_pipeline.py` covers preprocessing, unseen categories, monotonic directions, Stage 2 feature construction, KS cutoff selection, and SHAP reason-code output.
- [x] DONE **P3.9** (stretch) — Bias/fairness sanity check across sectors
  - Notes: `python -m creditsense.ml.fairness_check` writes `src/creditsense/ml/artifacts/fairness_by_sector.md`. Result: approval rate is 90.10% portfolio-wide, and every one of the 11 sectors sits within about 3 percentage points of that — nothing flagged as an outlier (the check flags anything more than 15 points off).
- [s] SKIPPED **P3.10** (stretch) — Drift monitoring checks
  - Notes: Deliberately left for later — there's no live production traffic yet to monitor drift against, so a drift check today would have nothing real to measure.
- [x] DONE **P3.11** (new, 2026-09-16) — Fix how the model's decline cutoff is chosen, and be honest about how good the model actually is
  - Notes: Two real problems, both fixed and retrained (`src/creditsense/ml/train.py`):
    1. **The cutoff was being picked using the same data it was then graded on** — like a student grading their own exam after choosing which questions count. Fixed by splitting the data three ways instead of two: one part to train on, a separate part to pick the cutoff, and a third part — untouched by either — to report the final numbers on.
    2. **The old cutoff declined 26.7% of applicants when only 10.2% actually default**, because it optimized for a statistical property instead of a business one. Replaced with a plain rule: decline roughly as often as applicants actually default. After retraining, the model now declines **10.25%** — a big improvement, with no invented "cost of a bad loan" number pretending to be more precise than it is. The full trade-off table (every cutoff option, not just the one chosen) is saved in `creditsense_bundle.metrics.json` so the choice can be revisited.
    3. Also recorded, for context: the model's accuracy (ROC-AUC 0.632) is close to the best any model could do on this data (0.646) — the dataset's labels have deliberate randomness baked in by design, so this was never going to reach a much higher number, and that's now written down in the metrics file instead of looking like an unexplained weak model.

### Phase 4 — RAG Corpus & Hybrid Retrieval
- [x] DONE **P4.1** — Source/simulate SBP prudential regulation text (target 200+ chunks) with intentional cross-references between clauses
  - Notes: Three files in `src/creditsense/data/raw_corpus/`, explained in that folder's `MANIFEST.md`: (1) **the real State Bank of Pakistan SME regulation document** (all 19 regulations plus its two annexures, fetched from sbp.org.pk, updated July 16, 2026 — this is genuine current regulation text, not invented); (2) a written internal credit policy document (41 clauses) that implements those regulations and cites them by number; (3) a written set of loan structuring templates (23 templates) that reference both. Deliberately labeled real vs. simulated everywhere so a demo never implies the system is quoting live law from a made-up file. **Honest shortfall:** the real regulation is only 19 clauses — nowhere near 200 on its own — so the combined total today is **87 chunks, not 200+**. Every one of those 87 is real, substantive content, not padding written just to inflate the count; reaching 200+ properly means writing more of the same (more sector policies, more document checklists), which is flagged as follow-up work rather than done under time pressure. **Also found while sourcing this:** the synthetic dataset's enterprise-size cutoffs (Small Enterprise capped at 150 million, Medium at 800 million) are out of date — the real regulation as of July 2026 says Small goes up to 400 million and Medium up to 2,000 million. Not yet fixed (that's a Phase 1 change); written down in the manifest so it isn't lost.
- [x] DONE **P4.2** — Chunk by clause number (not fixed token windows)
  - Notes: `src/creditsense/rag/chunking.py`. Splits strictly on regulation/policy/template headings, never on a fixed word count — every piece of text retrieval can return is a whole, citable clause with its own number. 20 automated tests confirm: no clause is orphaned from its number, no clause is too long, and every "see Regulation R-x" reference in the text actually points to a regulation that exists.
- [x] DONE **P4.3** — Embedding generation pipeline, populate `vector` column
  - Notes: `src/creditsense/rag/embed.py` + `src/creditsense/rag/ingest.py`. Uses OpenAI's `text-embedding-3-small` model. Saves a local cache file so re-running during development doesn't re-pay or re-wait for text that hasn't changed. Run with `python -m creditsense.rag.ingest` (add `--dry-run` to check counts without calling the API or touching the database — that's what CI uses). **Confirmed run:** all 87 chunks have been embedded successfully.
- [x] DONE **P4.4** — BM25 setup via `tsvector` / `ts_rank_cd`
  - Notes: Keyword search lives in `src/creditsense/rag/retrieval.py`, backed by the self-filling `content_tsv` column added in P2.1's migration (title and regulation number are weighted above the body text, so a title match ranks higher than a passing mention). **2026-09-16 bug found and fixed:** the first working version required every single word in a question to be found somewhere in the corpus before it would return anything at all — since a real question has 8-12 words and the corpus is small, this meant keyword search returned **zero results for all 41 test questions**, every time, with no error or warning. Switched to matching on "enough of the words," the same way a normal search engine works, and confirmed with real (non-zero) results afterward — see P4.7.
- [x] DONE **P4.5** — RRF fusion logic combining BM25 + dense results
  - Notes: Same file. Combines the keyword-search list and the meaning-search list into one ranked list using the standard Reciprocal Rank Fusion formula. Keyword search and meaning search run at the same time (not one after another) for speed. 7 automated tests check the math against numbers worked out by hand.
- [x] DONE **P4.6** — Reranker integration on top-N before final top-5
  - Notes: `src/creditsense/rag/rerank.py`. **Changed from the blueprint's suggestion** (`bge-reranker-base`, a model you'd run locally) to a single OpenAI call that re-scores all 20 candidates at once — see the Section 3 table and Decision Log for why. If the reranker call fails for any reason, retrieval still returns the fused results rather than breaking — checked by an automated test.
- [x] DONE **P4.7** — Retrieval evaluation harness (precision@5, recall@10) — including multi-hop clause cross-reference cases
  - Notes: `src/creditsense/rag/eval.py` + `src/creditsense/rag/eval_review.py`. 41 test questions (25 single-clause, 11 genuinely needing two connected clauses, 5 intentionally unrelated to test that the system correctly says "I don't know"), all 41 hand-checked before any score was allowed to count. Real numbers, saved in `reports/rag/retrieval_eval.md`:

    | Method | precision@5 | recall@10 | MRR | nDCG@5 | correctly says "I don't know" |
    |---|---|---|---|---|---|
    | Keyword search alone | 0.250 | 0.903 | 0.690 | 0.755 | 0/5 |
    | Meaning search alone | 0.278 | 0.972 | 0.883 | 0.933 | 0/5 |
    | Combined | 0.261 | 0.972 | 0.865 | 0.888 | 0/5 |
    | Combined + reranked | 0.267 | 0.972 | 0.943 | 0.953 | 4/5 |

    A precision@5 score of "only" ~0.25-0.28 sounds low but isn't: most questions have just one right answer among the top 5 results shown, so the best any system could ever score here is about 0.272 — meaning-search alone is already basically at that ceiling. The reranked combination gets the best overall ranking quality (MRR/nDCG@5) and is the only one of the four that can tell an out-of-scope question isn't answerable (4 of 5, not 0 of 5) — worth its cost. One out-of-scope question still slips through after reranking; noted below as a follow-up, not a blocker. Also found and fixed along the way: keyword search was returning nothing at all for every single question until 2026-09-16 — see P4.4.

### Phase 5 — AI Agents Layer
- [ ] TODO **P5.1** — Choose/confirm framework (LangGraph vs CrewAI) — record in Section 3
  - Notes: —
- [ ] TODO **P5.2** — `FinancialAnalystAgent`: parse/normalize uploaded financials, tolerate messy input, parse Urdu-English notes
  - Notes: —
- [ ] TODO **P5.3** — `RiskScoringAgent`: call ML endpoint, produce plain-English risk narrative
  - Notes: —
- [ ] TODO **P5.4** — `ComplianceAgent`: retrieve SBP clauses for proposed loan structure, flag breaches, enforce "no claim without a cited source chunk" rule
  - Notes: —
- [ ] TODO **P5.5** — `UnderwritingSupervisorAgent`: hard-coded guardrail (approve only if risk-in-range AND zero unresolved compliance flags)
  - Notes: —
- [ ] TODO **P5.6** — Conditional routing / escalation-to-human path when confidence is low
  - Notes: —
- [ ] TODO **P5.7** — Structured outputs via Pydantic schemas for all agent responses
  - Notes: —

### Phase 6 — Custom MCP Server
- [ ] TODO **P6.1** — MCP server scaffold (official Python MCP SDK)
  - Notes: —
- [ ] TODO **P6.2** — Tool: `get_applicant_financials(applicant_id)` — scoped read
  - Notes: —
- [ ] TODO **P6.3** — Tool: `search_sbp_regulations(topic, regulation_number)` — wraps hybrid RAG
  - Notes: —
- [ ] TODO **P6.4** — Tool: `get_portfolio_exposure(sector)`
  - Notes: —
- [ ] TODO **P6.5** — Tool: `submit_underwriting_decision(applicant_id, decision, rationale)` — write-scoped, audited
  - Notes: —
- [ ] TODO **P6.6** — Field-level data masking middleware (e.g., account numbers)
  - Notes: —
- [ ] TODO **P6.7** — API-key/scope-based auth middleware for MCP tool access
  - Notes: —
- [ ] TODO **P6.8** — Audit log table wired to every tool call (identity, timestamp, payload hash)
  - Notes: —

### Phase 7 — Backend Integration
- [ ] TODO **P7.1** — Wire ingestion endpoint → agents → ML endpoint → RAG → supervisor decision
  - Notes: —
- [ ] TODO **P7.2** — Error handling: ML service timeouts, malformed LLM outputs, retry logic
  - Notes: —
- [ ] TODO **P7.3** — Structured JSON logging (`structlog` or equivalent)
  - Notes: —
- [ ] TODO **P7.4** — Rate-limiting on write-tools
  - Notes: —

### Phase 8 — Testing & Evaluation
- [ ] TODO **P8.1** — Unit tests across ML, RAG, agents, MCP tools
  - Notes: —
- [ ] TODO **P8.2** — End-to-end integration test (full applicant → decision flow)
  - Notes: —
- [ ] TODO **P8.3** — Guardrail test: confirm supervisor cannot approve with unresolved compliance flags, even under adversarial prompt input
  - Notes: —
- [ ] TODO **P8.4** — Retrieval evaluation numbers recorded (precision@5, recall@10) — not just anecdotal demo success
  - Notes: —

### Phase 9 — Demo, Docs & Interview Readiness
- [ ] TODO **P9.1** — Minimal frontend or Postman collection for live demo
  - Notes: —
- [ ] TODO **P9.2** — Demo script / README walkthrough
  - Notes: —
- [ ] TODO **P9.3** — Decision audit trail dashboard (stretch)
  - Notes: —
- [ ] TODO **P9.4** — Prepare 90-second answers to the 4 interview questions in Section 7
  - Notes: —

---

## 5. Decision Log

> Append-only. One row per non-trivial decision made during the build — especially anywhere you deviated from the blueprint in Section 2.

| Date | Decision | Rationale | Alternatives Considered |
|---|---|---|---|
| _e.g. 2026-08-27_ | _e.g. Used LangGraph over CrewAI_ | _e.g. Better conditional-edge support for hard guardrail node_ | _e.g. CrewAI_ |
| 2026-09-16 | Agent framework: LangChain, not LangGraph | Project-level decision; the supervisor's hard approve/decline rule is enforced in plain Python code either way, so this doesn't weaken that guardrail | LangGraph, CrewAI |
| 2026-09-16 | Reranker: one OpenAI call scoring all candidates, not a locally-run `bge-reranker-base` model | Avoids installing `torch`/`sentence-transformers` (2-3GB). This machine had already lost real time to a Windows security feature (Smart App Control) blocking a much smaller ML package's file, purely for being an unfamiliar compiled file — a much bigger native dependency was judged not worth repeating that risk for | `bge-reranker-base` (blueprint's original suggestion) |
| 2026-09-16 | A regulation/policy document is allowed to have no applicant attached, instead of inventing a fake "SYSTEM" applicant to satisfy the database | Simpler and more honest — a regulation genuinely doesn't belong to any applicant | Creating a placeholder applicant row |
| 2026-09-16 | RAG corpus built as a mix of real SBP regulation text + written internal policy/templates, reaching 87 chunks rather than the 200+ target | The real regulation document is only 19 clauses — physically not a 200-clause document. Chose real, honest content over padding to hit a number | Writing 200+ fully invented clauses to hit the target exactly |
| 2026-09-16 | Decline-rate cutoff for the default model changed from "maximize statistical separation" (chosen on the same data it was graded on) to "decline about as often as applicants actually default" (chosen on a separate, untouched slice of data) | Old cutoff declined 26.7% of applicants when only 10.2% actually default, and the old grading method flattered its own result | A cost-based cutoff (rejected — would have required guessing a peso/rupee cost for a bad loan vs. a lost good customer, which nobody had a real number for) |

---

## 6. Proposed Repo Structure

> Edit to match reality once scaffolding (P0.1) is done — keep this in sync so any AI tool can navigate the repo correctly.

```
creditsense/
├── data/
│   ├── generators/          # synthetic data generation scripts
│   └── raw_corpus/          # SBP regulation source text before chunking
├── db/
│   ├── models.py            # SQLAlchemy models
│   └── migrations/
├── ml/
│   ├── train.py
│   ├── features.py
│   └── artifacts/           # saved model + metadata
├── rag/
│   ├── chunking.py
│   ├── embed.py
│   ├── retrieval.py         # BM25 + dense + RRF + rerank
│   └── eval.py
├── agents/
│   ├── financial_analyst.py
│   ├── risk_scoring.py
│   ├── compliance.py
│   └── supervisor.py
├── mcp_server/
│   ├── server.py
│   ├── tools.py
│   └── auth_middleware.py
├── api/
│   └── main.py               # FastAPI app
├── tests/
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 7. Interview Readiness Checklist (Context Reference)

Prepare a concrete, 90-second answer for each — tied to what was actually built, not the blueprint text:

1. What guardrail did you build that *prevents* a wrong agent action, with a concrete example? (Reference: `UnderwritingSupervisorAgent` hard-approval rule.)
2. What are your retrieval evaluation numbers (precision@k), not just "it works when I demo it"?
3. What real data-quality problem did you simulate, and how did your pipeline handle it? (Reference: OCR noise, currency formatting, Urdu-English code-switching.)
4. What are the system's failure modes — what happens when the ML service times out, or the LLM hallucinates a compliance clause?

---

## 8. Open Questions / Risks

> Running list — add as they come up, remove/resolve as answered.

- **The RAG corpus is 87 clauses, not the 200+ target** (P4.1). Real content, not padding — but short. Decide whether to write more real internal-policy content before the demo, or accept 87 as the working number.
- **The synthetic dataset's enterprise-size cutoffs are out of date** against the current real SBP regulation (Small Enterprise: dataset caps it at PKR 150M, real regulation says PKR 400M; Medium: dataset says PKR 800M, real regulation says PKR 2,000M). Not yet fixed — would mean regenerating the 50,000-row dataset and retraining. Details in `src/creditsense/data/raw_corpus/MANIFEST.md`.
- **One out-of-scope question still gets a confident wrong answer after reranking** (4 of 5 correctly rejected, not 5 of 5 — see P4.7's table). Worth identifying which question and why before relying on this in a live demo.
- **Combining keyword + meaning search (plain fusion) is slightly worse than meaning search alone** on ranking quality (see P4.7). Not wrong, just an honest result — equal-weight fusion let keyword's noisier results dilute a stronger signal. Worth knowing before claiming "hybrid search" is unconditionally better in an interview answer.
- This machine's Windows Smart App Control setting was blocking `scikit-learn` from running at all (a legitimate ML package's file was being treated as untrusted). Fixed by turning that setting off. Worth remembering if this project ever moves to a different machine.

---

## 9. Changelog of This Tracker File

| Date | Change |
|---|---|
| 2026-08-27 | Initial file created from AI_Capstone_Blueprints_2026.md, Blueprint 2 (CreditSense) |
| 2026-09-14 | Converted the notebook's Claude ML design into reusable preprocessing, training, calibration, SHAP, Stage 2 regression, bundle save/load, and scoring modules; added ML tests and declared SHAP dependency. |
| 2026-09-14 | Installed SHAP in the project environment, verified applicant reason-code generation, and added a passing SHAP regression test. |
| 2026-09-15 | Added the P3.7 FastAPI credit-risk prediction endpoint with Pydantic validation, lazy model loading, SHAP output, missing-model handling, and endpoint tests. |
| 2026-09-15 | Trained the two-stage ML bundle on all 50,000 synthetic applicants and verified a real `/predict/credit-risk` request returned risk, decision, credit limit, and SHAP reasons. |
| 2026-09-15 | Updated ML training to use separate calibrated out-of-fold Stage 1 probabilities for Stage 2 and added a reproducible training entry point with saved metrics. |
| 2026-09-15 | Expanded the saved training report with ROC-AUC, Brier score, class precision/recall/F1, confusion matrix, RMSE, median absolute error, and prediction bias; retrained the full 50,000-row bundle. |
| 2026-09-16 | Audited Phases 0/1/3 and fixed real problems the tracker didn't mention: no way to tell dataset rows apart (added applicant IDs), the decline cutoff was graded on the same data it was chosen from, config/Docker files pointed at the wrong things. Retrained the model with the fixed cutoff and confirmed the fairness check across sectors. |
| 2026-09-16 | Built Phase 4 (RAG): fixed three database problems blocking it (see P2.1); fetched and used the real, current SBP SME regulation text; wrote internal policy and template content citing it; built clause-based chunking, embedding, keyword search, meaning search, result fusion, and an OpenAI-based reranker; wrote a 41-question evaluation set pending manual verification. Corpus reached 87 of the 200+ target chunk count — recorded honestly rather than padded. |
| 2026-09-16 | Finished Phase 4: all 41 evaluation questions hand-checked; found and fixed a real bug where keyword search returned nothing for every single question (it required every word in a question to match, not just enough of them); re-ran the evaluation and got real numbers — meaning-search alone is close to the best any method could score, and adding the reranker gets the best overall ranking quality and is the only method that can say "I don't know" to an out-of-scope question (4 of 5 times). P4.7 marked done with numbers recorded. |
