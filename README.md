# CreditSense

**An SME underwriting copilot that scores credit risk with a gradient-boosted model and checks the proposed loan structure against State Bank of Pakistan prudential regulations — refusing to approve when either one says no.**

A credit officer opens a work queue, picks an application, and watches four agents work through it live: a messy document becomes clean data, an XGBoost model scores default risk, a compliance agent retrieves the relevant SBP clauses, and a supervisor applies a hard approval guardrail. Every number traces to where it came from, every compliance claim opens to the actual clause behind it, and the screen that matters most is the one where the system **refuses to decide** and hands the case to a human.

Everything runs locally on Docker Compose. The data is synthetic; the SBP regulation text is real (and the parts that aren't are labelled as such, in the UI, on purpose).

---

## What's actually in here

| Layer | What it does | Where |
|---|---|---|
| **ML** | Two-stage XGBoost: calibrated default probability + recommended credit limit. Monotonic constraints, Platt scaling, SHAP reason codes. | `src/creditsense/ml/` |
| **Hybrid RAG** | 87 clause-level chunks in Postgres/pgvector. BM25 (`ts_rank_cd`) + dense cosine run concurrently, fused with RRF, then LLM-reranked. | `src/creditsense/rag/` |
| **Agents** | Financial analyst → risk scoring → compliance → supervisor. Plain Python, no agent framework. | `src/creditsense/agents/` |
| **MCP server** | Four tools over stdio + streamable HTTP, with field masking, scoped auth, and per-call audit logging. | `src/creditsense/mcp_server/` |
| **API** | `/predict/credit-risk`, `/applications/underwrite` (+ SSE trace), applicant/decision/regulation/portfolio reads, PDF intake. | `src/creditsense/api/` |
| **Frontend** | Six screens, vanilla JS + Tailwind CDN, served by the same FastAPI app. No build step. | `src/creditsense/frontend/` |

<<<<<<< HEAD
=======
**348 offline tests** (plus 7 marker-gated live tests), Ruff-clean, coverage enforced in CI.
>>>>>>> origin/phase-0-foundation

---

## Quick start

```powershell
docker compose up -d --build app        # brings up Postgres (pgvector) + the API
docker compose exec app alembic upgrade head
docker compose exec app python -m creditsense.db.seed --cohort tests/fixtures/messy_applications.json
```

Open **http://localhost:8000/** → redirects to the UI.

`--cohort` makes the applicants table *be* exactly the 60 messy-fixture applicants — each one has a document with real parsing problems in it, which is what makes the demo interesting. Drop the flag to seed all 50,000 instead (slower, and most rows are clean).

### First run on a clean clone

The dataset, the trained model, and the embedded corpus are all git-ignored — regeneratable, and either large or environment-specific:

```powershell
$env:PYTHONPATH = "src"
python -m creditsense.data.generators.portfolio   # 50,000-row CSV + metadata
python -m creditsense.ml.train                    # trains and saves the model bundle
alembic upgrade head
<<<<<<< HEAD
=======
python -m creditsense.rag.ingest --dry-run        # parses the corpus, no API calls (this is what CI runs)
>>>>>>> origin/phase-0-foundation
python -m creditsense.rag.ingest                  # embeds and stores all 87 chunks (needs OPENAI_API_KEY)
python -m creditsense.db.seed --cohort tests/fixtures/messy_applications.json
```

Copy `.env.example` to `.env` first. `OPENAI_API_KEY` is required for anything that embeds, reranks, or runs an agent.

---

## Demo script (~6 minutes)

**1. The queue, not a dashboard** — `/ui/`. Three tabs; "Needs You" holds the escalations. The product framing is that this is the queue where the system's *deliberate refusals* get resolved by a person.

<<<<<<< HEAD
**2. Mess becoming clean** — open an applicant with `urdu_english_notes`. Left column shows the document *as received* (`"Rs. 2,130,819"`, a null collateral field, the Urdu-English note verbatim) and *as understood* — each field tagged `From document` / `Bank record` / `Derived` / `Unresolved`.
=======
**2. Mess becoming clean** — open an applicant with `ocr_digit_transposition` or `urdu_english_notes` noise. Left column shows the document *as received* (`"Rs. 2,130,819"`, a null collateral field, the Urdu-English note verbatim) and *as understood* — each field tagged `From document` / `Bank record` / `Derived` / `Unresolved`.
>>>>>>> origin/phase-0-foundation
> The line that lands: **the numbers were never touched by an LLM.** Currency parsing is rules-only, deliberately, because a model silently changing a digit in a turnover figure is undetectable downstream. The LLM only ever sees the free-text note.

**3. The agents running live** — the centre column streams real `stage.completed` events over SSE with real elapsed times. Compliance is visibly the slowest stage (retrieval + rerank + an LLM call) — that's worth showing, not hiding.

**4. The guardrail** — request an amount far over the R-5 per-party ceiling. Verdict is `DECLINE`, citing the real clause by number. Then the point: `supervisor.decide()` reads `severity`, never `summary`. There's no LLM in that function, so there's no path for model prose to reach the decision — and `submit_underwriting_decision` over MCP re-verifies and **refuses** an APPROVE over an open breach, logging the refused attempt as evidence.

**5. Saying "I don't know"** — Regulation Explorer, ask *"What is SBP's current monetary policy rate?"* Out of corpus scope → nothing above the confidence floor. Volunteer the real number: this works on **4 of 5** out-of-scope test questions, and that's recorded in `reports/rag/retrieval_eval.md`.

**6. Audit trail** — `/ui/audit.html`. Every automated decision, every human accept/override, each expandable to its full payload. Accepts and overrides *append*; they never mutate the original decision row.

**Optional:** New Applicant → download the fillable PDF template, fill it, upload it back. Fields read deterministically by name (no OCR guessing); anything unreadable comes back flagged **Unresolved** and editable inline without re-uploading.

---

## Numbers worth quoting

| Metric | Value | The honest read |
|---|---|---|
| Stage 1 ROC-AUC | **0.632** | Against a **0.646 ceiling** — the generator flips 4% of labels after drawing them, so no model can beat the hidden risk index against the resulting noisy label. That's ~98% of what's achievable on this data. |
| Stage 1 PR-AUC / Brier / KS | 0.192 / 0.088 / 0.186 | PR-AUC ceiling is 0.204. |
| Decision cutoff | 0.170 | Chosen so the **decline rate matches the observed default rate** (10.17%) — no invented cost ratio. Full trade-off table is in `creditsense_bundle.metrics.json`. |
<<<<<<< HEAD
| Retrieval (fused + reranked) | precision@5 **0.267**, recall@10 **0.972**, MRR **0.943**, nDCG@5 **0.953** | Reranking is an *ordering* win: recall is identical to fused-only; MRR went 0.865 → 0.943. |
| Out-of-scope no-hit rate | **0.80** | 4 of 5 unanswerable questions correctly return nothing. Not 5 of 5. |
=======
| Stage 2 (credit limit) R² | 0.974 | **Don't quote this.** The target is a near-deterministic formula of turnover/collateral/risk, so a high R² mostly reflects re-deriving arithmetic. Read `mae_pct_of_mean_actual` = **11.1%** instead. |
| Retrieval (fused + reranked) | precision@5 **0.267**, recall@10 **0.972**, MRR **0.943**, nDCG@5 **0.953** | Reranking is an *ordering* win: recall is identical to fused-only; MRR went 0.865 → 0.943. |
| Out-of-scope no-hit rate | **0.80** | 4 of 5 unanswerable questions correctly return nothing. Not 5 of 5. |
| Tests | **348** offline, 7 live-gated | Coverage 76%, enforced in CI. |
>>>>>>> origin/phase-0-foundation

All of these come from committed artifacts: `src/creditsense/ml/artifacts/creditsense_bundle.metrics.json` and `reports/rag/retrieval_eval.md`.

---

## Design decisions worth knowing before reading the code

- **Numbers never pass through an LLM.** Currency/figure parsing is regex-only (`agents/parsing.py`). The LLM sees free-text notes, writes the risk narrative, and runs the compliance advisory pass — nothing else. An OCR transposition is *flagged if implausible*, never auto-corrected, because one that lands inside a plausible range is indistinguishable from a real value.
- **A compliance flag cannot exist without a citation.** `ComplianceFlag.citations` is `Field(min_length=1)` — structurally unconstructable otherwise. And any citation naming a chunk that wasn't in that call's retrieved set is dropped in code and logged as an escalation reason.
- **Unverified never renders as passed.** A DB/retrieval failure degrades to `insufficient_data=True`, which the supervisor always escalates on and the UI shows as "couldn't verify."
- **`/applications/underwrite` is a sync `def` on purpose.** It self-calls `/predict/credit-risk` over blocking HTTP in the same process; `async def` would deadlock on its own call.
- **MCP masking is a default-deny allowlist.** The generator's row includes six columns that are the model's own answer key; an import-time assertion fails the build if one ever lands in the allowlist.
<<<<<<< HEAD

=======
- **No agent framework.** Four functions in sequence. A graph DSL would hide the guardrail, which is the one thing that should be obvious in the code.
>>>>>>> origin/phase-0-foundation

---

## Testing

```powershell
uv run pytest -q                    # 348 offline tests, no DB, no network
uv run pytest -m live               # 7 tests against the real stack — needs Docker + OPENAI_API_KEY (costs money)
uv run pytest --cov --cov-report=term-missing
uv run ruff check src/creditsense tests
```

The offline suite never touches a live database or a live API. The full-chain integration test runs all four agents for real and fakes only the outer transports — including faking `llm._get_client` rather than `complete_structured`, so `llm.py`'s own parsing and degradation logic actually executes instead of being bypassed.

---

## MCP server

Four tools: `get_applicant_financials`, `search_sbp_regulations`, `get_portfolio_exposure`, `submit_underwriting_decision` (write-scoped, audited, and unable to record an approval over an open breach).

**Claude Code / any stdio client** — `.mcp.json` is committed and works as-is:
```json
{"mcpServers": {"creditsense": {"command": "uv",
  "args": ["--directory", "D:\\CreditSense", "run", "python", "-m", "creditsense.mcp_server"]}}}
```

**HTTP transport** — mounted at `/mcp` on the running app. Authenticate with a bearer key (`MCP_READ_API_KEY` / `MCP_WRITE_API_KEY`) or complete the OAuth flow. Auth is only meaningful over HTTP; stdio has no token concept at all, and the code says so rather than pretending otherwise.

**Claude Desktop's chat connector** needs a public HTTPS URL, so it needs a tunnel:
```powershell
cloudflared tunnel --url http://localhost:8000     # or: ngrok http 8000
# set MCP_PUBLIC_BASE_URL=<that https origin> in .env, restart the app, then:
curl.exe <tunnel-url>/.well-known/oauth-authorization-server   # issuer must read <tunnel-url>/mcp
```
Then add `<tunnel-url>/mcp` as a custom connector. **While that tunnel is up, anyone with the URL can reach every tool including the write one** — the OAuth shim auto-approves, because there's no user directory here to authenticate against. Start it deliberately, stop it when you're done.

---

## Known limitations

Recorded rather than hidden — see §8 of `CreditSense_Master_Tracker.md` for the full list:

- Rate limiter is per-process; multiple uvicorn workers each get their own bucket.
- The ML self-call means the app depends on its own threadpool not being exhausted.
- Browser API routes have **no auth** — deliberate for a local demo, first thing to fix otherwise.
- Corpus is 87 chunks, not the 200+ originally targeted; retrieval metrics are over a small, genuinely-scoped corpus.
- Generator's enterprise-tier turnover breakpoints don't match current SBP regulation text (surfaced by the corpus `MANIFEST.md`, never reconciled).

---

## Documentation map

| File | What it's for |
|---|---|
| `CreditSense_Master_Tracker.md` | Single source of truth — phase status, every implementation note, decision log, open risks, interview answers |
| `FRONTEND_REQUIREMENTS.md` | Product/implementation contract for the UI |
| `reports/rag/retrieval_eval.md` | Retrieval evaluation across four configurations |
| `src/creditsense/data/raw_corpus/MANIFEST.md` | Which corpus text is real SBP regulation vs simulated policy |
| `src/creditsense/ml/artifacts/creditsense_bundle.metrics.json` | Full model metrics including ceilings and the cutoff trade-off table |
<<<<<<< HEAD

=======
>>>>>>> origin/phase-0-foundation
