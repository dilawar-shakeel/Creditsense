# CreditSense

CreditSense is an SME credit risk and SBP regulatory compliance assistant.

## Phase 0 quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
uvicorn creditsense.api.main:app --reload
```

Open `http://localhost:8000/health` to check that the API is running.

Copy `.env.example` to `.env` before changing local settings. Docker Compose starts PostgreSQL with pgvector and the API service (the `agent` service is commented out until Phase 5 — see the note in `docker-compose.yml`):

```powershell
docker compose up --build
```

## First run on a clean clone

The synthetic dataset and the trained model are both git-ignored (they're
regeneratable and either large or environment-specific). After cloning:

```powershell
$env:PYTHONPATH = "src"
python -m creditsense.data.generators.portfolio   # writes the 50,000-row CSV + metadata
python -m creditsense.ml.train                    # trains and saves the model bundle
uvicorn creditsense.api.main:app --reload          # now /predict/credit-risk has a model to load
```

`creditsense_bundle.metrics.json` (committed, not ignored) has the numbers worth
quoting — including where the model's ROC-AUC sits relative to the ceiling the dataset's
own label noise imposes, and the cutoff trade-off table it was chosen from. The credit
limit model's R² is high (~0.97) mostly because the target is a near-deterministic
formula (turnover × collateral/risk multipliers, then capped) — the metrics file's
`mae_pct_of_mean_actual` and cap-hit breakdown are the more honest read on it.

## RAG layer (Phase 4)

Requires Postgres running (`docker compose up db`) with migrations applied, and
`OPENAI_API_KEY` set in `.env` for anything that embeds, reranks, or evaluates.

```powershell
alembic upgrade head                              # applies migration 0002 (RAG schema)
python -m creditsense.rag.ingest --dry-run         # parses the corpus, no API calls — this is what CI runs
python -m creditsense.rag.ingest                   # embeds and stores all chunks (needs OPENAI_API_KEY)
python -m creditsense.rag.eval_review              # walk through and verify each eval label
python -m creditsense.rag.eval                     # writes reports/rag/retrieval_eval.md (refuses to run until every label is verified)
```

The corpus (`src/creditsense/data/raw_corpus/`) mixes real SBP regulation text with
simulated internal policy and loan-structuring templates that cite it — see
`MANIFEST.md` there for exactly which files are real vs. simulated, and a discrepancy
it surfaced between the current regulation and the data generator's enterprise-tier
turnover breakpoints (not yet fixed — that's a Phase 1 decision).

`python -m pytest tests/test_chunking.py tests/test_retrieval.py tests/test_eval.py`
runs everything that doesn't need a live database or API key — chunking, RRF fusion
math, reranker degrade-on-failure, and the eval metric formulas.
