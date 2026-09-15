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

Copy `.env.example` to `.env` before changing local settings. Docker Compose starts PostgreSQL with pgvector, the API service, and the agent service:

```powershell
docker compose up --build
```
