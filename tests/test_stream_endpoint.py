"""P9.1: GET /applications/underwrite/stream -- the SSE trace the frontend's
Application Review screen consumes (FRONTEND_REQUIREMENTS.md §2.3). Same fake-transport
pattern as test_integration_pipeline.py (fake `_get_http_client`/`_get_client`) so the
four real agents run for real but nothing live (DB, OpenAI, ML) is touched -- this is
the one place that exercises pipeline.py's `on_stage` callback wired through an actual
HTTP response rather than a direct function call.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from creditsense.agents.schemas import LoanApplication
from creditsense.api.main import app
from creditsense.db.session import get_db
from creditsense.rag.retrieval import RetrievedChunk


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr("creditsense.api.stream._limiter.check", lambda key: None)


@pytest.fixture(autouse=True)
def _fake_db():
    app.dependency_overrides[get_db] = lambda: iter([SimpleNamespace()])
    yield
    app.dependency_overrides.pop(get_db, None)


client = TestClient(app)


class _FakeStructuredClient:
    def __init__(self, schema):
        self._schema = schema

    def invoke(self, messages):
        name = self._schema.__name__
        if name == "_Narrative":
            return self._schema(narrative="Plain-English summary of the risk drivers.")
        if name == "NoteInsights":
            return self._schema(seasonality=False, summary_english="")
        return self._schema()


class _FakeLLMClient:
    def with_structured_output(self, schema):
        return _FakeStructuredClient(schema)


class _FakeMLClient:
    def post(self, path, json):
        body = {
            "default_probability": 0.05, "decision_cutoff": 0.17, "decision": "REFER_FOR_LIMIT",
            "recommended_credit_limit_pkr": 4_000_000.0, "explanation": "Main reasons:\n- ok",
        }
        return SimpleNamespace(status_code=200, json=lambda: body)


def _chunk(regulation_number: str, chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, regulation_number=regulation_number,
        clause_title=f"Title {regulation_number}", section_path="Part II",
        content=f"Text {regulation_number}.", source_type="sbp_regulation",
        cross_references=[], score=8.0, source="reranked",
    )


def _application() -> LoanApplication:
    return LoanApplication(
        applicant_id="SME-000001",
        raw_fields={
            "current_ratio": 1.5, "debt_to_equity_ratio": 0.8,
            "existing_loan_exposure_pkr": "2,000,000", "collateral_coverage_ratio": 0.9,
            "annual_bank_turnover_pkr": "40,000,000", "years_in_business": 8.0,
            "sector_risk_code": "Retail/Trade",
        },
        requested_amount_pkr=2_000_000.0,
    )


def _parse_sse(text: str) -> list[tuple[str, str]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.strip().splitlines()
        event = next((ln.split(": ", 1)[1] for ln in lines if ln.startswith("event: ")), None)
        data = next((ln.split(": ", 1)[1] for ln in lines if ln.startswith("data: ")), None)
        if event:
            events.append((event, data))
    return events


def test_stream_emits_four_stage_events_then_the_final_decision(clean_applicant):
    with patch("creditsense.api.stream.load_application", return_value=_application()), \
         patch("creditsense.agents.financial_analyst._fetch_applicant", return_value=clean_applicant), \
         patch("creditsense.agents.risk_scoring._get_http_client", return_value=_FakeMLClient()), \
         patch("creditsense.agents.llm._get_client", return_value=_FakeLLMClient()), \
         patch("creditsense.agents.compliance._sector_pct_of_book", return_value=5.0), \
         patch("creditsense.agents.compliance.lookup_by_regulation_number",
               side_effect=lambda session, rule_id: [_chunk(rule_id, f"chunk-{rule_id}")]), \
         patch("creditsense.agents.compliance.hybrid_search", return_value=[_chunk("P-21", "chunk-adv")]), \
         patch("creditsense.agents.pipeline.session_scope"):
        response = client.get("/applications/underwrite/stream?applicant_id=SME-000001")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    event_names = [name for name, _ in events]

    assert event_names == ["stage.completed", "stage.completed", "stage.completed", "stage.completed", "underwriting.decided"]

    import json
    stage_payloads = [json.loads(data) for name, data in events if name == "stage.completed"]
    assert [p["stage"] for p in stage_payloads] == ["analyze", "score", "check", "decide"]

    final = json.loads(events[-1][1])
    assert final["applicant_id"] == "SME-000001"
    assert final["decision"] in {"APPROVE", "DECLINE", "ESCALATE_TO_HUMAN"}
    assert final["parsed"] is not None


def test_stream_unknown_applicant_is_404(monkeypatch):
    monkeypatch.setattr("creditsense.api.stream.load_application", lambda *a, **k: None)

    response = client.get("/applications/underwrite/stream?applicant_id=SME-999999")

    assert response.status_code == 404


def test_stream_with_raw_fields_skips_load_application_and_never_404s(monkeypatch, clean_applicant):
    """The PDF-intake path (new-application.html): a brand-new applicant_id that
    `load_application` would 404 on must still run when raw_fields is supplied
    directly, exactly like POST /applications/underwrite's raw_fields branch."""
    import json as jsonlib

    def fail_if_called(*a, **k):
        raise AssertionError("load_application should not be called when raw_fields is supplied")

    monkeypatch.setattr("creditsense.api.stream.load_application", fail_if_called)
    raw_fields = jsonlib.dumps({"sector_risk_code": "Retail/Trade", "current_ratio": 1.5})

    with patch("creditsense.agents.financial_analyst._fetch_applicant", return_value=None), \
         patch("creditsense.agents.risk_scoring._get_http_client", return_value=_FakeMLClient()), \
         patch("creditsense.agents.llm._get_client", return_value=_FakeLLMClient()), \
         patch("creditsense.agents.compliance._sector_pct_of_book", return_value=5.0), \
         patch("creditsense.agents.compliance.lookup_by_regulation_number",
               side_effect=lambda session, rule_id: [_chunk(rule_id, f"chunk-{rule_id}")]), \
         patch("creditsense.agents.compliance.hybrid_search", return_value=[_chunk("P-21", "chunk-adv")]), \
         patch("creditsense.agents.pipeline.session_scope"):
        response = client.get(
            "/applications/underwrite/stream",
            params={"applicant_id": "SME-BRAND-NEW", "raw_fields": raw_fields},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [name for name, _ in events][-1] == "underwriting.decided"


def test_stream_with_invalid_raw_fields_json_is_422():
    response = client.get(
        "/applications/underwrite/stream",
        params={"applicant_id": "SME-BRAND-NEW", "raw_fields": "{not valid json"},
    )

    assert response.status_code == 422


def test_stream_rate_limit_returns_429(monkeypatch):
    from creditsense.ratelimit import RateLimitExceeded

    def always_exceeded(key):
        raise RateLimitExceeded(retry_after_seconds=3.0)

    monkeypatch.setattr("creditsense.api.stream._limiter.check", always_exceeded)

    response = client.get("/applications/underwrite/stream?applicant_id=SME-000001")

    assert response.status_code == 429
    assert "Retry-After" in response.headers
