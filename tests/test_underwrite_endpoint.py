"""P7.1: POST /applications/underwrite. Same style as test_prediction_endpoint.py --
inline TestClient(app), seams monkeypatched by string path, no live DB or LLM. The
rate limiter is a shared module-level singleton in api/applications.py, so every test
here neutralizes it explicitly rather than relying on call ordering.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from creditsense.agents.schemas import UnderwritingDecision
from creditsense.api.main import app
from creditsense.db.session import get_db


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr("creditsense.api.applications._limiter.check", lambda key: None)


@pytest.fixture(autouse=True)
def _fake_db():
    app.dependency_overrides[get_db] = lambda: iter([SimpleNamespace()])
    yield
    app.dependency_overrides.pop(get_db, None)


def _decision(decision: str = "DECLINE") -> UnderwritingDecision:
    return UnderwritingDecision(
        applicant_id="SME-000200", decision=decision, rationale="test rationale",
    )


def test_uploaded_document_reaches_the_pipeline(monkeypatch):
    captured = {}

    def fake_run_underwriting(application, session):
        captured["application"] = application
        return _decision("APPROVE")

    monkeypatch.setattr("creditsense.api.applications.run_underwriting", fake_run_underwriting)

    response = TestClient(app).post(
        "/applications/underwrite",
        json={
            "applicant_id": "SME-000200",
            "raw_fields": {"current_ratio": 1.5, "sector_risk_code": "Retail/Trade"},
            "requested_amount_pkr": 2_000_000.0,
        },
    )

    assert response.status_code == 200
    assert response.json()["decision"] == "APPROVE"
    assert captured["application"].raw_fields["current_ratio"] == 1.5


def test_omitted_raw_fields_loads_the_document(monkeypatch):
    loaded = object()

    def fake_load_application(applicant_id, session, **kwargs):
        assert applicant_id == "SME-000200"
        return loaded

    captured = {}

    def fake_run_underwriting(application, session):
        captured["application"] = application
        return _decision()

    monkeypatch.setattr(
        "creditsense.api.applications.load_application", fake_load_application
    )
    monkeypatch.setattr("creditsense.api.applications.run_underwriting", fake_run_underwriting)

    response = TestClient(app).post(
        "/applications/underwrite", json={"applicant_id": "SME-000200"}
    )

    assert response.status_code == 200
    assert captured["application"] is loaded


def test_unknown_applicant_with_no_document_is_404(monkeypatch):
    monkeypatch.setattr(
        "creditsense.api.applications.load_application", lambda *a, **k: None
    )

    response = TestClient(app).post(
        "/applications/underwrite", json={"applicant_id": "SME-999999"}
    )

    assert response.status_code == 404


def test_ml_service_down_still_returns_200_escalation(monkeypatch):
    """Confirms P7.1 doesn't turn P5.6's clean escalation-on-ML-failure into a 5xx --
    run_underwriting already returns a normal UnderwritingDecision in that case."""

    def fake_run_underwriting(application, session):
        return UnderwritingDecision(
            applicant_id="SME-000200",
            decision="ESCALATE_TO_HUMAN",
            rationale="ML service unavailable.",
            escalation_reasons=["Risk scoring returned no result."],
        )

    monkeypatch.setattr("creditsense.api.applications.run_underwriting", fake_run_underwriting)

    response = TestClient(app).post(
        "/applications/underwrite",
        json={"applicant_id": "SME-000200", "raw_fields": {"sector_risk_code": "Retail/Trade"}},
    )

    assert response.status_code == 200
    assert response.json()["decision"] == "ESCALATE_TO_HUMAN"


def test_malformed_body_is_422():
    response = TestClient(app).post("/applications/underwrite", json={"unexpected_field": 1})

    assert response.status_code == 422


def test_pipeline_value_error_is_422(monkeypatch):
    def raise_value_error(application, session):
        raise ValueError("bad applicant data")

    monkeypatch.setattr("creditsense.api.applications.run_underwriting", raise_value_error)

    response = TestClient(app).post(
        "/applications/underwrite",
        json={"applicant_id": "SME-000200", "raw_fields": {"sector_risk_code": "Retail/Trade"}},
    )

    assert response.status_code == 422


def test_an_unexpected_error_becomes_a_structured_500_with_no_stack_trace(monkeypatch):
    """P8.1: api/main.py's generic exception handler. An error type the route does not
    map itself (it maps KeyError/ValueError to 422) must still reach the caller as a
    clean JSON 500 -- never a stack trace, never an empty response."""

    def raise_unexpected(application, session):
        raise RuntimeError("database connection lost mid-request")

    monkeypatch.setattr("creditsense.api.applications.run_underwriting", raise_unexpected)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/applications/underwrite",
        json={"applicant_id": "SME-000200", "raw_fields": {"sector_risk_code": "Retail/Trade"}},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "An internal error occurred."}
    # The underlying error text must not leak to the caller.
    assert "database connection lost" not in response.text


def test_rate_limit_exceeded_returns_429(monkeypatch):
    from creditsense.ratelimit import RateLimitExceeded

    def always_exceeded(key):
        raise RateLimitExceeded(retry_after_seconds=5.0)

    monkeypatch.setattr("creditsense.api.applications._limiter.check", always_exceeded)

    response = TestClient(app).post(
        "/applications/underwrite",
        json={"applicant_id": "SME-000200", "raw_fields": {"sector_risk_code": "Retail/Trade"}},
    )

    assert response.status_code == 429
    assert "Retry-After" in response.headers
