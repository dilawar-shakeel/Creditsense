"""P8.2 -- the live half. Deselected by default (see `addopts` in pyproject.toml);
run deliberately with:

    uv run pytest -m live

This is the manual curl verification that has been repeated by hand at the end of
every phase, turned into something runnable. It needs the real Docker stack up
(`docker compose up -d`), a seeded database, a trained model bundle, and a real
OPENAI_API_KEY -- so it is NOT part of the default suite and CI never runs it.

Everything here goes over real HTTP against the running app: no patching, no fakes.
"""

from __future__ import annotations

import httpx
import pytest

from creditsense.config import get_settings

pytestmark = pytest.mark.live

BASE_URL = "http://localhost:8000"
DEMO_APPLICANT = "SME-000200"


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=120.0) as http_client:
        try:
            health = http_client.get("/health")
        except httpx.ConnectError:
            pytest.skip("Docker stack is not running -- start it with `docker compose up -d`")
        if health.status_code != 200:
            pytest.skip(f"/health returned {health.status_code}; stack is not ready")
        yield http_client


def test_health_reports_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_underwrite_a_real_applicant_end_to_end(client):
    """The real chain against real data: real Postgres lookup, real XGBoost model over
    real HTTP, real embeddings + pgvector retrieval, real OpenAI calls."""
    response = client.post(
        "/applications/underwrite",
        json={"applicant_id": DEMO_APPLICANT, "requested_amount_pkr": 2_000_000},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applicant_id"] == DEMO_APPLICANT
    assert body["decision"] in {"APPROVE", "DECLINE", "ESCALATE_TO_HUMAN"}
    # A decision that reached the model carries a real risk assessment with a real
    # SHAP explanation, not a placeholder.
    if body["decision"] != "ESCALATE_TO_HUMAN":
        assert body["risk"]["shap_explanation_raw"]
        assert 0.0 <= body["risk"]["default_probability"] <= 1.0
    assert response.headers["X-Request-ID"]


def test_the_guardrail_holds_against_real_regulation_data(client):
    """The centrepiece, live: a facility far over the R-5 ceiling must be declined
    citing the real SBP clause text loaded into Postgres -- not approved."""
    response = client.post(
        "/applications/underwrite",
        json={"applicant_id": DEMO_APPLICANT, "requested_amount_pkr": 150_000_000},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "DECLINE"

    breaches = [f for f in body["compliance"]["flags"] if f["severity"] == "BREACH"]
    assert "R-5" in {f["rule_id"] for f in breaches}
    # The citation is real retrieved clause text, not fabricated.
    r5 = next(f for f in breaches if f["rule_id"] == "R-5")
    assert r5["citations"][0]["excerpt"].strip()


def test_uploaded_messy_document_is_parsed_live(client):
    """The currency-format mess parsing.py exists to handle, end to end over HTTP."""
    response = client.post(
        "/applications/underwrite",
        json={
            "applicant_id": DEMO_APPLICANT,
            "raw_fields": {
                "current_ratio": 1.5,
                "debt_to_equity_ratio": 0.8,
                "existing_loan_exposure_pkr": "Rs. 2,000,000",
                "collateral_coverage_ratio": 0.9,
                "annual_bank_turnover_pkr": "PKR 40,000,000",
                "years_in_business": 8.0,
                "sector_risk_code": "Retail/Trade",
            },
            "notes": "client ka cash flow seasonal hai, Eid se pehle spike hota hai",
            "requested_amount_pkr": 2_000_000,
        },
    )

    assert response.status_code == 200
    assert response.json()["decision"] in {"APPROVE", "DECLINE", "ESCALATE_TO_HUMAN"}


def test_unknown_applicant_with_no_document_is_404(client):
    response = client.post(
        "/applications/underwrite", json={"applicant_id": "SME-000000-does-not-exist"}
    )
    assert response.status_code == 404


def test_mcp_http_transport_rejects_an_unauthenticated_call(client):
    response = client.post(
        "/mcp/",
        headers={"Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert response.status_code == 401


def test_mcp_http_transport_accepts_a_valid_read_key(client):
    settings = get_settings()
    response = client.post(
        "/mcp/",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {settings.mcp_read_api_key}",
        },
        json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "phase8-live-test", "version": "1"},
            },
        },
    )
    assert response.status_code == 200
