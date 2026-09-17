"""P9.1: the read/query routes in api/browse.py. Same style as
test_underwrite_endpoint.py -- inline TestClient(app), seams monkeypatched by string
path, no live DB. Each route's own private DB helper (search/fetch functions in
browse.py) is patched directly rather than faking SQLAlchemy's execute(select(...))
machinery, matching this repo's established seam convention.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from creditsense.api.main import app
from creditsense.db.session import get_db
from creditsense.rag.retrieval import RetrievedChunk


@pytest.fixture(autouse=True)
def _fake_db():
    app.dependency_overrides[get_db] = lambda: iter([SimpleNamespace()])
    yield
    app.dependency_overrides.pop(get_db, None)


client = TestClient(app)


def _applicant(**overrides) -> SimpleNamespace:
    defaults = dict(
        applicant_id="SME-000200", sector="Retail/Trade", years_in_business=8.0,
        documentation_tier="Audited Financials", account_number="PK36SCBL0000001123456702",
        cnic="35202-1234567-1", raw_profile_json={"current_ratio": 1.5},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _audit_row(**overrides) -> SimpleNamespace:
    defaults = dict(
        id="audit-1", actor="UnderwritingSupervisorAgent", action="underwriting_decision",
        resource_id="SME-000200", status="success",
        payload_json={"decision": "APPROVE", "approved_amount_pkr": 2_000_000.0, "rationale": "ok"},
        created_at=datetime(2026, 9, 17, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _chunk(regulation_number: str, chunk_id: str = "chunk-1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, regulation_number=regulation_number,
        clause_title=f"Title for {regulation_number}", section_path="Part II",
        content=f"Full text of {regulation_number}.", source_type="sbp_regulation",
        cross_references=["P-1"], score=7.5, source="reranked",
    )


def test_search_applicants(monkeypatch):
    monkeypatch.setattr("creditsense.api.browse._search_applicants", lambda s, q, limit: [_applicant()])

    response = client.get("/api/applicants?search=SME")

    assert response.status_code == 200
    assert response.json()[0]["applicant_id"] == "SME-000200"


def test_get_applicant_found_includes_masked_identifiers_and_raw_document(monkeypatch):
    monkeypatch.setattr("creditsense.api.browse._fetch_applicant", lambda s, aid: _applicant())
    monkeypatch.setattr(
        "creditsense.api.browse._fetch_fixture_document",
        lambda aid: {"raw_fields": {"current_ratio": "1.5"}, "notes": "note"},
    )

    response = client.get("/api/applicants/SME-000200")

    assert response.status_code == 200
    body = response.json()
    assert body["cnic"] == "35202-*******-1"
    assert body["account_number"].startswith("PK36") and body["account_number"].endswith("6702")
    assert body["financials"] == {"current_ratio": 1.5}
    assert body["raw_document"]["notes"] == "note"


def test_get_applicant_not_found_is_404(monkeypatch):
    monkeypatch.setattr("creditsense.api.browse._fetch_applicant", lambda s, aid: None)

    response = client.get("/api/applicants/SME-999999")

    assert response.status_code == 404


def test_create_applicant_inserts_a_new_row_when_none_exists(monkeypatch):
    """The PDF-intake write path (new-application.html): financial_analyst.py's own
    logic is untouched -- this just needs a real row to exist for it to find."""
    monkeypatch.setattr("creditsense.api.browse._fetch_applicant", lambda s, aid: None)
    added: list = []
    app.dependency_overrides[get_db] = _recording_get_db(added)

    response = client.post(
        "/api/applicants",
        json={
            "applicant_id": "SME-NEW-001",
            "sector": "Textile",
            "years_in_business": 3.0,
            "raw_profile_json": {"documentation_tier": "Audited Financials", "ecib_score": 720},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applicant_id"] == "SME-NEW-001"
    assert body["documentation_tier"] == "Audited Financials"
    assert len(added) == 1
    assert added[0].raw_profile_json == {"documentation_tier": "Audited Financials", "ecib_score": 720}


def test_create_applicant_upserts_an_existing_row(monkeypatch):
    existing = _applicant(applicant_id="SME-000200", sector="Retail/Trade")
    monkeypatch.setattr("creditsense.api.browse._fetch_applicant", lambda s, aid: existing)

    def _fake_get_db():
        yield SimpleNamespace(add=lambda o: None, commit=lambda: None)

    app.dependency_overrides[get_db] = _fake_get_db

    response = client.post(
        "/api/applicants",
        json={"applicant_id": "SME-000200", "sector": "Textile", "raw_profile_json": {}},
    )

    assert response.status_code == 200
    assert response.json()["sector"] == "Textile"
    # updated the same object rather than inserting a second one
    assert existing.sector == "Textile"


def test_list_decisions_reads_the_stored_payload(monkeypatch):
    monkeypatch.setattr("creditsense.api.browse._list_audit_logs", lambda s, status, limit: [_audit_row()])

    response = client.get("/api/decisions?status=success")

    assert response.status_code == 200
    row = response.json()[0]
    assert row["decision"] == "APPROVE"
    assert row["approved_amount_pkr"] == 2_000_000.0


def test_get_decision_not_found_is_404(monkeypatch):
    monkeypatch.setattr("creditsense.api.browse._fetch_audit_log", lambda s, decision_id: None)

    response = client.get("/api/decisions/does-not-exist")

    assert response.status_code == 404


def _recording_get_db(added: list):
    """Mimics just enough of the real Session's flush-time behavior for these two
    write routes: SQLAlchemy's ORM assigns AuditLog.id (a Python-side `default=`) and
    Postgres assigns created_at (a server_default) at flush/INSERT time, neither of
    which a bare SimpleNamespace().add()/.commit() would ever populate."""

    def _add(obj):
        obj.id = str(uuid4())
        obj.created_at = datetime.now(timezone.utc)
        added.append(obj)

    def _fake_get_db():
        yield SimpleNamespace(add=_add, commit=lambda: None)

    return _fake_get_db


def test_accept_decision_appends_a_new_row_without_mutating_the_original(monkeypatch):
    target = _audit_row()
    monkeypatch.setattr("creditsense.api.browse._fetch_audit_log", lambda s, decision_id: target)
    added: list = []
    app.dependency_overrides[get_db] = _recording_get_db(added)

    response = client.post("/api/decisions/audit-1/accept")

    assert response.status_code == 200
    assert response.json()["action"] == "decision_accepted"
    assert len(added) == 1
    assert added[0].payload_json == {"accepted_decision_id": "audit-1"}
    # the original row's own action is untouched
    assert target.action == "underwriting_decision"


def test_override_requires_a_meaningful_reason():
    response = client.post("/api/decisions/audit-1/override", json={"reason": "no"})

    assert response.status_code == 422


def test_override_decision_appends_a_row_with_the_reason(monkeypatch):
    target = _audit_row()
    monkeypatch.setattr("creditsense.api.browse._fetch_audit_log", lambda s, decision_id: target)
    added: list = []
    app.dependency_overrides[get_db] = _recording_get_db(added)

    response = client.post(
        "/api/decisions/audit-1/override", json={"reason": "Client relationship exception approved by branch manager."}
    )

    assert response.status_code == 200
    assert added[0].action == "decision_overridden"
    assert added[0].payload_json["reason"].startswith("Client relationship")


def test_search_regulations_tags_provenance(monkeypatch):
    monkeypatch.setattr(
        "creditsense.api.browse.hybrid_search",
        lambda session, q, **kwargs: [_chunk("R-5"), _chunk("P-21", "chunk-2")],
    )

    response = client.get("/api/regulations/search?q=exposure")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["provenance"] == "real_sbp_regulation"
    assert body[1]["provenance"] == "simulated_policy"


def test_get_regulation_concatenates_chunks_and_dedupes_cross_references(monkeypatch):
    monkeypatch.setattr(
        "creditsense.api.browse.lookup_by_regulation_number",
        lambda session, number: [_chunk(number, "c1"), _chunk(number, "c2")],
    )

    response = client.get("/api/regulations/R-5")

    assert response.status_code == 200
    body = response.json()
    assert body["provenance"] == "real_sbp_regulation"
    assert "Full text of R-5." in body["content"]
    assert body["content"].count("Full text of R-5.") == 2
    assert body["cross_references"] == ["P-1"]


def test_get_regulation_unknown_is_404(monkeypatch):
    monkeypatch.setattr("creditsense.api.browse.lookup_by_regulation_number", lambda session, number: [])

    response = client.get("/api/regulations/R-999")

    assert response.status_code == 404


def test_portfolio_exposure_includes_the_p1_limit(monkeypatch):
    row = SimpleNamespace(sector="Retail/Trade", applicant_count=100, total_exposure_pkr=5_000_000.0, pct_of_book=21.8)
    monkeypatch.setattr("creditsense.api.browse._fetch_portfolio_rows", lambda s: [row])

    response = client.get("/api/portfolio/exposure")

    assert response.status_code == 200
    body = response.json()
    assert body["rows"][0]["sector"] == "Retail/Trade"
    assert body["sector_concentration_limit_pct"] == 25.0
