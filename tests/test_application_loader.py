"""P8.1 gap: agents/application_loader.py was introduced in Phase 7 (moved out of
worker.py so the new HTTP endpoint could share it) with no direct tests of its own --
only indirect references that patched it out. It is the shared document-loading path
behind BOTH the worker CLI and POST /applications/underwrite, so it is worth real
coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from creditsense.agents.application_loader import load_application


class _FakeQuery:
    def __init__(self, result):
        self._result = result

    def filter_by(self, **kwargs):
        return self

    def first(self):
        return self._result


class _FakeSession:
    """Just the `session.query(Model).filter_by(...).first()` chain the loader uses."""

    def __init__(self, applicant=None):
        self._applicant = applicant
        self.query_calls = 0

    def query(self, model):
        self.query_calls += 1
        return _FakeQuery(self._applicant)


def _write_fixture(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "messy_applications.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _fixture_record() -> dict:
    return {
        "applicant_id": "SME-000001",
        "sector_risk_code": "Retail/Trade",
        "raw_fields": {"current_ratio": 1.4, "annual_bank_turnover_pkr": "Rs. 5,000,000"},
        "notes": "client ka cash flow seasonal hai",
    }


def test_loads_the_messy_fixture_when_the_applicant_is_in_it(tmp_path):
    fixture_path = _write_fixture(tmp_path, [_fixture_record()])
    session = _FakeSession()

    application = load_application("SME-000001", session, fixture_path=fixture_path)

    assert application is not None
    assert application.raw_fields["annual_bank_turnover_pkr"] == "Rs. 5,000,000"
    assert application.notes == "client ka cash flow seasonal hai"
    # The fixture hit short-circuits before any DB lookup happens at all.
    assert session.query_calls == 0


def test_sector_risk_code_is_backfilled_from_the_records_top_level_field(tmp_path):
    """The fixture carries sector_risk_code outside raw_fields; the loader folds it in
    without overwriting one that is already present in raw_fields."""
    fixture_path = _write_fixture(tmp_path, [_fixture_record()])

    application = load_application("SME-000001", _FakeSession(), fixture_path=fixture_path)

    assert application.raw_fields["sector_risk_code"] == "Retail/Trade"


def test_an_explicit_sector_in_raw_fields_is_not_overwritten(tmp_path):
    record = _fixture_record()
    record["raw_fields"]["sector_risk_code"] = "Textiles"
    fixture_path = _write_fixture(tmp_path, [record])

    application = load_application("SME-000001", _FakeSession(), fixture_path=fixture_path)

    assert application.raw_fields["sector_risk_code"] == "Textiles"


def test_falls_back_to_the_applicant_record_when_not_in_the_fixture(tmp_path):
    fixture_path = _write_fixture(tmp_path, [_fixture_record()])
    applicant = SimpleNamespace(
        applicant_id="SME-000002",
        raw_profile_json={
            "current_ratio": 2.1,
            "debt_to_equity_ratio": 0.4,
            "existing_loan_exposure_pkr": 500_000.0,
            "collateral_coverage_ratio": 1.2,
            "annual_bank_turnover_pkr": 9_000_000.0,
            "years_in_business": 12.0,
            "sector_risk_code": "Textiles",
        },
    )
    session = _FakeSession(applicant)

    application = load_application("SME-000002", session, fixture_path=fixture_path)

    assert application is not None
    assert application.raw_fields["current_ratio"] == 2.1
    assert application.raw_fields["sector_risk_code"] == "Textiles"
    assert application.notes is None
    assert session.query_calls == 1


def test_falls_back_to_the_applicant_record_when_the_fixture_file_is_absent(tmp_path):
    applicant = SimpleNamespace(applicant_id="SME-000003", raw_profile_json={"current_ratio": 1.1})

    application = load_application(
        "SME-000003", _FakeSession(applicant), fixture_path=tmp_path / "nonexistent.json"
    )

    assert application is not None
    assert application.raw_fields["current_ratio"] == 1.1


def test_returns_none_when_neither_source_has_the_applicant(tmp_path):
    """The signal the endpoint turns into a 404 and the worker CLI turns into exit 1."""
    application = load_application(
        "SME-999999", _FakeSession(None), fixture_path=tmp_path / "nonexistent.json"
    )

    assert application is None


def test_an_applicant_with_an_empty_profile_still_loads(tmp_path):
    """raw_profile_json of None must not crash -- every document field simply comes
    back unresolved for the analyst to report."""
    applicant = SimpleNamespace(applicant_id="SME-000004", raw_profile_json=None)

    application = load_application(
        "SME-000004", _FakeSession(applicant), fixture_path=tmp_path / "nonexistent.json"
    )

    assert application is not None
    assert all(value is None for value in application.raw_fields.values())


def test_loan_terms_are_threaded_through_both_paths(tmp_path):
    fixture_path = _write_fixture(tmp_path, [_fixture_record()])

    from_fixture = load_application(
        "SME-000001", _FakeSession(), fixture_path=fixture_path,
        requested_amount_pkr=3_000_000.0, is_clean_facility=True, tenor_months=24,
    )
    from_record = load_application(
        "SME-000002", _FakeSession(SimpleNamespace(raw_profile_json={})),
        fixture_path=fixture_path,
        requested_amount_pkr=3_000_000.0, is_clean_facility=True, tenor_months=24,
    )

    for application in (from_fixture, from_record):
        assert application.requested_amount_pkr == 3_000_000.0
        assert application.is_clean_facility is True
        assert application.tenor_months == 24
