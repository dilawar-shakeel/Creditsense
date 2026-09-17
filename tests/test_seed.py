import json
import math

import numpy as np

from creditsense.db.seed import (
    _clean_scalar,
    _load_cohort_ids,
    _row_to_values,
    synthetic_account_number,
    synthetic_cnic,
)


def test_clean_scalar_converts_numpy_types_to_plain_python():
    assert isinstance(_clean_scalar(np.int64(5)), int)
    assert isinstance(_clean_scalar(np.float64(1.5)), float)
    assert _clean_scalar(np.bool_(True)) is True


def test_clean_scalar_turns_nan_into_none():
    assert _clean_scalar(float("nan")) is None
    assert _clean_scalar(np.float64("nan")) is None


def test_clean_scalar_leaves_ordinary_values_alone():
    assert _clean_scalar("TEXTILE") == "TEXTILE"
    assert _clean_scalar(None) is None


def test_row_to_values_maps_generator_columns_to_applicant_fields():
    row = {
        "applicant_id": "SME-000001",
        "sector_risk_code": "TEXTILE",
        "years_in_business": np.float64(7.5),
        "documentation_tier": "Tier-2",
        "existing_loan_exposure_pkr": np.float64(12_000_000.0),
        "ecib_no_hit_flag": np.bool_(False),
        "declared_income_ratio": float("nan"),
    }

    values = _row_to_values(row)

    assert values["applicant_id"] == "SME-000001"
    assert values["sector"] == "TEXTILE"
    assert values["years_in_business"] == 7.5
    assert values["documentation_tier"] == "Tier-2"

    # The full cleaned row is preserved in raw_profile_json, including fields not
    # promoted to their own column — that's what the P2.4 exposure view reads from.
    profile = values["raw_profile_json"]
    assert profile["existing_loan_exposure_pkr"] == 12_000_000.0
    assert profile["ecib_no_hit_flag"] is False
    assert profile["declared_income_ratio"] is None  # NaN cleaned to None, not NaN
    assert not any(
        isinstance(v, float) and math.isnan(v) for v in profile.values()
    )


def test_row_to_values_adds_synthetic_identifiers():
    row = {"applicant_id": "SME-000001", "sector_risk_code": "TEXTILE"}
    values = _row_to_values(row)
    assert values["account_number"] == synthetic_account_number("SME-000001")
    assert values["cnic"] == synthetic_cnic("SME-000001")


def test_synthetic_account_number_is_deterministic_and_iban_shaped():
    first = synthetic_account_number("SME-000001")
    second = synthetic_account_number("SME-000001")
    assert first == second  # re-seeding the same applicant_id must be reproducible
    assert len(first) == 24
    assert first.startswith("PK")
    assert first[4:8].isalpha()
    assert first[8:].isdigit()


def test_synthetic_account_number_differs_per_applicant():
    assert synthetic_account_number("SME-000001") != synthetic_account_number("SME-000002")


def test_synthetic_cnic_is_deterministic_and_cnic_shaped():
    first = synthetic_cnic("SME-000001")
    second = synthetic_cnic("SME-000001")
    assert first == second
    assert len(first) == 15
    parts = first.split("-")
    assert [len(p) for p in parts] == [5, 7, 1]
    assert first.replace("-", "").isdigit()


def test_synthetic_cnic_differs_per_applicant():
    assert synthetic_cnic("SME-000001") != synthetic_cnic("SME-000002")


def test_cohort_seed_deletes_applicants_outside_the_cohort(tmp_path, monkeypatch):
    """--cohort must make the table BE the cohort, not just add to whatever was
    already there -- otherwise a demo could silently run against 50,060 applicants."""
    import pandas as pd

    from creditsense.db import seed as seed_module

    csv_path = tmp_path / "portfolio.csv"
    pd.DataFrame(
        [
            {"applicant_id": "SME-000001", "sector_risk_code": "Retail/Trade"},
            {"applicant_id": "SME-000002", "sector_risk_code": "Retail/Trade"},
            {"applicant_id": "SME-000003", "sector_risk_code": "Retail/Trade"},
        ]
    ).to_csv(csv_path, index=False)

    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps([{"applicant_id": "SME-000002"}]), encoding="utf-8")

    executed_statement_types: list[str] = []

    class FakeResult:
        rowcount = 1

    class FakeTableConn:
        def execute(self, stmt):
            executed_statement_types.append(type(stmt).__name__)
            return FakeResult()

    class FakeSession:
        def connection(self):
            return FakeTableConn()

    class FakeSessionScope:
        def __enter__(self):
            return FakeSession()

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(seed_module, "session_scope", lambda: FakeSessionScope())

    seed_module.seed(csv_path=csv_path, cohort_path=cohort_path)

    # A Delete must run before any Insert -- deleting the rest of the table is not an
    # afterthought, and a partial cohort file must never leave old and new data mixed.
    assert "Delete" in executed_statement_types
    assert executed_statement_types.index("Delete") < executed_statement_types.index("Insert")


def test_load_cohort_ids_reads_applicant_ids_from_a_json_fixture(tmp_path):
    cohort_file = tmp_path / "cohort.json"
    cohort_file.write_text(
        json.dumps([{"applicant_id": "SME-000001"}, {"applicant_id": "SME-000002"}]),
        encoding="utf-8",
    )
    assert _load_cohort_ids(cohort_file) == {"SME-000001", "SME-000002"}
