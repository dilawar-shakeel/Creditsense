import math

import numpy as np

from creditsense.db.seed import _clean_scalar, _row_to_values


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
