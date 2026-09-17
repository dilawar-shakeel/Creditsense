from creditsense.mcp_server.masking import (
    ALLOWED_PROFILE_FIELDS,
    allowed_profile_fields,
    mask_account_number,
    mask_cnic,
)
from creditsense.ml.features import LEAKAGE_COLUMNS


def test_mask_account_number_keeps_last_four_digits():
    assert mask_account_number("PK36SCBL0000001123456702") == "PK36****************6702"


def test_mask_account_number_handles_none_and_empty():
    assert mask_account_number(None) is None
    assert mask_account_number("") == ""


def test_mask_account_number_fully_masks_a_short_value():
    assert mask_account_number("ABCD") == "****"


def test_mask_cnic_keeps_first_group_and_last_digit():
    assert mask_cnic("35202-1234567-1") == "35202-*******-1"


def test_mask_cnic_handles_none_and_empty():
    assert mask_cnic(None) is None
    assert mask_cnic("") == ""


def test_mask_cnic_fully_masks_an_unexpected_shape():
    assert mask_cnic("not-a-cnic") == "**********"


def test_allowed_profile_fields_returns_only_the_18_model_features():
    profile = {name: 1 for name in ALLOWED_PROFILE_FIELDS}
    profile.update({name: 999 for name in LEAKAGE_COLUMNS})
    profile["some_future_generator_column"] = "unexpected"

    result = allowed_profile_fields(profile)

    assert set(result.keys()) == ALLOWED_PROFILE_FIELDS
    for leakage_field in LEAKAGE_COLUMNS:
        assert leakage_field not in result
    assert "some_future_generator_column" not in result


def test_allowed_profile_fields_handles_none_and_empty():
    assert allowed_profile_fields(None) == {}
    assert allowed_profile_fields({}) == {}


def test_no_leakage_column_can_ever_appear_in_the_allowlist():
    """Structural guarantee, not just a behavioral test -- mirrors the module-level
    assert in masking.py that would fail at import time if this ever regressed."""
    assert not (ALLOWED_PROFILE_FIELDS & LEAKAGE_COLUMNS)
