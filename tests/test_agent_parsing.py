import pytest

from creditsense.agents.parsing import flag_implausible_magnitude, parse_pkr_amount


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Rs. 2,130,819", 2_130_819.0),
        ("PKR 656,553", 656_553.0),
        ("2,252,132", 2_252_132.0),
        ("391324", 391_324.0),  # OCR-noised: bare digits, no commas, no prefix
        ("rs. 1,000", 1_000.0),  # case-insensitive prefix
        (1_234_567.0, 1_234_567.0),  # already numeric -- pass through
        (None, None),
        ("", None),
        ("   ", None),
    ],
)
def test_parse_pkr_amount_handles_every_corpus_format(raw, expected):
    assert parse_pkr_amount(raw) == expected


def test_flag_implausible_magnitude_lets_in_range_values_through():
    assert flag_implausible_magnitude("annual_bank_turnover_pkr", 40_000_000.0) is None


def test_flag_implausible_magnitude_catches_a_value_outside_generator_bounds():
    # Above the generator's own 800,000,000 truncation ceiling -- something a real
    # generated row could never contain, e.g. a transposition that added a digit.
    warning = flag_implausible_magnitude("annual_bank_turnover_pkr", 8_000_000_000.0)
    assert warning is not None
    assert "annual_bank_turnover_pkr" in warning


def test_flag_implausible_magnitude_ignores_none():
    assert flag_implausible_magnitude("annual_bank_turnover_pkr", None) is None


def test_flag_implausible_magnitude_is_a_noop_for_unconfigured_fields():
    # current_ratio has no configured plausible range -- must not raise or invent one.
    assert flag_implausible_magnitude("current_ratio", -999.0) is None
