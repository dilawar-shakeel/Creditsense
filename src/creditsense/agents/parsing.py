"""Pure parsing helpers for messy applicant documents (P5.2).

No LLM, no I/O, no database -- these functions only exist because the numbers in an
underwriting system must never pass through an LLM. An LLM that silently "corrects" a
turnover figure is the worst failure mode this project could ship, and nothing
downstream would be able to detect it. Free-text notes are the one thing in this agent
that does go to an LLM -- see `financial_analyst.py`.

Fixture reality this is written against (`tests/fixtures/messy_applications.json`, 60
records, built by `tests/fixtures/build_messy_applications.py`):
  - `existing_loan_exposure_pkr` and `annual_bank_turnover_pkr` are the only fields ever
    given as strings, in one of three formats cycling by row index (i % 3):
    "Rs. 2,130,819" / "PKR 656,553" / "2,252,132" (bare, comma-separated).
  - Rows with `ocr_digit_transposition` noise skip currency formatting entirely and are
    bare digit strings with no commas and no prefix ("391324") -- indistinguishable from
    a legitimate bare value by shape alone. `noise_applied` is the only reliable signal;
    this module does not try to detect or "fix" a transposition, only to flag a value
    that landed outside a plausible range after parsing (see `flag_implausible_magnitude`).
  - `current_ratio`, `debt_to_equity_ratio`, `years_in_business` are always raw floats,
    never string-formatted.
  - `collateral_coverage_ratio` is `null` for exactly two records (missing_collateral
    noise). `parse_pkr_amount`/passthrough must never invent a replacement value for a
    missing field -- the caller records it as unresolved instead.
"""

from __future__ import annotations

import re

_CURRENCY_PREFIX_RE = re.compile(r"^\s*(Rs\.?|PKR)\s*", re.IGNORECASE)

# Bounds a value must fall within to be considered plausible, keyed by field name.
# Matches the generator's own truncation bounds for the field (portfolio.py's
# truncated_ppf calls), so a flagged value is genuinely outside what the data-generation
# process itself would ever have produced -- not an arbitrary guess.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "annual_bank_turnover_pkr": (500_000.0, 800_000_000.0),
}


def parse_pkr_amount(value: str | float | int | None) -> float | None:
    """Parse a PKR amount in any of the corpus's known formats into a plain float.

    Accepts "Rs. 2,130,819", "PKR 656,553", "2,252,132", and bare digit strings with no
    separators ("391324", the OCR-noised case). Also accepts a value that's already
    numeric (pass-through, for callers that already have a parsed float). Returns None
    for None or an empty/whitespace-only string -- never a substituted number.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _CURRENCY_PREFIX_RE.sub("", value).strip()
    text = text.replace(",", "")
    if not text:
        return None
    return float(text)


def flag_implausible_magnitude(field: str, value: float | None) -> str | None:
    """Return a warning string if `value` falls outside the plausible range for
    `field`, else None. Does not correct the value -- an OCR digit transposition that
    happens to land inside the plausible range is indistinguishable from a real value
    and is deliberately let through; this only catches the ones that don't.
    """
    if value is None:
        return None
    bounds = PLAUSIBLE_RANGES.get(field)
    if bounds is None:
        return None
    lo, hi = bounds
    if lo <= value <= hi:
        return None
    return (
        f"{field}={value:,.0f} is outside the plausible range "
        f"[{lo:,.0f}, {hi:,.0f}] -- possibly OCR digit transposition; not auto-corrected"
    )
