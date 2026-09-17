"""Field-level data masking for MCP tool responses (P6.6).

Two layers, both required -- the first is the visible demo, the second is what
actually matters:

1. **Redact identifiers.** `account_number` and `cnic` (migration 0004) are the only
   identifier-shaped fields anywhere in the applicant record -- everything else is
   ratios, scores, and PKR amounts. Both are SYNTHETIC (deterministically generated
   from applicant_id by db/seed.py, never real), but redaction is applied exactly as
   it would be against real ones.

2. **Default-deny allowlist -- block the answer key.** `raw_profile_json` holds all 43
   generator columns, six of which are the generator's own ground truth
   (`ml/features.py`'s `LEAKAGE_COLUMNS`, imported here rather than retyped): the exact
   default probability, risk score, and recommended limit the XGBoost model is
   supposed to predict. An MCP tool that returns `raw_profile_json` unfiltered hands an
   LLM agent the answer key. `get_applicant_financials` (P6.2) returns only the 18
   model-input features (`RAW_FEATURES_STAGE1` + `STAGE2_EXTRA_FEATURES`) plus the
   masked identifiers -- an allowlist, not a blocklist, so a new generator column
   defaults to *excluded* rather than silently leaking.
"""

from __future__ import annotations

from typing import Any

from creditsense.ml.features import LEAKAGE_COLUMNS, RAW_FEATURES_STAGE1, STAGE2_EXTRA_FEATURES

ALLOWED_PROFILE_FIELDS = frozenset(RAW_FEATURES_STAGE1) | frozenset(STAGE2_EXTRA_FEATURES)

assert not (ALLOWED_PROFILE_FIELDS & LEAKAGE_COLUMNS), (
    "A leakage column ended up in the allowlist -- this would defeat the entire "
    "point of masking. If this fires, ml/features.py's lists changed; fix the "
    "overlap before returning anything from the MCP boundary."
)


def mask_account_number(value: str | None) -> str | None:
    """PK36SCBL0000001123456702 -> PK36****************6702. First 4 and last 4 kept
    so a human can still match a record against a document, the way real banking UIs
    do; the middle -- the actual account digits -- is fully redacted."""
    if not value:
        return value
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def mask_cnic(value: str | None) -> str | None:
    """35202-1234567-1 -> 35202-*******-1. First group (a location code, not
    identifying on its own) and the last digit are kept; the middle is redacted.
    Anything not shaped like a real CNIC (5-7-1 digit groups) is fully masked rather
    than partially -- a shape this code doesn't recognize gets no benefit of the
    doubt."""
    if not value:
        return value
    parts = value.split("-")
    if len(parts) == 3 and [len(p) for p in parts] == [5, 7, 1] and value.replace("-", "").isdigit():
        return f"{parts[0]}-{'*' * len(parts[1])}-{parts[2]}"
    return "*" * len(value)


def allowed_profile_fields(raw_profile_json: dict[str, Any] | None) -> dict[str, Any]:
    """Filters a full generator-row profile down to the 18 model-input features. This
    is the allowlist step -- fields not in ALLOWED_PROFILE_FIELDS are dropped
    unconditionally, including LEAKAGE_COLUMNS and anything not yet known about."""
    if not raw_profile_json:
        return {}
    return {k: v for k, v in raw_profile_json.items() if k in ALLOWED_PROFILE_FIELDS}
