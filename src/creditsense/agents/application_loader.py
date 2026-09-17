"""Shared document-loading logic for anything that builds a `LoanApplication` from an
applicant_id alone (P7.1): the worker CLI and the `/applications/underwrite` endpoint.

Moved out of `agents/worker.py` (previously a private `_load_application`) so the new
HTTP route doesn't import a private function from a CLI module, and so there is exactly
one place this logic lives.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from creditsense.agents.schemas import LoanApplication
from creditsense.db.models import Applicant

DEFAULT_FIXTURE_PATH = Path("tests/fixtures/messy_applications.json")


def load_application(
    applicant_id: str,
    session: Session,
    *,
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
    requested_amount_pkr: float | None = None,
    is_clean_facility: bool = False,
    tenor_months: int | None = None,
) -> LoanApplication | None:
    """Load the applicant's document for underwriting.

    Prefers the messy fixture (realistic OCR-noise/currency-format input) when a
    record for this applicant_id exists there; otherwise falls back to the
    applicant's own stored profile. Returns None if neither source has anything for
    this applicant_id -- the caller (worker CLI or endpoint) decides how to react to
    that (worker.py raises; the endpoint returns 404).
    """
    if fixture_path.exists():
        records = json.loads(fixture_path.read_text(encoding="utf-8"))
        record = next((r for r in records if r["applicant_id"] == applicant_id), None)
        if record is not None:
            raw_fields = dict(record["raw_fields"])
            raw_fields.setdefault("sector_risk_code", record.get("sector_risk_code"))
            return LoanApplication(
                applicant_id=applicant_id,
                raw_fields=raw_fields,
                notes=record.get("notes"),
                requested_amount_pkr=requested_amount_pkr,
                is_clean_facility=is_clean_facility,
                tenor_months=tenor_months,
            )

    applicant = session.query(Applicant).filter_by(applicant_id=applicant_id).first()
    if applicant is None:
        return None

    profile = applicant.raw_profile_json or {}
    raw_fields = {
        name: profile.get(name)
        for name in (
            "current_ratio", "debt_to_equity_ratio", "existing_loan_exposure_pkr",
            "collateral_coverage_ratio", "annual_bank_turnover_pkr", "years_in_business",
            "sector_risk_code",
        )
    }
    return LoanApplication(
        applicant_id=applicant_id,
        raw_fields=raw_fields,
        notes=None,
        requested_amount_pkr=requested_amount_pkr,
        is_clean_facility=is_clean_facility,
        tenor_months=tenor_months,
    )
