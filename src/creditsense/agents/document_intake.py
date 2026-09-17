"""PDF intake for a brand-new applicant's document (P9.1 follow-on): reads the fields
a credit officer either filled into `creditsense_intake_template.pdf`
(`scripts/generate_intake_template.py`, `frontend/static/assets/`) or typed into some
other PDF. No LLM anywhere in this module -- the same rule `parsing.py` states for
downstream numeric parsing applies one step earlier here too: a value this module
can't confidently find is reported unresolved, never guessed or "corrected".

Two extraction paths, in order:
  1. Form fields (`extraction_method="form_fields"`) -- the reliable path. If the
     upload has our named AcroForm fields (a filled copy of the real template), each
     is read directly by name. Deterministic: no scanning, no guessing.
  2. Text scan (`extraction_method="text_scan"`) -- best-effort fallback for a PDF
     with no recognizable form fields (labels retyped into a different tool, or a
     flattened/printed copy). A small `label: value` line scanner against a short
     alias list per field. Explicitly lower-confidence; the caller surfaces
     `extraction_method` so the frontend can flag it as such.

Neither path ever raises on a malformed/unrelated PDF -- worst case every field is
unresolved and the officer fills the form by hand, the same experience as any other
document this system can't fully read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import BytesIO

from pypdf import PdfReader

DOCUMENT_FIELD_NAMES = (
    "current_ratio",
    "debt_to_equity_ratio",
    "existing_loan_exposure_pkr",
    "collateral_coverage_ratio",
    "annual_bank_turnover_pkr",
    "years_in_business",
    "sector_risk_code",
)
# The 10 "bank record" fields financial_analyst.py has always read exclusively from an
# existing Applicant.raw_profile_json -- never from a document. A brand-new applicant
# has no such row yet, so the intake flow collects these too and POSTs them to
# POST /api/applicants (writing a real row) before underwriting runs, rather than
# handing them to run_underwriting() as part of the document itself. sbp_enterprise_tier
# is deliberately excluded: financial_analyst.py always derives it from turnover.
RECORD_FIELD_NAMES = (
    "documentation_tier",
    "ecib_score",
    "kibor_sensitivity_pct",
    "late_payment_count_12m",
    "bank_statement_volatility",
    "working_capital_cycle_days",
    "utility_default_count_12m",
    "revenue_growth_yoy_pct",
    "guarantor_net_worth_pkr",
    "group_associate_exposure_pkr",
)
ALL_FIELD_NAMES = DOCUMENT_FIELD_NAMES + RECORD_FIELD_NAMES + ("notes",)

# Best-effort aliases for the text-scan fallback only -- the form-field path needs
# none of this, since the PDF field names already match ALL_FIELD_NAMES exactly.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "current_ratio": ("current ratio",),
    "debt_to_equity_ratio": ("debt to equity ratio", "debt-to-equity ratio", "debt to equity"),
    "existing_loan_exposure_pkr": ("existing loan exposure",),
    "collateral_coverage_ratio": ("collateral coverage ratio", "collateral coverage"),
    "annual_bank_turnover_pkr": ("annual bank turnover", "bank turnover"),
    "years_in_business": ("years in business",),
    "sector_risk_code": ("sector",),
    "documentation_tier": ("documentation tier",),
    "ecib_score": ("ecib score",),
    "kibor_sensitivity_pct": ("kibor sensitivity",),
    "late_payment_count_12m": ("late payment count", "late-payment count"),
    "bank_statement_volatility": ("bank statement volatility",),
    "working_capital_cycle_days": ("working capital cycle",),
    "utility_default_count_12m": ("utility default count",),
    "revenue_growth_yoy_pct": ("revenue growth",),
    "guarantor_net_worth_pkr": ("guarantor net worth",),
    "group_associate_exposure_pkr": ("group / associate exposure", "group associate exposure"),
    "notes": ("notes", "credit officer's notes", "credit officer notes"),
}

_LINE_RE = re.compile(r"^\s*(?P<label>[^:]{2,60}):\s*(?P<value>.+?)\s*$")


@dataclass
class ParsedDocumentFields:
    fields: dict[str, str] = field(default_factory=dict)
    unresolved_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extraction_method: str = "form_fields"


def _from_form_fields(reader: PdfReader) -> ParsedDocumentFields | None:
    try:
        pdf_fields = reader.get_fields()
    except Exception:
        return None
    if not pdf_fields or not any(name in pdf_fields for name in ALL_FIELD_NAMES):
        return None

    result = ParsedDocumentFields(extraction_method="form_fields")
    for name in ALL_FIELD_NAMES:
        raw_value = pdf_fields.get(name)
        value = (raw_value.get("/V") if raw_value else None) or ""
        value = str(value).strip()
        if value:
            result.fields[name] = value
        else:
            result.unresolved_fields.append(name)
    return result


def _from_text_scan(reader: PdfReader) -> ParsedDocumentFields:
    result = ParsedDocumentFields(extraction_method="text_scan")
    try:
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        text = ""

    lines = text.splitlines()
    for field_name in ALL_FIELD_NAMES:
        aliases = _FIELD_ALIASES[field_name]
        found = None
        for line in lines:
            match = _LINE_RE.match(line)
            if not match:
                continue
            label = match.group("label").strip().lower()
            if any(alias in label for alias in aliases):
                found = match.group("value").strip()
                break
        if found:
            result.fields[field_name] = found
        else:
            result.unresolved_fields.append(field_name)

    if not text.strip():
        result.warnings.append("No extractable text found in this PDF.")
    return result


def parse_uploaded_pdf(file_bytes: bytes) -> ParsedDocumentFields:
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception:
        result = ParsedDocumentFields(extraction_method="text_scan")
        result.unresolved_fields = list(ALL_FIELD_NAMES)
        result.warnings.append("Could not read this file as a PDF.")
        return result

    form_result = _from_form_fields(reader)
    if form_result is not None:
        return form_result

    return _from_text_scan(reader)
