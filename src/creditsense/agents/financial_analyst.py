"""FinancialAnalystAgent (P5.2): parse an uploaded applicant document, tolerate the
mess in it, and merge it with the bank's own held data for the applicant into the 18
fields the ML API needs.

A real uploaded document never carries everything the model needs -- an ECIB score and
12-month late-payment history are the bank's own records, not something a customer's
bank statement contains. So this agent merges two sources:

  - the 6 "document" fields (current_ratio, debt_to_equity_ratio,
    existing_loan_exposure_pkr, collateral_coverage_ratio, annual_bank_turnover_pkr,
    years_in_business), parsed from `application.raw_fields` with `parsing.py`. These
    are never backfilled from the applicant record even if missing -- a document that
    omits collateral stays unresolved, it does not silently become the bank's own
    on-file figure for a different point in time.
  - the remaining 11 fields (documentation_tier, ecib_score, kibor_sensitivity_pct,
    late_payment_count_12m, bank_statement_volatility, working_capital_cycle_days,
    utility_default_count_12m, revenue_growth_yoy_pct, guarantor_net_worth_pkr,
    group_associate_exposure_pkr, sbp_enterprise_tier), looked up from the applicant's
    `raw_profile_json` in Postgres (loaded by P2.3's seed script).

`sector_risk_code` is treated as document-known first, falling back to the applicant
record's `sector` column -- the one field that's redundant across both sources.

Numbers are never touched by an LLM (see parsing.py's docstring for why). The only LLM
call here is for the free-text Urdu-English notes field, and its failure never blocks
scoring.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from creditsense.agents.llm import complete_structured
from creditsense.agents.parsing import flag_implausible_magnitude, parse_pkr_amount
from creditsense.agents.schemas import LoanApplication, NoteInsights, ParsedFinancials
from creditsense.data.generators.portfolio import SBP_TIER_TURNOVER_BREAKS
from creditsense.db.models import Applicant
from creditsense.ml.features import RAW_FEATURES_STAGE1, STAGE2_EXTRA_FEATURES

# The 6 fields a document is expected to carry itself. If the document doesn't have one,
# it goes to unresolved -- never silently pulled from the applicant record instead.
DOCUMENT_FIELDS = (
    "current_ratio",
    "debt_to_equity_ratio",
    "existing_loan_exposure_pkr",
    "collateral_coverage_ratio",
    "annual_bank_turnover_pkr",
    "years_in_business",
)
_PKR_FIELDS = {"existing_loan_exposure_pkr", "annual_bank_turnover_pkr"}

ALL_REQUIRED_FIELDS = tuple(RAW_FEATURES_STAGE1) + tuple(STAGE2_EXTRA_FEATURES)
# Everything the ML API needs that isn't one of the 6 document fields or sector_risk_code
# (handled specially -- see module docstring) comes from the applicant's bank record.
_RECORD_FIELDS = tuple(
    f for f in ALL_REQUIRED_FIELDS if f not in DOCUMENT_FIELDS and f != "sector_risk_code"
)

_NOTES_SYSTEM_PROMPT = (
    "You read short credit-officer notes about an SME loan applicant, often mixing "
    "Urdu and English. Extract whether the business's cash flow is seasonal, when its "
    "peak is (if stated), and any concerns the officer raised. Never invent a number "
    "or a figure that isn't stated in the text."
)


def _assign_tier(turnover: float) -> str:
    for tier, (lo, hi) in SBP_TIER_TURNOVER_BREAKS.items():
        if lo <= turnover < hi:
            return tier
    return "ME"


def _fetch_applicant(session: Session, applicant_id: str) -> Applicant | None:
    """Isolated so tests can patch this one function rather than fake a SQLAlchemy
    Session's `execute(select(...))` machinery."""
    return session.execute(
        select(Applicant).where(Applicant.applicant_id == applicant_id)
    ).scalar_one_or_none()


def analyze(application: LoanApplication, session: Session) -> ParsedFinancials:
    fields: dict[str, float | int | str] = {}
    field_sources: dict[str, str] = {}
    unresolved: list[str] = []
    warnings: list[str] = []

    # 1. Document fields.
    for name in DOCUMENT_FIELDS:
        raw_value = application.raw_fields.get(name)
        value = parse_pkr_amount(raw_value) if name in _PKR_FIELDS else raw_value
        if value is None:
            unresolved.append(name)
            continue
        warning = flag_implausible_magnitude(name, float(value))
        if warning:
            warnings.append(warning)
        fields[name] = value
        field_sources[name] = "document"

    # 2. sector_risk_code: document-known first, applicant record as fallback.
    applicant = _fetch_applicant(session, application.applicant_id)
    profile = (applicant.raw_profile_json or {}) if applicant else {}

    sector = application.raw_fields.get("sector_risk_code")
    if sector:
        fields["sector_risk_code"] = sector
        field_sources["sector_risk_code"] = "document"
    elif applicant is not None and applicant.sector:
        fields["sector_risk_code"] = applicant.sector
        field_sources["sector_risk_code"] = "applicant_record"
    else:
        unresolved.append("sector_risk_code")

    # 3. The 11 bank-held fields, from the applicant's stored profile.
    for name in _RECORD_FIELDS:
        value = profile.get(name)
        if value is None:
            unresolved.append(name)
            continue
        fields[name] = value
        field_sources[name] = "applicant_record"

    # 4. Derive sbp_enterprise_tier if it's still missing but turnover is known.
    if "sbp_enterprise_tier" in unresolved and "annual_bank_turnover_pkr" in fields:
        unresolved.remove("sbp_enterprise_tier")
        fields["sbp_enterprise_tier"] = _assign_tier(float(fields["annual_bank_turnover_pkr"]))
        field_sources["sbp_enterprise_tier"] = "derived"

    # 5. Free-text notes -> LLM, isolated from every numeric field above.
    note_insights: NoteInsights | None = None
    if application.notes:
        note_insights = complete_structured(
            _NOTES_SYSTEM_PROMPT, application.notes, NoteInsights, fallback=None
        )
        if note_insights is None:
            warnings.append("Note parsing failed or was unavailable; notes not summarized.")

    return ParsedFinancials(
        applicant_id=application.applicant_id,
        fields=fields,
        field_sources=field_sources,
        unresolved_fields=unresolved,
        note_insights=note_insights,
        parse_warnings=warnings,
    )
