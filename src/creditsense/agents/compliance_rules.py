"""Deterministic compliance rules (P5.4): the numeric ceilings that must be checked in
plain Python, never left to retrieval luck or LLM arithmetic.

Figures are quoted verbatim from the corpus:
  - R-5 (sbp_prudential_sme_regulations.md): "Micro and Small Enterprise can avail
    facilities ... of up to PKR 100 million, and Medium Enterprise (ME) up to
    PKR 500 million from a single bank or from all banks/DFIs/MFBs."
  - R-9 (same): "the clean exposure against an SME shall not exceed PKR 50 million."
  - P-2 (internal_credit_policy.md): "Branch Managers may approve SME facilities up to
    PKR 5 million ... Regional Credit Committees approve up to PKR 50 million.
    Facilities above PKR 50 million, or any facility approaching the per-party exposure
    ceiling in Regulation R-5, require Head Office Credit Committee approval regardless
    of amount."
  - P-7/P-8/P-9 (same): Bank-Statement-Only tier caps at PKR 3 million; Unaudited
    Financials at PKR 30 million; Audited Financials has no tier cap (still subject to
    R-5).
  - P-1 (same): "maximum single-sector concentration of 25% of the SME book."

Known inconsistency, carried rather than papered over: SBP_TIER_TURNOVER_BREAKS in
data/generators/portfolio.py (SE 30-150M, ME 150-800M turnover) predates the real SBP
circular's current breakpoints (Small up to 400M, Medium up to 2,000M) -- logged as an
Open Risk in the tracker. These rules key off the tier *label* already stored on the
applicant (Micro/SE/ME), which is internally consistent with how the dataset and model
were built, but that label's derivation is stale against current regulation. Fixing it
means regenerating the Phase 1 dataset and retraining -- out of scope for Phase 5.
"""

from __future__ import annotations

from creditsense.agents.schemas import LoanApplication, ParsedFinancials

R5_PER_PARTY_CEILING_PKR = {"Micro": 100_000_000.0, "SE": 100_000_000.0, "ME": 500_000_000.0}
R9_CLEAN_FACILITY_CEILING_PKR = 50_000_000.0
P2_BRANCH_MANAGER_CEILING_PKR = 5_000_000.0
P2_REGIONAL_COMMITTEE_CEILING_PKR = 50_000_000.0
TIER_DOCUMENTATION_CAPS_PKR = {
    "Bank-Statement-Only": 3_000_000.0,
    "Unaudited Financials": 30_000_000.0,
    # "Audited Financials": no tier cap -- still subject to R-5, checked separately.
}
P1_SECTOR_CONCENTRATION_CEILING_PCT = 25.0


class RuleFlag:
    """Plain container -- compliance.py turns this into a schemas.ComplianceFlag once
    it has attached a real citation looked up from the corpus."""

    def __init__(self, rule_id: str, severity: str, summary: str) -> None:
        self.rule_id = rule_id
        self.severity = severity
        self.summary = summary


def check_r5_per_party_exposure(
    parsed: ParsedFinancials, proposed_amount: float
) -> RuleFlag | None:
    tier = parsed.fields.get("sbp_enterprise_tier")
    ceiling = R5_PER_PARTY_CEILING_PKR.get(tier)
    if ceiling is None:
        return None
    existing = float(parsed.fields.get("existing_loan_exposure_pkr", 0) or 0)
    group = float(parsed.fields.get("group_associate_exposure_pkr", 0) or 0)
    total = existing + group + proposed_amount
    if total > ceiling:
        return RuleFlag(
            "R-5", "BREACH",
            f"Total exposure (existing {existing:,.0f} + group {group:,.0f} + proposed "
            f"{proposed_amount:,.0f} = {total:,.0f} PKR) exceeds the R-5 per-party "
            f"ceiling of {ceiling:,.0f} PKR for tier {tier}.",
        )
    return None


def check_r9_clean_facility(
    application: LoanApplication, proposed_amount: float
) -> RuleFlag | None:
    if not application.is_clean_facility:
        return None
    if proposed_amount > R9_CLEAN_FACILITY_CEILING_PKR:
        return RuleFlag(
            "R-9", "BREACH",
            f"Proposed clean (unsecured) facility of {proposed_amount:,.0f} PKR "
            f"exceeds the R-9 clean-exposure ceiling of "
            f"{R9_CLEAN_FACILITY_CEILING_PKR:,.0f} PKR.",
        )
    return None


def check_p2_approval_authority(proposed_amount: float, r5_ceiling: float | None) -> RuleFlag | None:
    """Advisory, not a breach on its own -- names which authority must approve. Flags
    only when the amount is above Branch Manager level, since that's the point at
    which an omitted escalation would actually be a compliance problem."""
    if proposed_amount <= P2_BRANCH_MANAGER_CEILING_PKR:
        return None
    approaching_r5 = r5_ceiling is not None and proposed_amount >= 0.9 * r5_ceiling
    if proposed_amount > P2_REGIONAL_COMMITTEE_CEILING_PKR or approaching_r5:
        return RuleFlag(
            "P-2", "ADVISORY",
            f"Proposed amount {proposed_amount:,.0f} PKR requires Head Office Credit "
            f"Committee approval (above 50,000,000 PKR or approaching the R-5 ceiling), "
            f"regardless of any other delegated authority.",
        )
    return RuleFlag(
        "P-2", "ADVISORY",
        f"Proposed amount {proposed_amount:,.0f} PKR requires Regional Credit "
        f"Committee approval (above the 5,000,000 PKR Branch Manager limit).",
    )


def check_documentation_tier_cap(
    parsed: ParsedFinancials, proposed_amount: float
) -> RuleFlag | None:
    doc_tier = parsed.fields.get("documentation_tier")
    cap = TIER_DOCUMENTATION_CAPS_PKR.get(doc_tier)
    if cap is None:  # Audited Financials, or unknown tier -- no tier cap here
        return None
    if proposed_amount > cap:
        rule_id = "P-7" if doc_tier == "Bank-Statement-Only" else "P-8"
        return RuleFlag(
            rule_id, "BREACH",
            f"Proposed amount {proposed_amount:,.0f} PKR exceeds the {cap:,.0f} PKR "
            f"cap for documentation tier '{doc_tier}'.",
        )
    return None


def check_p1_sector_concentration(sector_pct_of_book: float | None) -> RuleFlag | None:
    if sector_pct_of_book is None:
        return None
    if sector_pct_of_book > P1_SECTOR_CONCENTRATION_CEILING_PCT:
        return RuleFlag(
            "P-1", "BREACH",
            f"This sector already represents {sector_pct_of_book:.2f}% of the SME "
            f"book, above the P-1 appetite limit of "
            f"{P1_SECTOR_CONCENTRATION_CEILING_PCT:.0f}%.",
        )
    return None


def run_all(
    application: LoanApplication,
    parsed: ParsedFinancials,
    *,
    proposed_amount: float,
    sector_pct_of_book: float | None,
) -> list[RuleFlag]:
    """Run every deterministic rule and return whichever flags fired. `proposed_amount`
    is resolved by the caller (compliance.py) -- the requested amount if given, else the
    model's recommended credit limit."""
    tier = parsed.fields.get("sbp_enterprise_tier")
    r5_ceiling = R5_PER_PARTY_CEILING_PKR.get(tier)

    checks = [
        check_r5_per_party_exposure(parsed, proposed_amount),
        check_r9_clean_facility(application, proposed_amount),
        check_p2_approval_authority(proposed_amount, r5_ceiling),
        check_documentation_tier_cap(parsed, proposed_amount),
        check_p1_sector_concentration(sector_pct_of_book),
    ]
    return [flag for flag in checks if flag is not None]
