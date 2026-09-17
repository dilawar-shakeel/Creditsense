from creditsense.agents import compliance_rules as rules
from creditsense.agents.schemas import LoanApplication, ParsedFinancials


def _parsed(**overrides) -> ParsedFinancials:
    fields = {
        "sbp_enterprise_tier": "SE",
        "existing_loan_exposure_pkr": 0.0,
        "group_associate_exposure_pkr": 0.0,
        "documentation_tier": "Audited Financials",
        **overrides,
    }
    return ParsedFinancials(
        applicant_id="SME-000001", fields=fields,
        field_sources={k: "document" for k in fields}, unresolved_fields=[],
    )


def _application(**overrides) -> LoanApplication:
    return LoanApplication(applicant_id="SME-000001", raw_fields={}, **overrides)


# --- R-5: Micro/SE ceiling 100,000,000; ME ceiling 500,000,000 -----------------------

def test_r5_micro_and_se_share_the_100m_ceiling_just_under_is_fine():
    parsed = _parsed(sbp_enterprise_tier="SE")
    assert rules.check_r5_per_party_exposure(parsed, 99_999_999.0) is None


def test_r5_micro_and_se_breach_just_over_100m():
    parsed = _parsed(sbp_enterprise_tier="SE")
    flag = rules.check_r5_per_party_exposure(parsed, 100_000_001.0)
    assert flag is not None
    assert flag.rule_id == "R-5"
    assert flag.severity == "BREACH"


def test_r5_me_ceiling_is_500m_not_100m():
    parsed = _parsed(sbp_enterprise_tier="ME")
    assert rules.check_r5_per_party_exposure(parsed, 400_000_000.0) is None
    flag = rules.check_r5_per_party_exposure(parsed, 500_000_001.0)
    assert flag is not None and flag.rule_id == "R-5"


def test_r5_aggregates_existing_and_group_exposure_with_the_proposed_amount():
    parsed = _parsed(
        sbp_enterprise_tier="SE",
        existing_loan_exposure_pkr=60_000_000.0,
        group_associate_exposure_pkr=30_000_000.0,
    )
    # 60M + 30M + 11M = 101M > 100M ceiling
    flag = rules.check_r5_per_party_exposure(parsed, 11_000_000.0)
    assert flag is not None


# --- R-9: clean facility ceiling 50,000,000 -------------------------------------------

def test_r9_only_applies_to_clean_facilities():
    application = _application(is_clean_facility=False)
    assert rules.check_r9_clean_facility(application, 60_000_000.0) is None


def test_r9_flags_a_clean_facility_over_50m():
    application = _application(is_clean_facility=True)
    assert rules.check_r9_clean_facility(application, 50_000_000.0) is None
    flag = rules.check_r9_clean_facility(application, 50_000_001.0)
    assert flag is not None and flag.rule_id == "R-9" and flag.severity == "BREACH"


# --- P-2: approval authority (advisory, not a breach) ---------------------------------

def test_p2_no_flag_at_or_below_branch_manager_ceiling():
    assert rules.check_p2_approval_authority(5_000_000.0, r5_ceiling=100_000_000.0) is None


def test_p2_flags_regional_committee_band():
    flag = rules.check_p2_approval_authority(5_000_001.0, r5_ceiling=100_000_000.0)
    assert flag is not None
    assert "Regional" in flag.summary
    assert flag.severity == "ADVISORY"


def test_p2_flags_head_office_above_50m():
    flag = rules.check_p2_approval_authority(50_000_001.0, r5_ceiling=100_000_000.0)
    assert flag is not None
    assert "Head Office" in flag.summary


def test_p2_flags_head_office_when_approaching_r5_even_under_50m():
    # 91M is under the 50M regional cap but >=90% of a 100M R-5 ceiling.
    flag = rules.check_p2_approval_authority(91_000_000.0, r5_ceiling=100_000_000.0)
    assert flag is not None
    assert "Head Office" in flag.summary


# --- P-7/P-8: documentation tier caps --------------------------------------------------

def test_bank_statement_only_cap_is_3m():
    parsed = _parsed(documentation_tier="Bank-Statement-Only")
    assert rules.check_documentation_tier_cap(parsed, 3_000_000.0) is None
    flag = rules.check_documentation_tier_cap(parsed, 3_000_001.0)
    assert flag is not None and flag.rule_id == "P-7"


def test_unaudited_financials_cap_is_30m():
    parsed = _parsed(documentation_tier="Unaudited Financials")
    assert rules.check_documentation_tier_cap(parsed, 30_000_000.0) is None
    flag = rules.check_documentation_tier_cap(parsed, 30_000_001.0)
    assert flag is not None and flag.rule_id == "P-8"


def test_audited_financials_has_no_tier_cap():
    parsed = _parsed(documentation_tier="Audited Financials")
    assert rules.check_documentation_tier_cap(parsed, 500_000_000.0) is None


# --- P-1: sector concentration ceiling 25% ---------------------------------------------

def test_p1_no_flag_at_or_below_25_pct():
    assert rules.check_p1_sector_concentration(25.0) is None


def test_p1_flags_over_25_pct():
    flag = rules.check_p1_sector_concentration(25.01)
    assert flag is not None and flag.rule_id == "P-1" and flag.severity == "BREACH"


def test_p1_no_flag_when_exposure_unknown():
    assert rules.check_p1_sector_concentration(None) is None


# --- run_all wiring ----------------------------------------------------------------------

def test_run_all_returns_every_flag_that_fires():
    parsed = _parsed(sbp_enterprise_tier="SE", documentation_tier="Bank-Statement-Only")
    application = _application()
    flags = rules.run_all(
        application, parsed, proposed_amount=200_000_000.0, sector_pct_of_book=30.0
    )
    rule_ids = {f.rule_id for f in flags}
    assert "R-5" in rule_ids   # 200M > 100M SE ceiling
    assert "P-7" in rule_ids   # 200M > 3M bank-statement-only cap
    assert "P-1" in rule_ids   # 30% > 25% concentration ceiling


def test_run_all_returns_nothing_for_a_clean_small_facility():
    parsed = _parsed(sbp_enterprise_tier="SE", documentation_tier="Audited Financials")
    application = _application()
    flags = rules.run_all(
        application, parsed, proposed_amount=2_000_000.0, sector_pct_of_book=10.0
    )
    assert flags == []
