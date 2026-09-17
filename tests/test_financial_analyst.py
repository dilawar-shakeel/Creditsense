from types import SimpleNamespace
from unittest.mock import patch

from creditsense.agents.financial_analyst import analyze
from creditsense.agents.schemas import LoanApplication, NoteInsights


def test_document_fields_win_over_applicant_record(clean_document_application, clean_applicant, fake_session):
    with patch("creditsense.agents.financial_analyst._fetch_applicant", return_value=clean_applicant):
        parsed = analyze(clean_document_application, fake_session)

    assert parsed.fields["current_ratio"] == 1.5
    assert parsed.fields["existing_loan_exposure_pkr"] == 2_000_000.0
    assert parsed.field_sources["current_ratio"] == "document"
    assert parsed.field_sources["existing_loan_exposure_pkr"] == "document"


def test_bank_held_fields_come_from_the_applicant_record(clean_document_application, clean_applicant, fake_session):
    with patch("creditsense.agents.financial_analyst._fetch_applicant", return_value=clean_applicant):
        parsed = analyze(clean_document_application, fake_session)

    assert parsed.fields["ecib_score"] == 720.0
    assert parsed.field_sources["ecib_score"] == "applicant_record"
    assert parsed.fields["documentation_tier"] == "Audited Financials"
    assert parsed.field_sources["documentation_tier"] == "applicant_record"
    assert parsed.fields["sbp_enterprise_tier"] == "SE"
    assert parsed.field_sources["sbp_enterprise_tier"] == "applicant_record"


def test_all_18_required_fields_are_present_for_a_clean_case(clean_document_application, clean_applicant, fake_session):
    from creditsense.agents.risk_scoring import _REQUIRED_FIELDS

    with patch("creditsense.agents.financial_analyst._fetch_applicant", return_value=clean_applicant):
        parsed = analyze(clean_document_application, fake_session)

    missing = [f for f in _REQUIRED_FIELDS if f not in parsed.fields]
    assert missing == []
    assert parsed.unresolved_fields == []


def test_missing_collateral_in_document_is_unresolved_not_backfilled(clean_applicant, fake_session):
    application = LoanApplication(
        applicant_id="SME-004145",
        raw_fields={
            "current_ratio": 1.43,
            "debt_to_equity_ratio": 1.52,
            "existing_loan_exposure_pkr": "551,450",
            "collateral_coverage_ratio": None,
            "annual_bank_turnover_pkr": "251,073,686",
            "years_in_business": 5.11,
            "sector_risk_code": "Retail/Trade",
        },
    )
    with patch("creditsense.agents.financial_analyst._fetch_applicant", return_value=clean_applicant):
        parsed = analyze(application, fake_session)

    assert "collateral_coverage_ratio" not in parsed.fields
    assert "collateral_coverage_ratio" in parsed.unresolved_fields


def test_missing_applicant_record_leaves_bank_held_fields_unresolved(clean_document_application, fake_session):
    with patch("creditsense.agents.financial_analyst._fetch_applicant", return_value=None):
        parsed = analyze(clean_document_application, fake_session)

    assert "ecib_score" in parsed.unresolved_fields
    assert "documentation_tier" in parsed.unresolved_fields
    # Document fields are still present -- only the bank-held ones are lost.
    assert parsed.fields["current_ratio"] == 1.5


def test_sbp_enterprise_tier_is_derived_when_absent_but_turnover_known(clean_document_application, fake_session):
    applicant_without_tier = SimpleNamespace(
        applicant_id="SME-000001",
        sector="Retail/Trade",
        raw_profile_json={"documentation_tier": "Audited Financials", "ecib_score": 700.0,
                           "kibor_sensitivity_pct": 0.3, "late_payment_count_12m": 0,
                           "bank_statement_volatility": 0.4, "working_capital_cycle_days": 30.0,
                           "utility_default_count_12m": 0, "revenue_growth_yoy_pct": 5.0,
                           "guarantor_net_worth_pkr": 1_000_000.0, "group_associate_exposure_pkr": 0.0},
    )
    with patch("creditsense.agents.financial_analyst._fetch_applicant", return_value=applicant_without_tier):
        parsed = analyze(clean_document_application, fake_session)

    # 40,000,000 falls in the SE bracket (30,000,000 - 150,000,000).
    assert parsed.fields["sbp_enterprise_tier"] == "SE"
    assert parsed.field_sources["sbp_enterprise_tier"] == "derived"


def test_notes_are_summarized_by_the_llm(clean_applicant, fake_session):
    application = LoanApplication(
        applicant_id="SME-000001",
        raw_fields={
            "current_ratio": 1.5, "debt_to_equity_ratio": 0.8,
            "existing_loan_exposure_pkr": "2,000,000", "collateral_coverage_ratio": 0.9,
            "annual_bank_turnover_pkr": "40,000,000", "years_in_business": 8.0,
            "sector_risk_code": "Retail/Trade",
        },
        notes="client ka cash flow seasonal hai, Eid se pehle spike hota hai",
    )
    fake_insights = NoteInsights(seasonality=True, seasonal_peak="pre-Eid")
    with patch("creditsense.agents.financial_analyst._fetch_applicant", return_value=clean_applicant), \
         patch("creditsense.agents.financial_analyst.complete_structured", return_value=fake_insights):
        parsed = analyze(application, fake_session)

    assert parsed.note_insights is not None
    assert parsed.note_insights.seasonality is True


def test_note_llm_failure_does_not_block_the_rest_of_parsing(clean_applicant, fake_session):
    application = LoanApplication(
        applicant_id="SME-000001",
        raw_fields={
            "current_ratio": 1.5, "debt_to_equity_ratio": 0.8,
            "existing_loan_exposure_pkr": "2,000,000", "collateral_coverage_ratio": 0.9,
            "annual_bank_turnover_pkr": "40,000,000", "years_in_business": 8.0,
            "sector_risk_code": "Retail/Trade",
        },
        notes="some note",
    )
    with patch("creditsense.agents.financial_analyst._fetch_applicant", return_value=clean_applicant), \
         patch("creditsense.agents.financial_analyst.complete_structured", return_value=None):
        parsed = analyze(application, fake_session)

    assert parsed.note_insights is None
    assert parsed.fields["current_ratio"] == 1.5
    assert any("note" in w.lower() for w in parsed.parse_warnings)


def test_sector_risk_code_falls_back_to_applicant_record_when_document_lacks_it(clean_applicant, fake_session):
    application = LoanApplication(
        applicant_id="SME-000001",
        raw_fields={
            "current_ratio": 1.5, "debt_to_equity_ratio": 0.8,
            "existing_loan_exposure_pkr": "2,000,000", "collateral_coverage_ratio": 0.9,
            "annual_bank_turnover_pkr": "40,000,000", "years_in_business": 8.0,
        },
    )
    with patch("creditsense.agents.financial_analyst._fetch_applicant", return_value=clean_applicant):
        parsed = analyze(application, fake_session)

    assert parsed.fields["sector_risk_code"] == "Retail/Trade"
    assert parsed.field_sources["sector_risk_code"] == "applicant_record"
