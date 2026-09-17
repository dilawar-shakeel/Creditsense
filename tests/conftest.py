"""Shared fixtures for the agents test suite (Phase 5). This is the first root-level
conftest.py in the project -- tests/synthetic_data/conftest.py is scoped to that
subfolder only and does not apply here.

No fixture here talks to a live database or a live OpenAI API -- matching the rest of
the suite, which does neither today.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from creditsense.agents.schemas import LoanApplication


class FakeApplicant(SimpleNamespace):
    """Stand-in for creditsense.db.models.Applicant with only the attributes the
    agents actually read: applicant_id, sector, raw_profile_json."""


@pytest.fixture
def fake_session():
    """A minimal object satisfying just enough of the Session surface used directly
    (session.add/commit in pipeline.py). Functions that need a query result (applicant
    lookups, sector exposure) are patched at their own module-level seam instead of
    faked here -- see test_financial_analyst.py and test_compliance_agent.py."""
    return SimpleNamespace(added=[], add=lambda obj: None, commit=lambda: None)


@pytest.fixture
def clean_applicant():
    """A fully-populated applicant record with every one of the 11 bank-held fields
    a document never carries."""
    return FakeApplicant(
        applicant_id="SME-000001",
        sector="Retail/Trade",
        raw_profile_json={
            "documentation_tier": "Audited Financials",
            "ecib_score": 720.0,
            "kibor_sensitivity_pct": 0.3,
            "late_payment_count_12m": 0,
            "bank_statement_volatility": 0.4,
            "working_capital_cycle_days": 35.0,
            "utility_default_count_12m": 0,
            "revenue_growth_yoy_pct": 6.5,
            "guarantor_net_worth_pkr": 15_000_000.0,
            "group_associate_exposure_pkr": 0.0,
            "sbp_enterprise_tier": "SE",
        },
    )


@pytest.fixture
def clean_document_application():
    """A LoanApplication whose document carries all 6 document fields cleanly (no
    noise) for applicant SME-000001."""
    return LoanApplication(
        applicant_id="SME-000001",
        raw_fields={
            "current_ratio": 1.5,
            "debt_to_equity_ratio": 0.8,
            "existing_loan_exposure_pkr": "2,000,000",
            "collateral_coverage_ratio": 0.9,
            "annual_bank_turnover_pkr": "40,000,000",
            "years_in_business": 8.0,
            "sector_risk_code": "Retail/Trade",
        },
        requested_amount_pkr=2_000_000.0,
    )
