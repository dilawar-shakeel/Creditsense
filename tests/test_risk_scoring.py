from unittest.mock import MagicMock, patch

import httpx

from creditsense.agents.risk_scoring import _REQUIRED_FIELDS, score
from creditsense.agents.schemas import ParsedFinancials

VALID_FIELDS = {
    "current_ratio": 1.2, "debt_to_equity_ratio": 2.0, "revenue_growth_yoy_pct": 8.0,
    "sector_risk_code": "Retail/Trade", "years_in_business": 5.0,
    "existing_loan_exposure_pkr": 1_000_000.0, "late_payment_count_12m": 0,
    "collateral_coverage_ratio": 0.5, "bank_statement_volatility": 0.7,
    "ecib_score": 700.0, "kibor_sensitivity_pct": 0.3,
    "documentation_tier": "Audited Financials", "working_capital_cycle_days": 40.0,
    "utility_default_count_12m": 0, "annual_bank_turnover_pkr": 10_000_000.0,
    "guarantor_net_worth_pkr": 20_000_000.0, "group_associate_exposure_pkr": 0.0,
    "sbp_enterprise_tier": "Micro",
}


def _parsed(fields: dict) -> ParsedFinancials:
    return ParsedFinancials(
        applicant_id="SME-000001",
        fields=fields,
        field_sources={k: "document" for k in fields},
        unresolved_fields=[],
    )


def test_all_18_required_fields_are_sent_and_nothing_else():
    parsed = _parsed(VALID_FIELDS)
    fake_response = MagicMock(status_code=200)
    fake_response.json.return_value = {
        "default_probability": 0.05, "decision_cutoff": 0.17,
        "decision": "REFER_FOR_LIMIT", "recommended_credit_limit_pkr": 5_000_000.0,
        "explanation": "Main reasons behind this score:\n- x = 1 (pushed risk up)",
    }
    fake_client = MagicMock()
    fake_client.post.return_value = fake_response

    with patch("creditsense.agents.risk_scoring._get_http_client", return_value=fake_client), \
         patch("creditsense.agents.risk_scoring._build_narrative", side_effect=lambda e: e):
        result = score(parsed)

    assert result is not None
    sent_payload = fake_client.post.call_args.kwargs["json"]
    assert set(sent_payload.keys()) == set(_REQUIRED_FIELDS)
    assert "applicant_id" not in sent_payload


def test_bad_documentation_tier_is_rejected_client_side_without_a_network_call():
    fields = dict(VALID_FIELDS, documentation_tier="Some Typo Tier")
    parsed = _parsed(fields)
    fake_client = MagicMock()
    with patch("creditsense.agents.risk_scoring._get_http_client", return_value=fake_client):
        result = score(parsed)

    assert result is None
    fake_client.post.assert_not_called()


def test_bad_sector_code_is_rejected_client_side():
    fields = dict(VALID_FIELDS, sector_risk_code="Not A Real Sector")
    parsed = _parsed(fields)
    fake_client = MagicMock()
    with patch("creditsense.agents.risk_scoring._get_http_client", return_value=fake_client):
        result = score(parsed)

    assert result is None
    fake_client.post.assert_not_called()


def test_missing_fields_return_none_without_a_network_call():
    incomplete = dict(VALID_FIELDS)
    del incomplete["ecib_score"]
    parsed = _parsed(incomplete)
    fake_client = MagicMock()
    with patch("creditsense.agents.risk_scoring._get_http_client", return_value=fake_client):
        result = score(parsed)

    assert result is None
    fake_client.post.assert_not_called()


def test_decline_response_has_no_credit_limit():
    parsed = _parsed(VALID_FIELDS)
    fake_response = MagicMock(status_code=200)
    fake_response.json.return_value = {
        "default_probability": 0.9, "decision_cutoff": 0.17,
        "decision": "DECLINE", "recommended_credit_limit_pkr": None,
        "explanation": "Main reasons behind this score:\n- x = 1 (pushed risk up)",
    }
    fake_client = MagicMock()
    fake_client.post.return_value = fake_response

    with patch("creditsense.agents.risk_scoring._get_http_client", return_value=fake_client), \
         patch("creditsense.agents.risk_scoring._build_narrative", side_effect=lambda e: e):
        result = score(parsed)

    assert result.model_decision == "DECLINE"
    assert result.recommended_credit_limit_pkr is None


def test_service_unavailable_returns_none_after_retries():
    parsed = _parsed(VALID_FIELDS)
    fake_client = MagicMock()
    fake_client.post.side_effect = httpx.ConnectError("connection refused")

    with patch("creditsense.agents.risk_scoring._get_http_client", return_value=fake_client), \
         patch("creditsense.agents.risk_scoring.time.sleep"):
        result = score(parsed)

    assert result is None
    assert fake_client.post.call_count == 3  # MAX_RETRIES=2 -> 1 initial + 2 retries


def test_timeout_returns_none_after_retries():
    parsed = _parsed(VALID_FIELDS)
    fake_client = MagicMock()
    fake_client.post.side_effect = httpx.TimeoutException("timed out")

    with patch("creditsense.agents.risk_scoring._get_http_client", return_value=fake_client), \
         patch("creditsense.agents.risk_scoring.time.sleep"):
        result = score(parsed)

    assert result is None
