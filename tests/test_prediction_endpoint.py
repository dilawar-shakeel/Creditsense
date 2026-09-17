import pandas as pd
from fastapi.testclient import TestClient

from creditsense.api.main import app
from creditsense.api.predict import get_model_bundle


class FakeBundle:
    def score_applicant(self, row: pd.DataFrame) -> dict:
        assert len(row) == 1
        return {
            "default_probability": 0.12,
            "decision_cutoff": 0.35,
            "decision": "REFER_FOR_LIMIT",
            "recommended_credit_limit_pkr": 1_000_000.0,
            "explanation": "Main reasons behind this score:\n- current_ratio = 1.2 (pulled risk down)",
        }


def request_payload() -> dict:
    return {
        "current_ratio": 1.2,
        "debt_to_equity_ratio": 2.0,
        "revenue_growth_yoy_pct": 8.0,
        "sector_risk_code": "Retail/Trade",
        "years_in_business": 5.0,
        "existing_loan_exposure_pkr": 1_000_000,
        "late_payment_count_12m": 0,
        "collateral_coverage_ratio": 0.5,
        "bank_statement_volatility": 0.7,
        "ecib_score": 700.0,
        "kibor_sensitivity_pct": 0.3,
        "documentation_tier": "Audited Financials",
        "working_capital_cycle_days": 40.0,
        "utility_default_count_12m": 0,
        "annual_bank_turnover_pkr": 10_000_000,
        "guarantor_net_worth_pkr": 20_000_000,
        "group_associate_exposure_pkr": 0,
        "sbp_enterprise_tier": "Micro",
    }


def test_prediction_endpoint_returns_model_result(monkeypatch):
    get_model_bundle.cache_clear()
    monkeypatch.setattr(
        "creditsense.api.predict.get_model_bundle", lambda: FakeBundle()
    )

    response = TestClient(app).post("/predict/credit-risk", json=request_payload())

    assert response.status_code == 200
    assert response.json()["default_probability"] == 0.12
    assert "explanation" in response.json()


def test_prediction_endpoint_rejects_missing_fields():
    response = TestClient(app).post("/predict/credit-risk", json={})

    assert response.status_code == 422


def test_prediction_endpoint_reports_missing_model(monkeypatch):
    monkeypatch.setattr(
        "creditsense.api.predict.get_model_bundle",
        lambda: (_ for _ in ()).throw(FileNotFoundError()),
    )

    response = TestClient(app).post("/predict/credit-risk", json=request_payload())

    assert response.status_code == 503