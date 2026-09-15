import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from creditsense.ml.features import (
    Stage1Preprocessor,
    build_monotone_constraints,
    build_stage2_features,
)
from creditsense.ml.models import stage2_monotone_constraints
from creditsense.ml.models import explain_decision
from creditsense.ml.train import recommend_threshold


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "current_ratio": [1.0, 2.0, 0.8, 1.5],
            "debt_to_equity_ratio": [2.0, 1.0, 3.0, 2.5],
            "revenue_growth_yoy_pct": [5.0, 10.0, -2.0, 8.0],
            "sector_risk_code": ["Retail/Trade", "IT/Tech Services", "Retail/Trade", "Food Processing"],
            "years_in_business": [5.0, 8.0, 2.0, 12.0],
            "existing_loan_exposure_pkr": [1e6, 2e6, 3e6, 4e6],
            "late_payment_count_12m": [0, 1, 3, 2],
            "collateral_coverage_ratio": [0.5, 1.0, 0.1, 1.5],
            "bank_statement_volatility": [0.5, 0.7, 1.2, 0.9],
            "ecib_score": [700.0, 750.0, np.nan, 600.0],
            "kibor_sensitivity_pct": [0.3, 0.5, 0.2, 0.7],
            "documentation_tier": ["Audited Financials", "Unaudited Financials", "Bank-Statement-Only", "Audited Financials"],
            "working_capital_cycle_days": [40.0, 60.0, 120.0, 80.0],
            "utility_default_count_12m": [0, 1, 3, 2],
            "annual_bank_turnover_pkr": [10e6, 20e6, 30e6, 40e6],
            "guarantor_net_worth_pkr": [20e6, 30e6, 10e6, 50e6],
            "group_associate_exposure_pkr": [0.0, 1e6, 2e6, 0.0],
            "sbp_enterprise_tier": ["Micro", "SE", "SE", "ME"],
        }
    )


def test_preprocessor_learns_training_thresholds_and_stable_columns(sample_frame):
    preprocessor = Stage1Preprocessor().fit(sample_frame)
    transformed = preprocessor.transform(sample_frame)

    assert preprocessor.bsv_75_ == pytest.approx(0.975)
    assert preprocessor.wcc_75_ == pytest.approx(90.0)
    assert "financial_stress_count" in transformed
    assert "score_adjusted_risk" in transformed
    assert "ecib_score" not in transformed
    assert transformed.shape[0] == len(sample_frame)
    assert transformed["sector_risk_code"].dtype.name == "category"


def test_unknown_categories_are_ignored_after_fit(sample_frame):
    preprocessor = Stage1Preprocessor().fit(sample_frame)
    new_frame = sample_frame.iloc[[0]].copy()
    new_frame["documentation_tier"] = "New Tier"
    transformed = preprocessor.transform(new_frame)

    assert transformed.shape[0] == 1


def test_monotonic_constraints_follow_business_directions():
    columns = ["debt_to_equity_ratio", "current_ratio", "sector_risk_code"]
    assert build_monotone_constraints(columns) == (1, -1, 0)


def test_stage2_constraints_make_turnover_positive_and_pd_negative():
    columns = ["annual_bank_turnover_pkr", "stage1_predicted_pd", "sector_risk_code"]
    assert stage2_monotone_constraints(columns) == (1, -1, 0)


def test_stage2_features_add_probability_and_tier_encoding(sample_frame):
    preprocessor = Stage1Preprocessor().fit(sample_frame)
    stage1 = preprocessor.transform(sample_frame)
    features, encoder = build_stage2_features(
        stage1, sample_frame, np.array([0.1, 0.2, 0.3, 0.4]), fit_encoder=True
    )

    assert "stage1_predicted_pd" in features
    assert "sbp_enterprise_tier_Micro" in features
    assert encoder is not None


def test_recommend_threshold_returns_a_valid_cutoff():
    threshold, ks = recommend_threshold(
        pd.Series([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])
    )

    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= ks <= 1.0


def test_shap_explanation_returns_plain_english_reason_codes():
    x_train = pd.DataFrame({"risk_signal": [0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.8, 0.9, 1.0]})
    y_train = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    model = XGBClassifier(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.1,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    model.fit(x_train, y_train)
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(x_train, y_train)

    explanation = explain_decision(calibrated, x_train.iloc[[0]], top_n=1)

    assert explanation.startswith("Main reasons behind this score:")
    assert "risk_signal" in explanation
