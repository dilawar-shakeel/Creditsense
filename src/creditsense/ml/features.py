from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

RAW_FEATURES_STAGE1 = [
    "current_ratio",
    "debt_to_equity_ratio",
    "revenue_growth_yoy_pct",
    "sector_risk_code",
    "years_in_business",
    "existing_loan_exposure_pkr",
    "late_payment_count_12m",
    "collateral_coverage_ratio",
    "bank_statement_volatility",
    "ecib_score",
    "kibor_sensitivity_pct",
    "documentation_tier",
    "working_capital_cycle_days",
    "utility_default_count_12m",
]
TARGET_1 = "default_probability_12m"
STAGE2_EXTRA_FEATURES = [
    "annual_bank_turnover_pkr",
    "guarantor_net_worth_pkr",
    "group_associate_exposure_pkr",
    "sbp_enterprise_tier",
]
TARGET_2 = "recommend_credit_limit_pkr"

MONOTONE_DIRECTIONS = {
    "current_ratio": -1,
    "debt_to_equity_ratio": 1,
    "collateral_coverage_ratio": -1,
    "late_payment_count_12m": 1,
    "utility_default_count_12m": 1,
    "existing_loan_exposure_pkr": 1,
    "financial_stress_count": 1,
    "score_adjusted_risk": -1,
}


class Stage1Preprocessor:
    """Fit training-only transformations and reuse them for new applicants."""

    def __init__(self) -> None:
        self.bsv_75_: float | None = None
        self.wcc_75_: float | None = None
        self.onehot_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._fitted = False

    def fit(self, frame: pd.DataFrame) -> "Stage1Preprocessor":
        missing = sorted(set(RAW_FEATURES_STAGE1) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing Stage 1 columns: {missing}")
        self.bsv_75_ = float(frame["bank_statement_volatility"].quantile(0.75))
        self.wcc_75_ = float(frame["working_capital_cycle_days"].quantile(0.75))
        self.onehot_.fit(frame[["documentation_tier"]])
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or self.bsv_75_ is None or self.wcc_75_ is None:
            raise RuntimeError("Call fit() on the training data before transform().")

        missing = sorted(set(RAW_FEATURES_STAGE1) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing Stage 1 columns: {missing}")

        output = frame[RAW_FEATURES_STAGE1].copy()
        output["financial_stress_count"] = (
            (output["late_payment_count_12m"] >= 2).astype(int)
            + (output["bank_statement_volatility"] >= self.bsv_75_).astype(int)
            + (output["working_capital_cycle_days"] >= self.wcc_75_).astype(int)
            + (output["utility_default_count_12m"] >= 2).astype(int)
        )
        output["score_adjusted_risk"] = (
            output["ecib_score"] - output["financial_stress_count"] * 50
        )
        output = output.drop(columns=["ecib_score"])

        encoded = self.onehot_.transform(output[["documentation_tier"]])
        encoded_columns = self.onehot_.get_feature_names_out(["documentation_tier"])
        encoded_frame = pd.DataFrame(
            encoded, columns=encoded_columns, index=output.index
        )
        output = pd.concat(
            [output.drop(columns=["documentation_tier"]), encoded_frame], axis=1
        )
        output["sector_risk_code"] = output["sector_risk_code"].astype("category")
        return output

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


def build_monotone_constraints(columns: Iterable[str]) -> tuple[int, ...]:
    return tuple(MONOTONE_DIRECTIONS.get(column, 0) for column in columns)


def build_stage2_features(
    stage1_processed: pd.DataFrame,
    raw_frame: pd.DataFrame,
    stage1_pd: object,
    onehot_tier_encoder: OneHotEncoder | None = None,
    fit_encoder: bool = False,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """Add Stage 1 probabilities and Stage 2 inputs to a processed frame."""
    output = stage1_processed.copy()
    output["stage1_predicted_pd"] = stage1_pd
    for column in STAGE2_EXTRA_FEATURES[:-1]:
        output[column] = raw_frame[column].to_numpy()

    if onehot_tier_encoder is None:
        onehot_tier_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
    if fit_encoder:
        onehot_tier_encoder.fit(raw_frame[["sbp_enterprise_tier"]])

    encoded = onehot_tier_encoder.transform(raw_frame[["sbp_enterprise_tier"]])
    encoded_columns = onehot_tier_encoder.get_feature_names_out(
        ["sbp_enterprise_tier"]
    )
    encoded_frame = pd.DataFrame(
        encoded, columns=encoded_columns, index=output.index
    )
    return pd.concat([output, encoded_frame], axis=1), onehot_tier_encoder
