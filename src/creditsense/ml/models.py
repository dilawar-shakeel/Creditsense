from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier, XGBRegressor

from creditsense.ml.features import (
    Stage1Preprocessor,
    build_monotone_constraints,
    build_stage2_features,
)


RANDOM_STATE = 42


def get_raw_xgb_model(calibrated_model: CalibratedClassifierCV) -> XGBClassifier:
    calibrated = calibrated_model.calibrated_classifiers_[0]
    estimator = calibrated.estimator
    if isinstance(estimator, FrozenEstimator):
        estimator = estimator.estimator
    return estimator


def explain_decision(
    calibrated_model: CalibratedClassifierCV,
    row: pd.DataFrame,
    top_n: int = 3,
) -> str:
    """Return top SHAP drivers; SHAP is loaded only when explanations are requested."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP explanations require the optional 'shap' package."
        ) from exc

    raw_model = get_raw_xgb_model(calibrated_model)
    shap_values = shap.TreeExplainer(raw_model).shap_values(row)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    contributions = pd.Series(np.ravel(shap_values), index=row.columns)
    contributions = contributions.reindex(
        contributions.abs().sort_values(ascending=False).index
    )
    lines = []
    for feature, contribution in contributions.head(top_n).items():
        direction = "pushed risk up" if contribution > 0 else "pulled risk down"
        value = row[feature].iloc[0]
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        lines.append(f"- {feature} = {value} ({direction})")
    return "Main reasons behind this score:\n" + "\n".join(lines)


@dataclass
class CreditSenseBundle:
    stage1_preprocessor: Stage1Preprocessor
    stage1_model: CalibratedClassifierCV
    stage1_threshold: float
    stage2_tier_encoder: Any
    stage2_model: XGBRegressor
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)
        metadata_path = Path(path).with_suffix(".metadata.json")
        metadata_path.write_text(
            json.dumps(self.metadata, indent=2, default=str), encoding="utf-8"
        )

    @staticmethod
    def load(path: str | Path) -> "CreditSenseBundle":
        return joblib.load(path)

    def score_applicant(self, raw_row: pd.DataFrame) -> dict[str, Any]:
        if len(raw_row) != 1:
            raise ValueError("score_applicant expects exactly one applicant row.")
        stage1_frame = self.stage1_preprocessor.transform(raw_row)
        probability = float(self.stage1_model.predict_proba(stage1_frame)[0, 1])
        decision = "DECLINE" if probability >= self.stage1_threshold else "REFER_FOR_LIMIT"
        result: dict[str, Any] = {
            "default_probability": probability,
            "decision_cutoff": self.stage1_threshold,
            "decision": decision,
            "explanation": explain_decision(self.stage1_model, stage1_frame),
        }
        if decision == "REFER_FOR_LIMIT":
            stage2_frame, _ = build_stage2_features(
                stage1_frame,
                raw_row,
                np.array([probability]),
                self.stage2_tier_encoder,
            )
            columns = self.metadata["stage2_columns"]
            result["recommended_credit_limit_pkr"] = float(
                self.stage2_model.predict(stage2_frame.reindex(columns=columns, fill_value=0))[0]
            )
        return result


def stage2_monotone_constraints(columns: list[str]) -> tuple[int, ...]:
    directions = dict(zip(columns, build_monotone_constraints(columns)))
    directions["annual_bank_turnover_pkr"] = 1
    directions["stage1_predicted_pd"] = -1
    return tuple(directions[column] for column in columns)


def build_stage2_regressor(columns: list[str]) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method="hist",
        monotone_constraints=stage2_monotone_constraints(columns),
        random_state=RANDOM_STATE,
    )
