from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from xgboost import XGBClassifier

from creditsense.ml.features import (
    TARGET_1,
    TARGET_2,
    Stage1Preprocessor,
    build_monotone_constraints,
    build_stage2_features,
)
from creditsense.ml.models import (
    RANDOM_STATE,
    CreditSenseBundle,
    build_stage2_regressor,
)

DEFAULT_CSV_PATH = Path("src/creditsense/data/raw/creditsense_synthetic_portfolio.csv")
DEFAULT_OUTPUT_PATH = Path("src/creditsense/ml/artifacts/creditsense_bundle.joblib")


def tune_stage1(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 25,
    cv_folds: int = 5,
) -> tuple[dict[str, Any], float]:
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        enable_categorical=True,
        tree_method="hist",
        monotone_constraints=build_monotone_constraints(list(x_train.columns)),
        random_state=RANDOM_STATE,
    )
    search = RandomizedSearchCV(
        base_model,
        {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [200, 300, 400, 600],
            "min_child_weight": [1, 3, 5, 10],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        },
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(cv_folds, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search.best_params_, scale_pos_weight


def fit_stage1_calibrated(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, Any],
    scale_pos_weight: float,
) -> CalibratedClassifierCV:
    x_core, x_calib, y_core, y_calib = train_test_split(
        x_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    model = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        enable_categorical=True,
        tree_method="hist",
        monotone_constraints=build_monotone_constraints(list(x_train.columns)),
        random_state=RANDOM_STATE,
    )
    model.fit(x_core, y_core)
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(x_calib, y_calib)
    return calibrated


def recommend_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    order = np.argsort(probabilities)
    actual = np.asarray(y_true)[order]
    scores = probabilities[order]
    bad_total = max((actual == 1).sum(), 1)
    good_total = max((actual == 0).sum(), 1)
    separation = np.abs(
        np.cumsum(actual == 1) / bad_total - np.cumsum(actual == 0) / good_total
    )
    index = int(np.argmax(separation))
    return float(scores[index]), float(separation[index])


def generate_oof_calibrated_pd(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, Any],
    scale_pos_weight: float,
    n_splits: int = 5,
) -> np.ndarray:
    """Generate calibrated, out-of-fold Stage 1 probabilities for Stage 2.

    Every row is scored by a calibrated model that was not trained on that row.
    This keeps Stage 2's training input on the same probability scale it receives
    when the API scores a new applicant.
    """
    out_of_fold = np.zeros(len(x_train), dtype=float)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for fit_indices, holdout_indices in folds.split(x_train, y_train):
        x_fit = x_train.iloc[fit_indices]
        y_fit = y_train.iloc[fit_indices]
        x_holdout = x_train.iloc[holdout_indices]
        calibrated_fold = fit_stage1_calibrated(
            x_fit,
            y_fit,
            best_params,
            scale_pos_weight,
        )
        out_of_fold[holdout_indices] = calibrated_fold.predict_proba(x_holdout)[:, 1]

    return out_of_fold


def train_bundle(
    frame: pd.DataFrame,
    best_params: dict[str, Any] | None = None,
    tuning_iterations: int = 25,
    cv_folds: int = 5,
) -> tuple[CreditSenseBundle, dict[str, Any]]:
    train_frame, test_frame = train_test_split(
        frame, test_size=0.2, stratify=frame[TARGET_1], random_state=RANDOM_STATE
    )
    preprocessor = Stage1Preprocessor()
    x_train_1 = preprocessor.fit_transform(train_frame)
    x_test_1 = preprocessor.transform(test_frame)
    y_train_1 = train_frame[TARGET_1]
    y_test_1 = test_frame[TARGET_1]

    if best_params is None:
        best_params, scale_weight = tune_stage1(
            x_train_1, y_train_1, tuning_iterations, cv_folds
        )
    else:
        scale_weight = float((y_train_1 == 0).sum() / (y_train_1 == 1).sum())

    stage1_model = fit_stage1_calibrated(
        x_train_1, y_train_1, best_params, scale_weight
    )
    test_pd = stage1_model.predict_proba(x_test_1)[:, 1]
    threshold, ks_score = recommend_threshold(y_test_1, test_pd)

    train_oof_pd = generate_oof_calibrated_pd(
        x_train_1,
        y_train_1,
        best_params,
        scale_weight,
    )
    x_train_2, tier_encoder = build_stage2_features(
        x_train_1, train_frame, train_oof_pd, fit_encoder=True
    )
    x_test_2, _ = build_stage2_features(
        x_test_1, test_frame, test_pd, tier_encoder
    )
    stage2_model = build_stage2_regressor(list(x_train_2.columns))
    stage2_model.fit(x_train_2, train_frame[TARGET_2])
    stage2_predictions = stage2_model.predict(x_test_2)

    threshold_predictions = (test_pd >= threshold).astype(int)
    threshold_matrix = confusion_matrix(y_test_1, threshold_predictions, labels=[0, 1])
    stage2_actual = test_frame[TARGET_2].to_numpy()
    stage2_errors = stage2_predictions - stage2_actual

    bundle = CreditSenseBundle(
        preprocessor,
        stage1_model,
        threshold,
        tier_encoder,
        stage2_model,
        metadata={"stage2_columns": list(x_train_2.columns), "ks_score": ks_score},
    )
    metrics: dict[str, Any] = {
        "training_rows": int(len(frame)),
        "test_rows": int(len(test_frame)),
        "stage1": {
            "threshold": threshold,
            "pr_auc": float(average_precision_score(y_test_1, test_pd)),
            "roc_auc": float(roc_auc_score(y_test_1, test_pd)),
            "brier_score": float(brier_score_loss(y_test_1, test_pd)),
            "ks_score": ks_score,
            "positive_rate": float(y_test_1.mean()),
            "predicted_positive_rate": float(threshold_predictions.mean()),
            "precision_default": float(
                precision_score(y_test_1, threshold_predictions, pos_label=1, zero_division=0)
            ),
            "recall_default": float(
                recall_score(y_test_1, threshold_predictions, pos_label=1, zero_division=0)
            ),
            "f1_default": float(
                f1_score(y_test_1, threshold_predictions, pos_label=1, zero_division=0)
            ),
            "precision_non_default": float(
                precision_score(y_test_1, threshold_predictions, pos_label=0, zero_division=0)
            ),
            "recall_non_default": float(
                recall_score(y_test_1, threshold_predictions, pos_label=0, zero_division=0)
            ),
            "confusion_matrix_labels": ["non_default", "default"],
            "confusion_matrix": threshold_matrix.tolist(),
        },
        "stage2": {
            "r2": float(r2_score(stage2_actual, stage2_predictions)),
            "explained_variance": float(
                explained_variance_score(stage2_actual, stage2_predictions)
            ),
            "mae_pkr": float(mean_absolute_error(stage2_actual, stage2_predictions)),
            "rmse_pkr": float(mean_squared_error(stage2_actual, stage2_predictions) ** 0.5),
            "median_absolute_error_pkr": float(
                median_absolute_error(stage2_actual, stage2_predictions)
            ),
            "mean_actual_pkr": float(stage2_actual.mean()),
            "mean_predicted_pkr": float(stage2_predictions.mean()),
            "mean_error_pkr": float(stage2_errors.mean()),
        },
    }
    bundle.metadata.update(
        {
            "best_params": best_params,
            "scale_pos_weight": scale_weight,
            "metrics": metrics,
        }
    )
    return bundle, metrics


def main(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    tuning_iterations: int = 25,
    cv_folds: int = 5,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> CreditSenseBundle:
    """Train the complete two-stage model and save the API bundle."""
    frame = pd.read_csv(csv_path)
    bundle, metrics = train_bundle(
        frame,
        tuning_iterations=tuning_iterations,
        cv_folds=cv_folds,
    )
    bundle.save(output_path)
    Path(output_path).with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return bundle


if __name__ == "__main__":
    main()
