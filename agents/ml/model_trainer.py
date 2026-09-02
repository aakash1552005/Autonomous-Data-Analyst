"""
agents/ml/model_trainer.py
==========================
Deterministic model training, baseline establishment, metric evaluation,
and best model selection across classification and regression tasks.
Handles optional XGBoost dependency gracefully.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def evaluate_classification_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_classes: int,
) -> dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    """
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    
    if n_classes == 2:
        # Binary Classification
        prec = float(precision_score(y_test, y_pred, zero_division=0, average="binary"))
        rec = float(recall_score(y_test, y_pred, zero_division=0, average="binary"))
        f1 = float(f1_score(y_test, y_pred, zero_division=0, average="binary"))
        
        roc_auc = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_test)
                if probs.shape[1] == 2:
                    roc_auc = float(roc_auc_score(y_test, probs[:, 1]))
            except Exception:
                roc_auc = None
    else:
        # Multiclass Classification
        prec = float(precision_score(y_test, y_pred, zero_division=0, average="macro"))
        rec = float(recall_score(y_test, y_pred, zero_division=0, average="macro"))
        f1 = float(f1_score(y_test, y_pred, zero_division=0, average="macro"))
        roc_auc = None

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm,
    }


def evaluate_regression_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Compute comprehensive regression metrics.
    """
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(math.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def extract_feature_importances(
    pipeline: Pipeline,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Extract feature importances or coefficient magnitudes from trained pipeline.
    """
    estimator = pipeline.named_steps.get("estimator")
    if estimator is None:
        return {}

    raw_importances = None
    if hasattr(estimator, "feature_importances_"):
        raw_importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        raw_importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)

    if raw_importances is None:
        return {}

    # Try to map back to preprocessed feature names
    preprocessor = pipeline.named_steps.get("preprocessor")
    try:
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            out_names = [n.split("__")[-1] for n in preprocessor.get_feature_names_out()]
        else:
            out_names = feature_names[:len(raw_importances)]
    except Exception:
        out_names = feature_names[:len(raw_importances)]

    importance_dict: dict[str, float] = {}
    for name, imp in zip(out_names, raw_importances):
        importance_dict[str(name)] = round(float(imp), 4)

    # Sort descending by importance
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
    preprocessor: Any,
    random_seed: int = 42,
    min_rows_xgboost: int = 500,
) -> dict[str, Any]:
    """
    Train baselines and candidate models, evaluate on test set, and select best model.
    Enforces min_rows_xgboost threshold and handles optional XGBoost dependency.
    """
    candidate_records: list[dict[str, Any]] = []
    fitted_pipelines: dict[str, Pipeline] = {}
    total_rows = len(X_train) + len(X_test)

    if task_type == "classification":
        n_classes = int(y_train.nunique())

        # 1. Baseline: DummyClassifier
        base_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("estimator", DummyClassifier(strategy="most_frequent")),
        ])
        base_pipe.fit(X_train, y_train)
        base_metrics = evaluate_classification_model(base_pipe, X_test, y_test, n_classes)

        # 2. Candidate 1: Logistic Regression
        lr_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("estimator", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_seed)),
        ])
        lr_pipe.fit(X_train, y_train)
        lr_metrics = evaluate_classification_model(lr_pipe, X_test, y_test, n_classes)
        fitted_pipelines["logistic_regression"] = lr_pipe
        candidate_records.append({
            "model_name": "logistic_regression",
            "status": "trained",
            "metrics": lr_metrics,
        })

        # 3. Candidate 2: Random Forest Classifier
        rf_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("estimator", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_seed)),
        ])
        rf_pipe.fit(X_train, y_train)
        rf_metrics = evaluate_classification_model(rf_pipe, X_test, y_test, n_classes)
        fitted_pipelines["random_forest"] = rf_pipe
        candidate_records.append({
            "model_name": "random_forest",
            "status": "trained",
            "metrics": rf_metrics,
        })

        # 4. Candidate 3: XGBoost Classifier (Optional & Threshold-Guarded)
        if total_rows < min_rows_xgboost:
            candidate_records.append({
                "model_name": "xgboost",
                "status": "skipped",
                "reason": f"dataset has {total_rows} rows, below min_rows_xgboost threshold of {min_rows_xgboost}",
            })
        else:
            try:
                from xgboost import XGBClassifier
                xgb_pipe = Pipeline([
                    ("preprocessor", preprocessor),
                    ("estimator", XGBClassifier(random_state=random_seed, eval_metric="logloss")),
                ])
                xgb_pipe.fit(X_train, y_train)
                xgb_metrics = evaluate_classification_model(xgb_pipe, X_test, y_test, n_classes)
                fitted_pipelines["xgboost"] = xgb_pipe
                candidate_records.append({
                    "model_name": "xgboost",
                    "status": "trained",
                    "metrics": xgb_metrics,
                })
            except Exception as e:
                candidate_records.append({
                    "model_name": "xgboost",
                    "status": "skipped",
                    "reason": f"not available: {str(e)}",
                })

        # Deterministic Selection: Max F1 -> Max Recall -> Max Precision -> Priority
        trained_candidates = [c for c in candidate_records if c["status"] == "trained"]
        selected = max(
            trained_candidates,
            key=lambda c: (c["metrics"]["f1"], c["metrics"]["recall"], c["metrics"]["precision"]),
        )
        selected_name = selected["model_name"]
        selected_pipeline = fitted_pipelines[selected_name]
        selected_metrics = selected["metrics"]
        selection_reason = f"Selected {selected_name} with highest F1 score ({selected_metrics['f1']})"
        improved_over_baseline = selected_metrics["f1"] > base_metrics["f1"]

    else:
        # Regression
        # 1. Baseline: DummyRegressor
        base_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("estimator", DummyRegressor(strategy="mean")),
        ])
        base_pipe.fit(X_train, y_train)
        base_metrics = evaluate_regression_model(base_pipe, X_test, y_test)

        # 2. Candidate 1: Ridge Regression
        ridge_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("estimator", Ridge(random_state=random_seed)),
        ])
        ridge_pipe.fit(X_train, y_train)
        ridge_metrics = evaluate_regression_model(ridge_pipe, X_test, y_test)
        fitted_pipelines["ridge_regression"] = ridge_pipe
        candidate_records.append({
            "model_name": "ridge_regression",
            "status": "trained",
            "metrics": ridge_metrics,
        })

        # 3. Candidate 2: Random Forest Regressor
        rf_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("estimator", RandomForestRegressor(n_estimators=100, random_state=random_seed)),
        ])
        rf_pipe.fit(X_train, y_train)
        rf_metrics = evaluate_regression_model(rf_pipe, X_test, y_test)
        fitted_pipelines["random_forest"] = rf_pipe
        candidate_records.append({
            "model_name": "random_forest",
            "status": "trained",
            "metrics": rf_metrics,
        })

        # 4. Candidate 3: XGBoost Regressor (Optional & Threshold-Guarded)
        if total_rows < min_rows_xgboost:
            candidate_records.append({
                "model_name": "xgboost",
                "status": "skipped",
                "reason": f"dataset has {total_rows} rows, below min_rows_xgboost threshold of {min_rows_xgboost}",
            })
        else:
            try:
                from xgboost import XGBRegressor
                xgb_pipe = Pipeline([
                    ("preprocessor", preprocessor),
                    ("estimator", XGBRegressor(random_state=random_seed)),
                ])
                xgb_pipe.fit(X_train, y_train)
                xgb_metrics = evaluate_regression_model(xgb_pipe, X_test, y_test)
                fitted_pipelines["xgboost"] = xgb_pipe
                candidate_records.append({
                    "model_name": "xgboost",
                    "status": "trained",
                    "metrics": xgb_metrics,
                })
            except Exception as e:
                candidate_records.append({
                    "model_name": "xgboost",
                    "status": "skipped",
                    "reason": f"not available: {str(e)}",
                })

        # Deterministic Selection: Min RMSE -> Min MAE -> Max R2
        trained_candidates = [c for c in candidate_records if c["status"] == "trained"]
        selected = min(
            trained_candidates,
            key=lambda c: (c["metrics"]["rmse"], c["metrics"]["mae"], -c["metrics"]["r2"]),
        )
        selected_name = selected["model_name"]
        selected_pipeline = fitted_pipelines[selected_name]
        selected_metrics = selected["metrics"]
        selection_reason = f"Selected {selected_name} with lowest RMSE ({selected_metrics['rmse']})"
        improved_over_baseline = selected_metrics["rmse"] < base_metrics["rmse"]

    # Extract Feature Importances
    feature_importances = extract_feature_importances(selected_pipeline, list(X_train.columns))

    return {
        "baseline_metrics": base_metrics,
        "candidate_models": candidate_records,
        "selected_model": selected_name,
        "selected_pipeline": selected_pipeline,
        "metrics": selected_metrics,
        "selection_reason": selection_reason,
        "improved_over_baseline": improved_over_baseline,
        "feature_importance": feature_importances,
    }
