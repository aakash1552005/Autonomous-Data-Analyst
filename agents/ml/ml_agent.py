"""
agents/ml/ml_agent.py
=====================
Agent 4: Machine Learning Agent.
Performs deterministic task detection, target validation, leakage prevention,
leakage-safe preprocessing, baseline comparison, candidate model training,
model selection, pipeline artifact persistence, and hashing.
Strict zero-LLM decision making and DIO boundary isolation.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import time
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core.base_agent import BaseAgent, ProgressState
from core.config import AppConfig, load_config
from core.dio import DIO
from core.persistence import resolve_run_path
from agents.ml.task_detector import detect_ml_task
from agents.ml.target_validator import validate_target_and_data
from agents.ml.leakage_detector import detect_and_exclude_leakage
from agents.ml.preprocessor import build_preprocessing_pipeline
from agents.ml.model_trainer import train_and_evaluate_models
from agents.ml.model_persister import save_and_verify_model_artifact


class MLAgent(BaseAgent):
    """
    Agent 4: Machine Learning Agent.
    """
    name = "ml"

    def __init__(self, config: AppConfig | None = None) -> None:
        super().__init__()
        self.config = config or load_config()

    def run(
        self,
        df: pd.DataFrame,
        dio: DIO | dict[str, Any],
        run_dir: Path | str | None = None,
        preferred_target: str | None = None,
    ) -> tuple[pd.DataFrame, DIO | dict[str, Any]]:
        """
        Execute ML pipeline.
        Consumes cleaned DataFrame + DIO.
        Mutates ONLY dio['ml'] and dio['artifacts']['model_pkl'].
        """
        start_time = time.time()
        dio["progress"][self.name] = ProgressState.RUNNING.value
        warnings_count = 0
        errors_count = 0

        try:
            # 1. Determine Artifact Directory
            if run_dir is not None:
                target_dir = Path(run_dir).resolve()
            else:
                base_runs = Path("runs").resolve()
                ds_id = str(dio.get("dataset_id", "default_run"))
                target_dir = resolve_run_path(base_runs, ds_id)

            target_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir = target_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            columns_info = dio.get("columns", [])

            # 2. Deterministic Task Detection & Target Selection
            task_info = detect_ml_task(df, columns_info=columns_info, preferred_target=preferred_target)
            task_type = task_info["task_type"]
            target_col = task_info.get("target_column")

            if task_type == "none" or target_col is None:
                dio["ml"] = {
                    "status": "unsupported",
                    "task_type": "none",
                    "target_column": None,
                    "reason": task_info["reason"],
                    "target_validation": {},
                    "feature_columns": [],
                    "excluded_columns": [],
                    "split": {},
                    "preprocessing": {},
                    "baseline": {},
                    "candidate_models": [],
                    "selected_model": None,
                    "metrics": {},
                    "reproducibility": {},
                    "leakage_checks": [],
                }
                dio["progress"][self.name] = ProgressState.FINISHED.value
                return df, dio

            # 3. Target Validation and Data Sufficiency Checks
            min_rows = self.config.ml.min_rows_for_ml
            min_per_class = self.config.ml.min_rows_per_class
            is_valid_target, target_val_details, error_code = validate_target_and_data(
                df=df,
                target_col=target_col,
                task_type=task_type,
                min_rows_for_ml=min_rows,
                min_rows_per_class=min_per_class,
            )

            if not is_valid_target:
                dio["ml"] = {
                    "status": "insufficient_data" if error_code == "ML_003_INSUFFICIENT_DATA" else "invalid_target",
                    "error_code": error_code,
                    "task_type": task_type,
                    "target_column": target_col,
                    "target_validation": target_val_details,
                    "reason": target_val_details.get("reason", "Target validation failed"),
                    "feature_columns": [],
                    "excluded_columns": [],
                    "split": {},
                    "preprocessing": {},
                    "baseline": {},
                    "candidate_models": [],
                    "selected_model": None,
                    "metrics": {},
                    "reproducibility": {},
                    "leakage_checks": [],
                }
                if error_code:
                    dio["errors"].append({
                        "agent": self.name,
                        "code": error_code,
                        "message": target_val_details.get("reason", "Validation failed"),
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    })
                dio["progress"][self.name] = ProgressState.FINISHED.value
                return df, dio

            # 4. Leakage Prevention and Feature Filtering
            leakage_res = detect_and_exclude_leakage(
                df=df,
                target_col=target_col,
                columns_info=columns_info,
                leakage_threshold=self.config.ml.leakage_threshold,
            )

            usable_features = leakage_res["usable_features"]
            excluded_features = leakage_res["excluded_features"]
            leakage_checks = leakage_res["leakage_checks"]

            if not usable_features:
                dio["ml"] = {
                    "status": "leakage_detected",
                    "error_code": "ML_004_LEAKAGE_DETECTED",
                    "task_type": task_type,
                    "target_column": target_col,
                    "target_validation": target_val_details,
                    "reason": "All candidate features were excluded due to leakage/PII/identifier constraints",
                    "feature_columns": [],
                    "excluded_columns": excluded_features,
                    "split": {},
                    "preprocessing": {},
                    "baseline": {},
                    "candidate_models": [],
                    "selected_model": None,
                    "metrics": {},
                    "reproducibility": {},
                    "leakage_checks": leakage_checks,
                }
                dio["errors"].append({
                    "agent": self.name,
                    "code": "ML_004_LEAKAGE_DETECTED",
                    "message": "Zero usable features remaining after leakage exclusion",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })
                dio["progress"][self.name] = ProgressState.FINISHED.value
                return df, dio

            # 5. Train/Test Splitting (Pre-processing fitting occurs strictly after split)
            clean_ml_df = df.dropna(subset=[target_col]).copy()
            X = clean_ml_df[usable_features]
            y = clean_ml_df[target_col]

            test_size = 1.0 - self.config.ml.train_test_split
            stratify = y if (task_type == "classification" and target_val_details.get("min_class_count", 0) >= 2) else None

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=self.config.ml.random_state,
                stratify=stratify,
            )

            split_info = {
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "train_pct": round(self.config.ml.train_test_split * 100.0, 1),
                "test_pct": round(test_size * 100.0, 1),
                "stratified": stratify is not None,
                "random_state": self.config.ml.random_state,
            }

            # 6. Build Preprocessing Pipeline
            preprocessor, num_feats, cat_feats = build_preprocessing_pipeline(
                df=clean_ml_df,
                feature_cols=usable_features,
                columns_info=columns_info,
            )

            preprocessing_info = {
                "numeric_features": num_feats,
                "categorical_features": cat_feats,
                "numeric_transformer": "SimpleImputer(median) + StandardScaler",
                "categorical_transformer": "SimpleImputer(Unknown) + OneHotEncoder(ignore)",
            }

            # 7. Model Training & Selection
            train_results = train_and_evaluate_models(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                task_type=task_type,
                preprocessor=preprocessor,
                random_seed=self.config.ml.random_state,
                min_rows_xgboost=self.config.ml.min_rows_xgboost,
            )

            # 8. Persist Model Artifact & Compute SHA-256
            model_pkl_path, model_hash = save_and_verify_model_artifact(
                pipeline=train_results["selected_pipeline"],
                output_dir=artifacts_dir,
                sample_X=X_test,
            )

            # 9. Populate DIO ML and Artifacts
            dio["artifacts"]["model_pkl"] = model_pkl_path

            dio["ml"] = {
                "status": "trained",
                "task_type": task_type,
                "target_column": target_col,
                "target_reason": task_info.get("target_reason", ""),
                "target_validation": target_val_details,
                "feature_columns": usable_features,
                "excluded_columns": excluded_features,
                "split": split_info,
                "preprocessing": preprocessing_info,
                "baseline": train_results["baseline_metrics"],
                "candidate_models": train_results["candidate_models"],
                "selected_model": train_results["selected_model"],
                "selection_reason": train_results["selection_reason"],
                "improved_over_baseline": train_results["improved_over_baseline"],
                "metrics": train_results["metrics"],
                "feature_importance": train_results["feature_importance"],
                "model_artifact": {
                    "path": model_pkl_path,
                    "sha256": model_hash,
                },
                "reproducibility": {
                    "random_seed": self.config.ml.random_state,
                    "split_strategy": "stratified" if stratify is not None else "random",
                    "train_test_split": self.config.ml.train_test_split,
                    "dataset_hash": dio.dataset_hash,
                },
                "leakage_checks": leakage_checks,
            }

            # 10. Record Provenance
            dio["decision_log"].append({
                "agent": self.name,
                "action": "ml_training_completed",
                "task_type": task_type,
                "target_column": target_col,
                "selected_model": train_results["selected_model"],
                "model_sha256": model_hash,
                "improved_over_baseline": train_results["improved_over_baseline"],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })

            dio["progress"][self.name] = ProgressState.FINISHED.value

        except Exception as e:
            errors_count += 1
            dio["progress"][self.name] = ProgressState.FAILED.value
            dio["errors"].append({
                "agent": self.name,
                "code": "ML_005_TRAINING_FAILED",
                "message": f"ML agent failed: {str(e)}",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            raise

        finally:
            runtime_seconds = round(time.time() - start_time, 3)
            dio["agent_metrics"].append({
                "agent": self.name,
                "runtime_seconds": runtime_seconds,
                "warnings": warnings_count,
                "errors": errors_count,
            })

        return df, dio
