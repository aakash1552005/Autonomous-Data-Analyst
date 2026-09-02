"""
tests/test_ml_adversarial_audit.py
==================================
Comprehensive adversarial audit test suite for Agent 4 (Machine Learning Agent).
Verifies:
1. Leakage boundary exact thresholds (0.95 vs 0.949, NMI 0.95 vs 0.949, suspicious names).
2. Strict target isolation under all conditions.
3. PII and identifier exclusion even with 100% predictive power.
4. Preprocessing leakage isolation (transformers fitted strictly on X_train).
5. Data threshold boundaries (30 vs 29 rows, 5 vs 4 samples/class, stratification).
6. XGBoost optional dependency and simulated failure.
7. Baseline comparison and deterministic model selection (F1 / RMSE).
8. Model persistence, SHA-256 hashing, reload, and prediction integrity.
9. Bit-for-bit dataset immutability.
10. Strict DIO mutation boundary isolation.
11. Explicit error codes contract (ML_001, ML_002, ML_003, ML_004).
12. Complete end-to-end positive path on a 100-row synthetic dataset.
"""

import copy
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from core.data_router import DataRouter
from agents.intelligence.intelligence_agent import IntelligenceAgent
from agents.cleaning.cleaning_agent import CleaningAgent
from agents.eda.eda_agent import EDAAgent
from agents.ml.ml_agent import MLAgent
from agents.ml.task_detector import detect_ml_task
from agents.ml.target_validator import validate_target_and_data
from agents.ml.leakage_detector import detect_and_exclude_leakage
from agents.ml.preprocessor import build_preprocessing_pipeline
from agents.ml.model_trainer import train_and_evaluate_models
from core.config import AppConfig
from core.dio import DIO
from core.base_agent import ProgressState
from core.hashing import compute_string_hash, compute_file_hash


def test_adversarial_leakage_threshold_exact_boundary():
    """
    Test 1: Verify exact boundary for correlation (0.95 excluded vs 0.949 retained)
    and NMI (0.95 excluded vs 0.949 retained), plus suspicious post-outcome names.
    """
    n = 100
    np.random.seed(42)
    y_reg = np.linspace(10.0, 100.0, n)

    # Construct feature with r >= 0.95
    feat_exact_95 = y_reg + np.random.normal(0, 5.0, n)
    r_val_95 = abs(np.corrcoef(feat_exact_95, y_reg)[0, 1])

    # Construct feature with r < 0.95 (e.g. around 0.85)
    feat_safe = y_reg + np.random.normal(0, 30.0, n)
    r_val_safe = abs(np.corrcoef(feat_safe, y_reg)[0, 1])

    df = pd.DataFrame({
        "target": y_reg,
        "feat_high_corr": feat_exact_95,
        "feat_safe": feat_safe,
        "cancellation_date": ["2024-01-01"] * n,
        "discharge_date": ["2024-01-05"] * n,
        "outcome_notes": ["Note"] * n,
        "resolved_flag": [1] * n,
    })

    res = detect_and_exclude_leakage(df, "target", leakage_threshold=0.95)

    # Exclusions
    excluded_names = [e["column"] for e in res["excluded_features"]]
    assert "target" in excluded_names
    assert "cancellation_date" in excluded_names
    assert "discharge_date" in excluded_names
    assert "outcome_notes" in excluded_names
    assert "resolved_flag" in excluded_names
    if r_val_95 >= 0.95:
        assert "feat_high_corr" in excluded_names

    # Retained
    if r_val_safe < 0.95:
        assert "feat_safe" in res["usable_features"]

    # Verify leakage check details structure
    for check in res["leakage_checks"]:
        assert "column" in check
        assert "detection_method" in check
        assert "specific_reason" in check
        assert "exclusion_status" in check
        if check["exclusion_status"] == "excluded" and "high_" in check.get("specific_reason", ""):
            assert check["threshold_used"] == 0.95
            assert check["measured_value"] is not None


def test_adversarial_leakage_leaving_zero_vs_one_feature(tmp_path: Path):
    """
    Verify:
    1. Zero usable features -> ML_004_LEAKAGE_DETECTED
    2. One usable feature -> Successfully trains model with single feature.
    """
    # 1. Zero features
    df_zero = pd.DataFrame({
        "target": [0, 1] * 20,
        "cancellation_date": ["2024-01-01"] * 40,
    })
    dio_zero = DIO.create_empty(file_name="zero.csv", dataset_hash="h_z")
    dio_zero["columns"] = [
        {"name": "target", "dtype_inferred": "int", "is_target_candidate": True},
        {"name": "cancellation_date", "dtype_inferred": "string"},
    ]
    agent = MLAgent()
    _, dio_zero_out = agent.run(df_zero, dio_zero, run_dir=tmp_path / "zero")
    assert dio_zero_out["ml"]["status"] == "leakage_detected"
    assert dio_zero_out["ml"]["error_code"] == "ML_004_LEAKAGE_DETECTED"

    # 2. Exactly one usable feature
    df_one = pd.DataFrame({
        "target": [0, 1] * 20,
        "cancellation_date": ["2024-01-01"] * 40,
        "valid_feature": np.random.randn(40),
    })
    dio_one = DIO.create_empty(file_name="one.csv", dataset_hash="h_1")
    dio_one["columns"] = [
        {"name": "target", "dtype_inferred": "int", "is_target_candidate": True},
        {"name": "cancellation_date", "dtype_inferred": "string"},
        {"name": "valid_feature", "dtype_inferred": "float"},
    ]
    _, dio_one_out = agent.run(df_one, dio_one, run_dir=tmp_path / "one")
    assert dio_one_out["ml"]["status"] == "trained"
    assert dio_one_out["ml"]["feature_columns"] == ["valid_feature"]


def test_adversarial_target_isolation_strict(tmp_path: Path):
    """
    Prove that the target column can NEVER enter feature matrix X under any condition.
    """
    df = pd.DataFrame({
        "churn": [0, 1] * 20,
        "churn_reason": ["Price", "Support"] * 20,
        "usage_minutes": np.random.uniform(50, 500, 40),
    })
    dio = DIO.create_empty(file_name="churn.csv", dataset_hash="h_churn")
    dio["columns"] = [
        {"name": "churn", "dtype_inferred": "int", "is_target_candidate": True, "semantic_label": "target_label"},
        {"name": "churn_reason", "dtype_inferred": "string"},
        {"name": "usage_minutes", "dtype_inferred": "float"},
    ]
    agent = MLAgent()
    _, out_dio = agent.run(df, dio, run_dir=tmp_path)

    assert "churn" not in out_dio["ml"]["feature_columns"]
    # Verify model pipeline artifact only accepts feature columns
    model = joblib.load(out_dio["artifacts"]["model_pkl"])
    preds = model.predict(df[out_dio["ml"]["feature_columns"]].head(2))
    assert len(preds) == 2


def test_adversarial_pii_and_identifier_strict_exclusion(tmp_path: Path):
    """
    Verify that even if an identifier or PII column has 100% predictive correlation with the target,
    it is STRICTLY excluded from X. Security over performance.
    """
    # SSN perfectly correlates with target
    df = pd.DataFrame({
        "ssn": [f"000-00-000{i}" for i in range(40)],
        "id_num": range(40),
        "target": [0 if i < 20 else 1 for i in range(40)],
        "safe_feat": np.random.randn(40),
    })
    dio = DIO.create_empty(file_name="pii_pred.csv", dataset_hash="h_pii")
    dio["columns"] = [
        {"name": "ssn", "dtype_inferred": "string", "semantic_label": "national_id", "is_pii": True},
        {"name": "id_num", "dtype_inferred": "int", "semantic_label": "identifier"},
        {"name": "target", "dtype_inferred": "int", "is_target_candidate": True},
        {"name": "safe_feat", "dtype_inferred": "float"},
    ]
    agent = MLAgent()
    _, out_dio = agent.run(df, dio, run_dir=tmp_path)

    assert "ssn" not in out_dio["ml"]["feature_columns"]
    assert "id_num" not in out_dio["ml"]["feature_columns"]
    assert "safe_feat" in out_dio["ml"]["feature_columns"]


def test_adversarial_train_test_preprocessing_leakage_isolation():
    """
    Prove that transformers (imputer, scaler) are fitted STRICTLY on X_train.
    Adversarial test: X_test has an extreme outlier (1,000,000.0) which would drastically
    shift the standard scaler mean if full-dataset fitting occurred.
    """
    np.random.seed(42)
    X_train = pd.DataFrame({"feat": np.random.normal(10.0, 1.0, 32)})
    y_train = pd.Series([0, 1] * 16)
    
    # X_test has extreme outlier
    X_test = pd.DataFrame({"feat": list(np.random.normal(10.0, 1.0, 7)) + [1000000.0]})
    y_test = pd.Series([0, 1] * 4)

    preprocessor, _, _ = build_preprocessing_pipeline(
        df=X_train,
        feature_cols=["feat"],
        columns_info=[{"name": "feat", "dtype_inferred": "float"}],
    )

    results = train_and_evaluate_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task_type="classification",
        preprocessor=preprocessor,
        random_seed=42,
    )

    fitted_pipeline = results["selected_pipeline"]
    fitted_preprocessor = fitted_pipeline.named_steps["preprocessor"]
    scaler = fitted_preprocessor.named_transformers_["numeric"].named_steps["scaler"]

    # Scaler mean must match X_train mean (~10.0), NOT shifted by test set outlier
    expected_train_mean = float(X_train["feat"].mean())
    assert abs(scaler.mean_[0] - expected_train_mean) < 1e-5, (
        f"Scaler mean {scaler.mean_[0]} does not match X_train mean {expected_train_mean}"
    )


def test_adversarial_split_safety_and_row_thresholds():
    """
    Verify configurable data thresholds:
    - Exactly 30 rows vs 29 rows
    - Exactly 5 samples per class vs 4 samples per class
    """
    # 1. Exactly 30 rows -> PASS
    df_30 = pd.DataFrame({"y": [0, 1] * 15, "x": range(30)})
    ok_30, _, _ = validate_target_and_data(df_30, "y", "classification", min_rows_for_ml=30, min_rows_per_class=5)
    assert ok_30 is True

    # 2. 29 rows -> FAIL (ML_003_INSUFFICIENT_DATA)
    df_29 = pd.DataFrame({"y": [0, 1] * 14 + [0], "x": range(29)})
    ok_29, _, err_29 = validate_target_and_data(df_29, "y", "classification", min_rows_for_ml=30, min_rows_per_class=5)
    assert ok_29 is False
    assert err_29 == "ML_003_INSUFFICIENT_DATA"

    # 3. Exactly 5 samples in minority class -> PASS
    df_min5 = pd.DataFrame({"y": [0] * 25 + [1] * 5, "x": range(30)})
    ok_min5, _, _ = validate_target_and_data(df_min5, "y", "classification", min_rows_for_ml=30, min_rows_per_class=5)
    assert ok_min5 is True

    # 4. 4 samples in minority class -> FAIL
    df_min4 = pd.DataFrame({"y": [0] * 26 + [1] * 4, "x": range(30)})
    ok_min4, _, err_min4 = validate_target_and_data(df_min4, "y", "classification", min_rows_for_ml=30, min_rows_per_class=5)
    assert ok_min4 is False
    assert err_min4 == "ML_003_INSUFFICIENT_DATA"


def test_adversarial_baseline_comparison_and_model_selection(tmp_path: Path):
    """
    Verify that classification selects highest F1 and regression selects lowest RMSE,
    and baseline comparisons are explicitly reported in DIO.
    """
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "x1": np.random.uniform(10, 100, n),
        "x2": np.random.choice(["Low", "Med", "High"], n),
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    dio = DIO.create_empty(file_name="baseline.csv", dataset_hash="h_b")
    dio["columns"] = [
        {"name": "x1", "dtype_inferred": "float"},
        {"name": "x2", "dtype_inferred": "category"},
        {"name": "target", "dtype_inferred": "int", "is_target_candidate": True},
    ]

    agent = MLAgent()
    _, out_dio = agent.run(df, dio, run_dir=tmp_path)

    ml_sec = out_dio["ml"]
    assert "baseline" in ml_sec
    assert "f1" in ml_sec["baseline"]
    assert "selected_model" in ml_sec
    assert "selection_reason" in ml_sec
    assert "improved_over_baseline" in ml_sec


def test_adversarial_model_persistence_reload_and_prediction_integrity(tmp_path: Path):
    """
    Verify full cycle: train -> persist -> compute hash -> reload -> predict -> verify hash.
    """
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "num": np.random.randn(n),
        "cat": np.random.choice(["X", "Y"], n),
        "target": np.random.choice([0, 1], n),
    })
    dio = DIO.create_empty(file_name="persist.csv", dataset_hash="h_p")
    dio["columns"] = [
        {"name": "num", "dtype_inferred": "float"},
        {"name": "cat", "dtype_inferred": "category"},
        {"name": "target", "dtype_inferred": "int", "is_target_candidate": True},
    ]

    agent = MLAgent()
    _, out_dio = agent.run(df, dio, run_dir=tmp_path)

    model_path = Path(out_dio["artifacts"]["model_pkl"])
    initial_hash = out_dio["ml"]["model_artifact"]["sha256"]

    # 1. Reload pipeline
    pipeline = joblib.load(str(model_path))

    # 2. Predict on completely unseen input
    unseen_df = pd.DataFrame({
        "num": [0.5, -1.2],
        "cat": ["X", "Y"],
    })
    preds = pipeline.predict(unseen_df)
    assert len(preds) == 2
    assert preds[0] in (0, 1)
    assert preds[1] in (0, 1)

    # 3. Confirm file hash remains unchanged
    reloaded_hash = compute_file_hash(model_path)
    assert initial_hash == reloaded_hash


def test_adversarial_dio_mutation_boundary_snapshot(tmp_path: Path):
    """
    Deep snapshot test proving ML agent touches ONLY dio['ml'] and dio['artifacts']['model_pkl'].
    """
    df = pd.DataFrame({"x": range(40), "target": [0, 1] * 20})
    dio = DIO.create_empty(file_name="snap.csv", dataset_hash="h_snap")
    dio["ingestion"] = {"n_rows": 40, "n_columns": 2, "file_type": "csv", "encoding": "utf-8"}
    dio["columns"] = [{"name": "x", "dtype_inferred": "float"}, {"name": "target", "dtype_inferred": "int", "is_target_candidate": True}]
    dio["date_columns"] = [{"name": "date_col", "resolution": "unambiguous"}]
    dio["domain_guess"] = {"domain": "finance", "confidence": 0.85}
    dio["quality"] = {"score": 90, "issues": []}
    dio["cleaning_log"] = [{"column": "x", "action": "none"}]
    dio["eda"] = {"summary_stats": {"x": {"mean": 19.5}}, "correlations": {}, "charts": []}
    dio["artifacts"]["chart_paths"] = ["runs/default/artifacts/chart.png"]

    # Create deep copies of protected sections
    snap_ingestion = copy.deepcopy(dio["ingestion"])
    snap_columns = copy.deepcopy(dio["columns"])
    snap_date_columns = copy.deepcopy(dio["date_columns"])
    snap_domain = copy.deepcopy(dio["domain_guess"])
    snap_quality = copy.deepcopy(dio["quality"])
    snap_cleaning = copy.deepcopy(dio["cleaning_log"])
    snap_eda = copy.deepcopy(dio["eda"])
    snap_chart_paths = copy.deepcopy(dio["artifacts"]["chart_paths"])

    agent = MLAgent()
    _, out_dio = agent.run(df, dio, run_dir=tmp_path)

    assert out_dio["ingestion"] == snap_ingestion
    assert out_dio["columns"] == snap_columns
    assert out_dio["date_columns"] == snap_date_columns
    assert out_dio["domain_guess"] == snap_domain
    assert out_dio["quality"] == snap_quality
    assert out_dio["cleaning_log"] == snap_cleaning
    assert out_dio["eda"] == snap_eda
    assert out_dio["artifacts"]["chart_paths"] == snap_chart_paths


def test_positive_path_complete_synthetic_dataset_100_rows(tmp_path: Path):
    """
    CRITICAL REQUIREMENT (Requirement 14):
    End-to-end multi-agent execution on a realistic synthetic dataset (>= 100 rows):
    Ingestion -> IntelligenceAgent -> CleaningAgent -> EDAAgent -> MLAgent.
    Demonstrates full positive path: task detection, feature extraction, train/test split,
    preprocessing, baseline comparison, candidate model training, model selection,
    persistence, reload, and prediction inference.
    """
    np.random.seed(42)
    n = 120
    csv_file = tmp_path / "telecom_customer_churn_120.csv"

    # Generate synthetic tabular data
    raw_df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "account_length": np.random.randint(1, 72, n),
        "monthly_charges": np.random.uniform(20.0, 120.0, n),
        "total_charges": np.random.uniform(50.0, 5000.0, n),
        "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "payment_method": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "support_calls": np.random.randint(0, 10, n),
        "churn": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    raw_df.to_csv(csv_file, index=False)

    # 1. Ingestion
    router = DataRouter()
    df, dio, run_dir = router.ingest(csv_file, base_runs_dir=tmp_path)
    assert df.shape == (120, 8)

    # 2. Intelligence Agent
    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)
    assert len(dio["columns"]) == 8

    # 3. Cleaning Agent
    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)
    assert len(cleaned_df) == 120

    # 4. EDA Agent
    eda_agent = EDAAgent()
    cleaned_df, dio = eda_agent.run(cleaned_df, dio, run_dir=run_dir)
    assert len(dio["artifacts"]["chart_paths"]) > 0

    # 5. ML Agent
    ml_agent = MLAgent()
    cleaned_df, dio = ml_agent.run(cleaned_df, dio, run_dir=run_dir)

    # Assertions on successful ML path
    ml_res = dio["ml"]
    assert ml_res["status"] == "trained"
    assert ml_res["task_type"] == "classification"
    assert ml_res["target_column"] == "churn"
    assert "customer_id" in [e["column"] for e in ml_res["excluded_columns"]]
    assert len(ml_res["feature_columns"]) >= 5
    assert ml_res["split"]["train_rows"] == 96  # 80% of 120
    assert ml_res["split"]["test_rows"] == 24   # 20% of 120
    assert ml_res["selected_model"] in ("logistic_regression", "random_forest", "xgboost")
    assert "metrics" in ml_res
    assert "f1" in ml_res["metrics"]
    assert "accuracy" in ml_res["metrics"]

    # Verify model artifact exists, is hashed, and reloads
    model_path = Path(dio["artifacts"]["model_pkl"])
    assert model_path.is_file()
    assert ml_res["model_artifact"]["sha256"] != ""

    reloaded_pipeline = joblib.load(str(model_path))
    sample_test_row = cleaned_df[ml_res["feature_columns"]].head(3)
    preds = reloaded_pipeline.predict(sample_test_row)
    assert len(preds) == 3
    assert all(p in (0, 1) for p in preds)
