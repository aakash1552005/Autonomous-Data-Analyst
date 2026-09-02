"""
tests/test_ml_agent.py
======================
Comprehensive test suite for Agent 4 (Machine Learning Agent).
Verifies:
1. Deterministic task detection (binary, multiclass, regression, unsupported).
2. Data thresholds (min_rows_for_ml, min_rows_per_class).
3. Leakage prevention (target, identifiers, PII, high-correlation, high-MI, post-outcome names).
4. Preprocessing and train/test splitting safety.
5. Baseline establishment and model evaluation.
6. Deterministic model selection.
7. Model artifact serialization, SHA-256 hashing, and inference reload.
8. Optional XGBoost graceful handling.
9. Zero-LLM guarantee and DIO section isolation.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from agents.ml.ml_agent import MLAgent
from agents.ml.task_detector import detect_ml_task
from agents.ml.target_validator import validate_target_and_data
from agents.ml.leakage_detector import detect_and_exclude_leakage
from agents.ml.preprocessor import build_preprocessing_pipeline
from agents.ml.model_trainer import train_and_evaluate_models
from core.config import AppConfig, MLConfig
from core.dio import DIO
from core.base_agent import ProgressState


def test_target_candidate_detection():
    df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "full_name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "churn": [0, 1, 0],
    })
    columns_info = [
        {"name": "user_id", "dtype_inferred": "int", "semantic_label": "identifier"},
        {"name": "full_name", "dtype_inferred": "string", "semantic_label": "person_name", "is_pii": True},
        {"name": "age", "dtype_inferred": "float", "semantic_label": "age"},
        {"name": "churn", "dtype_inferred": "int", "semantic_label": "target_label", "is_target_candidate": True},
    ]

    task_res = detect_ml_task(df, columns_info=columns_info)
    assert task_res["task_type"] == "classification"
    assert task_res["subtype"] == "binary"
    assert task_res["target_column"] == "churn"


def test_multiclass_and_regression_detection():
    # Multiclass
    df_multi = pd.DataFrame({"segment": ["Low", "Med", "High", "Low", "Med", "High"], "feat": [1, 2, 3, 4, 5, 6]})
    col_multi = [{"name": "segment", "dtype_inferred": "category", "is_target_candidate": True}, {"name": "feat", "dtype_inferred": "int"}]
    res_multi = detect_ml_task(df_multi, columns_info=col_multi)
    assert res_multi["task_type"] == "classification"
    assert res_multi["subtype"] == "multiclass"

    # Regression
    df_reg = pd.DataFrame({"price": np.linspace(100.0, 500.0, 20), "feat": range(20)})
    col_reg = [{"name": "price", "dtype_inferred": "float", "semantic_label": "target_value"}, {"name": "feat", "dtype_inferred": "int"}]
    res_reg = detect_ml_task(df_reg, columns_info=col_reg)
    assert res_reg["task_type"] == "regression"
    assert res_reg["subtype"] == "continuous"


def test_constant_target_rejection():
    df_const = pd.DataFrame({"target": [1, 1, 1, 1], "feat": [1, 2, 3, 4]})
    col_const = [{"name": "target", "dtype_inferred": "int", "is_target_candidate": True}]
    res_const = detect_ml_task(df_const, columns_info=col_const)
    assert res_const["task_type"] == "none"
    assert res_const["status"] == "constant_target"


def test_insufficient_data_thresholds():
    # Fewer than min_rows_for_ml (e.g. 20 rows < 30)
    df_small = pd.DataFrame({"y": [0, 1] * 10, "x": range(20)})
    is_valid, details, err = validate_target_and_data(df_small, "y", "classification", min_rows_for_ml=30, min_rows_per_class=5)
    assert is_valid is False
    assert err == "ML_003_INSUFFICIENT_DATA"

    # Minority class below min_rows_per_class (e.g. 4 samples < 5 in 35 rows)
    df_imbal = pd.DataFrame({"y": [0] * 31 + [1] * 4, "x": range(35)})
    is_valid_imbal, details_imbal, err_imbal = validate_target_and_data(df_imbal, "y", "classification", min_rows_for_ml=30, min_rows_per_class=5)
    assert is_valid_imbal is False
    assert err_imbal == "ML_003_INSUFFICIENT_DATA"
    assert "Minority class" in details_imbal["reason"]


def test_pii_and_identifier_leakage_exclusion():
    df = pd.DataFrame({
        "id": range(40),
        "email": [f"user_{i}@corp.com" for i in range(40)],
        "valid_num": np.random.randn(40),
        "valid_cat": ["A", "B", "C", "A"] * 10,
        "target": [0, 1] * 20,
    })
    columns_info = [
        {"name": "id", "semantic_label": "identifier"},
        {"name": "email", "semantic_label": "email", "is_pii": True},
        {"name": "valid_num", "dtype_inferred": "float"},
        {"name": "valid_cat", "dtype_inferred": "category"},
        {"name": "target", "dtype_inferred": "int", "is_target_candidate": True},
    ]

    res = detect_and_exclude_leakage(df, "target", columns_info=columns_info)
    assert "id" not in res["usable_features"]
    assert "email" not in res["usable_features"]
    assert "target" not in res["usable_features"]
    assert "valid_num" in res["usable_features"]
    assert "valid_cat" in res["usable_features"]


def test_high_correlation_and_high_mi_leakage():
    # Feature 1: Near-perfect correlation with continuous target (r = 0.999 > 0.95)
    y_reg = np.linspace(10.0, 100.0, 40)
    feat_leaky_num = y_reg + np.random.normal(0, 0.01, 40)
    feat_clean = np.random.normal(0, 1, 40)

    df_reg = pd.DataFrame({
        "target": y_reg,
        "feat_leaky_num": feat_leaky_num,
        "feat_clean": feat_clean,
    })
    res_reg = detect_and_exclude_leakage(df_reg, "target", leakage_threshold=0.95)
    assert "feat_leaky_num" not in res_reg["usable_features"]
    assert "feat_clean" in res_reg["usable_features"]

    # Feature 2: High mutual information categorical leakage (NMI = 1.0 > 0.95)
    y_clf = ["Yes", "No"] * 20
    feat_leaky_cat = y_clf  # Exact duplicate
    df_clf = pd.DataFrame({
        "target": y_clf,
        "feat_leaky_cat": feat_leaky_cat,
        "feat_clean": np.random.randn(40),
    })
    res_clf = detect_and_exclude_leakage(df_clf, "target", leakage_threshold=0.95)
    assert "feat_leaky_cat" not in res_clf["usable_features"]
    assert "feat_clean" in res_clf["usable_features"]


def test_post_outcome_column_name_leakage():
    df = pd.DataFrame({
        "target": [0, 1] * 20,
        "cancellation_date": ["2024-01-01"] * 40,
        "outcome_notes": ["Resolved"] * 40,
        "regular_feature": range(40),
    })
    res = detect_and_exclude_leakage(df, "target")
    assert "cancellation_date" not in res["usable_features"]
    assert "outcome_notes" not in res["usable_features"]
    assert "regular_feature" in res["usable_features"]


def test_full_ml_agent_classification_run(tmp_path: Path):
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "cust_id": [f"C{i}" for i in range(n)],
        "age": np.random.randint(20, 70, n).astype(float),
        "income": np.random.uniform(30000, 120000, n),
        "plan_type": np.random.choice(["Basic", "Premium", "Enterprise"], n),
        "churn": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    })

    dio = DIO.create_empty(file_name="telecom.csv", dataset_hash="hash_telecom")
    dio["columns"] = [
        {"name": "cust_id", "dtype_inferred": "string", "semantic_label": "identifier"},
        {"name": "age", "dtype_inferred": "float", "semantic_label": "age"},
        {"name": "income", "dtype_inferred": "float", "semantic_label": "financial_metric"},
        {"name": "plan_type", "dtype_inferred": "category", "semantic_label": "category"},
        {"name": "churn", "dtype_inferred": "int", "semantic_label": "target_label", "is_target_candidate": True},
    ]

    config = AppConfig()
    config.ml.min_rows_for_ml = 30
    config.ml.min_rows_per_class = 5

    agent = MLAgent(config=config)
    returned_df, updated_dio = agent.run(df, dio, run_dir=tmp_path)

    # 1. Check Progress and ML Status
    assert updated_dio["progress"]["ml"] == ProgressState.FINISHED.value
    assert updated_dio["ml"]["status"] == "trained"
    assert updated_dio["ml"]["task_type"] == "classification"
    assert updated_dio["ml"]["target_column"] == "churn"

    # 2. Check Candidate Models & Baseline
    assert "baseline" in updated_dio["ml"]
    assert "accuracy" in updated_dio["ml"]["baseline"]
    assert updated_dio["ml"]["selected_model"] in ("logistic_regression", "random_forest", "xgboost")
    assert "metrics" in updated_dio["ml"]
    assert "f1" in updated_dio["ml"]["metrics"]

    # 3. Check Artifact Persistence & Reload
    model_path = Path(updated_dio["artifacts"]["model_pkl"])
    assert model_path.is_file()
    assert updated_dio["ml"]["model_artifact"]["sha256"] != ""

    # Test Reload & Predict
    reloaded = joblib.load(str(model_path))
    sample_preds = reloaded.predict(df[["age", "income", "plan_type"]].head(3))
    assert len(sample_preds) == 3


def test_full_ml_agent_regression_run(tmp_path: Path):
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "house_id": [f"H{i}" for i in range(n)],
        "sqft": np.random.uniform(800, 3500, n),
        "bedrooms": np.random.randint(1, 6, n).astype(float),
        "neighborhood": np.random.choice(["Downtown", "Suburbs", "Rural"], n),
        "price": np.random.uniform(150000, 750000, n),
    })

    dio = DIO.create_empty(file_name="houses.csv", dataset_hash="hash_houses")
    dio["columns"] = [
        {"name": "house_id", "dtype_inferred": "string", "semantic_label": "identifier"},
        {"name": "sqft", "dtype_inferred": "float"},
        {"name": "bedrooms", "dtype_inferred": "float"},
        {"name": "neighborhood", "dtype_inferred": "category"},
        {"name": "price", "dtype_inferred": "float", "semantic_label": "target_value", "is_target_candidate": True},
    ]

    agent = MLAgent()
    _, updated_dio = agent.run(df, dio, run_dir=tmp_path)

    assert updated_dio["ml"]["status"] == "trained"
    assert updated_dio["ml"]["task_type"] == "regression"
    assert updated_dio["ml"]["target_column"] == "price"
    assert updated_dio["ml"]["selected_model"] in ("ridge_regression", "random_forest", "xgboost")
    assert "rmse" in updated_dio["ml"]["metrics"]
    assert updated_dio["ml"]["metrics"]["rmse"] > 0


def test_dio_mutation_boundary_ml(tmp_path: Path):
    """
    CRITICAL REQUIREMENT:
    Verify that ML agent mutates ONLY dio['ml'] and dio['artifacts']['model_pkl'].
    """
    df = pd.DataFrame({"y": [0, 1] * 20, "x": range(40)})
    dio = DIO.create_empty(file_name="iso.csv", dataset_hash="h_iso")
    dio["ingestion"] = {"n_rows": 40, "n_columns": 2, "file_type": "csv", "encoding": "utf-8"}
    dio["columns"] = [{"name": "x", "dtype_inferred": "float"}, {"name": "y", "dtype_inferred": "int", "is_target_candidate": True}]
    dio["quality"] = {"score": 95, "issues": []}
    dio["cleaning_log"] = [{"column": "x", "method": "none"}]
    dio["eda"] = {"summary_stats": {"x": {"mean": 19.5}}, "correlations": {}, "charts": []}

    agent = MLAgent()
    _, updated_dio = agent.run(df, dio, run_dir=tmp_path)

    # Upstream sections MUST remain strictly untouched
    assert updated_dio["ingestion"]["n_rows"] == 40
    assert updated_dio["quality"]["score"] == 95
    assert len(updated_dio["cleaning_log"]) == 1
    assert updated_dio["eda"]["summary_stats"]["x"]["mean"] == 19.5
    # Future sections MUST remain untouched
    assert len(updated_dio["insights"]) == 0
    assert updated_dio["reports"]["pdf_path"] is None


def test_zero_llm_ml_guarantee(tmp_path: Path):
    """
    CRITICAL REQUIREMENT:
    Asserts that ML execution operates 100% deterministically without making any LLM calls.
    """
    df = pd.DataFrame({"feat": range(40), "target": [0, 1] * 20})
    dio = DIO.create_empty(file_name="test.csv", dataset_hash="h1")
    dio["columns"] = [{"name": "feat", "dtype_inferred": "float"}, {"name": "target", "dtype_inferred": "int", "is_target_candidate": True}]

    agent = MLAgent()
    _, updated_dio = agent.run(df, dio, run_dir=tmp_path)

    assert updated_dio["llm_usage"]["total_calls"] == 0
    assert updated_dio["llm_usage"]["total_tokens"] == 0


def test_insufficient_features_after_leakage_exclusion(tmp_path: Path):
    # Dataset where the only non-target feature is a post-outcome cancellation date
    df = pd.DataFrame({
        "target": [0, 1] * 20,
        "cancellation_date": ["2024-01-01"] * 40,
    })
    dio = DIO.create_empty(file_name="leak.csv", dataset_hash="h_leak")
    dio["columns"] = [
        {"name": "target", "dtype_inferred": "int", "is_target_candidate": True},
        {"name": "cancellation_date", "dtype_inferred": "string"},
    ]

    agent = MLAgent()
    _, updated_dio = agent.run(df, dio, run_dir=tmp_path)

    assert updated_dio["ml"]["status"] == "leakage_detected"
    assert updated_dio["ml"]["error_code"] == "ML_004_LEAKAGE_DETECTED"
    assert any(err["code"] == "ML_004_LEAKAGE_DETECTED" for err in updated_dio["errors"])


def test_optional_xgboost_simulation_when_unavailable(tmp_path: Path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("No module named 'xgboost'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    df = pd.DataFrame({
        "x": range(40),
        "target": [0, 1] * 20,
    })
    dio = DIO.create_empty(file_name="xgb_sim.csv", dataset_hash="h_xgb")
    dio["columns"] = [
        {"name": "x", "dtype_inferred": "float"},
        {"name": "target", "dtype_inferred": "int", "is_target_candidate": True},
    ]

    config = AppConfig()
    config.ml.min_rows_xgboost = 30  # Allow 40 rows to attempt XGBoost import

    agent = MLAgent(config=config)
    _, updated_dio = agent.run(df, dio, run_dir=tmp_path)

    assert updated_dio["ml"]["status"] == "trained"
    xgb_record = next((c for c in updated_dio["ml"]["candidate_models"] if c["model_name"] == "xgboost"), None)
    assert xgb_record is not None
    assert xgb_record["status"] == "skipped"
    assert "not available" in xgb_record["reason"]


def test_classification_metrics_ground_truth():
    from agents.ml.model_trainer import evaluate_classification_model
    from sklearn.dummy import DummyClassifier

    # 10 samples: 6 class 0, 4 class 1
    X_test = pd.DataFrame({"x": range(10)})
    y_test = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    # Model always predicts 0
    dummy = DummyClassifier(strategy="constant", constant=0)
    dummy.fit(X_test, y_test)

    metrics = evaluate_classification_model(dummy, X_test, y_test, n_classes=2)
    assert metrics["accuracy"] == 0.6  # 6/10
    assert metrics["precision"] == 0.0  # 0 true positives for class 1
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["confusion_matrix"] == [[6, 0], [4, 0]]


def test_regression_metrics_ground_truth():
    from agents.ml.model_trainer import evaluate_regression_model
    from sklearn.dummy import DummyRegressor

    X_test = pd.DataFrame({"x": range(4)})
    y_test = pd.Series([10.0, 20.0, 30.0, 40.0])  # Mean = 25.0

    # Dummy regressor predicts mean = 25.0
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_test, y_test)

    metrics = evaluate_regression_model(dummy, X_test, y_test)
    # Errors: |10-25|=15, |20-25|=5, |30-25|=5, |40-25|=15 -> MAE = (15+5+5+15)/4 = 10.0
    # Squared errors: 225 + 25 + 25 + 225 = 500 -> MSE = 500/4 = 125.0 -> RMSE = sqrt(125) = 11.1803
    # R2 for mean prediction = 0.0
    assert metrics["mae"] == 10.0
    assert metrics["mse"] == 125.0
    assert metrics["rmse"] == round(125.0 ** 0.5, 4)
    assert metrics["r2"] == 0.0


def test_configuration_changes_runtime_behavior_without_code_modification():
    config_default = AppConfig()
    config_default.ml.min_rows_for_ml = 50

    df = pd.DataFrame({"x": range(40), "target": [0, 1] * 20})
    is_valid, details, err = validate_target_and_data(
        df, "target", "classification", min_rows_for_ml=config_default.ml.min_rows_for_ml
    )
    assert is_valid is False
    assert err == "ML_003_INSUFFICIENT_DATA"

    # Lower threshold dynamically via configuration
    config_lower = AppConfig()
    config_lower.ml.min_rows_for_ml = 30
    is_valid_lower, _, _ = validate_target_and_data(
        df, "target", "classification", min_rows_for_ml=config_lower.ml.min_rows_for_ml
    )
    assert is_valid_lower is True


def test_dataset_immutability_during_ml_training(tmp_path: Path):
    from core.hashing import compute_string_hash
    df = pd.DataFrame({"feat": range(40), "target": [0, 1] * 20})
    df_copy = df.copy()
    initial_hash = compute_string_hash(df.to_csv(index=False))

    dio = DIO.create_empty(file_name="immut.csv", dataset_hash="h_immut")
    dio["columns"] = [{"name": "feat", "dtype_inferred": "float"}, {"name": "target", "dtype_inferred": "int", "is_target_candidate": True}]

    agent = MLAgent()
    returned_df, _ = agent.run(df, dio, run_dir=tmp_path)

    final_hash = compute_string_hash(returned_df.to_csv(index=False))
    assert initial_hash == final_hash
    pd.testing.assert_frame_equal(returned_df, df_copy)

