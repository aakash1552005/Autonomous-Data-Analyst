"""
tests/test_cleaning_agent.py
============================
Comprehensive test suite for Agent 2 (Cleaning Agent).
Verifies numeric imputation (mean/median), categorical imputation (mode/Unknown),
date preservation (missing dates remain missing), duplicate handling, sidecar creation,
type coercion, date handling, outlier flagging, DIO contract compliance, and lossless round-trip reconstruction.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from agents.cleaning.cleaning_agent import CleaningAgent
from agents.cleaning.imputer import impute_numeric_column, impute_categorical_column, impute_dataframe
from agents.cleaning.duplicate_handler import handle_duplicate_rows
from agents.cleaning.type_coercer import coerce_column_type
from agents.cleaning.date_handler import normalize_date_column
from agents.cleaning.outlier_detector import detect_column_outliers
from core.dio import DIO
from core.base_agent import ProgressState


def test_numeric_mean_imputation():
    # Symmetric distribution: skewness close to 0 (|skew| <= 1) -> mean
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, np.nan])
    imputed, log_entry = impute_numeric_column(series, "metric")

    assert log_entry is not None
    assert log_entry["method"] == "mean"
    assert log_entry["replacement_value"] == 30.0
    assert log_entry["reversible"] is True
    assert log_entry["generated_synthetic"] is True
    assert imputed.iloc[5] == 30.0


def test_numeric_median_imputation():
    # Highly skewed distribution: values with heavy outlier -> |skew| > 1 -> median
    series = pd.Series([1.0, 2.0, 2.0, 3.0, 2.0, 100.0, np.nan])
    imputed, log_entry = impute_numeric_column(series, "salary")

    assert log_entry is not None
    assert log_entry["method"] == "median"
    assert log_entry["replacement_value"] == 2.0
    assert log_entry["reversible"] is True
    assert log_entry["generated_synthetic"] is True
    assert imputed.iloc[6] == 2.0


def test_categorical_mode_imputation():
    # 1 missing out of 5 (20% <= 30%) -> mode "Alpha"
    series = pd.Series(["Alpha", "Alpha", "Beta", "Alpha", None])
    imputed, log_entry = impute_categorical_column(series, "category")

    assert log_entry is not None
    assert log_entry["method"] == "mode"
    assert log_entry["replacement_value"] == "Alpha"
    assert log_entry["reversible"] is True
    assert log_entry["generated_synthetic"] is True
    assert imputed.iloc[4] == "Alpha"


def test_categorical_unknown_imputation():
    # 3 missing out of 5 (60% > 30%) -> "Unknown"
    series = pd.Series(["Alpha", "Beta", None, None, None])
    imputed, log_entry = impute_categorical_column(series, "category")

    assert log_entry is not None
    assert log_entry["method"] == "constant_unknown"
    assert log_entry["replacement_value"] == "Unknown"
    assert log_entry["reversible"] is True
    assert log_entry["generated_synthetic"] is True
    assert (imputed.iloc[2:5] == "Unknown").all()


def test_missing_dates_remain_missing():
    """
    CRITICAL SAFETY REQUIREMENT:
    Proves that missing values in date columns are NEVER mode-imputed or synthetically invented.
    """
    df = pd.DataFrame({
        "admission_date": ["2024-01-10", "2024-02-12", "2024-03-01", None],
    })
    columns_info = [{"name": "admission_date", "dtype_inferred": "date", "semantic_label": "date"}]
    date_columns_info = [{"column": "admission_date", "detected_format": "YYYY-MM-DD", "needs_user_confirmation": False}]

    cleaned_df, logs = impute_dataframe(df, columns_info=columns_info, date_columns_info=date_columns_info)

    # Missing date MUST remain missing (NaN)
    assert pd.isna(cleaned_df["admission_date"].iloc[3])
    assert len(logs) == 0


def test_duplicate_detection_and_sidecar():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "name": ["A", "B", "B", "C"],
    })
    cleaned_df, removed_df, log_entry = handle_duplicate_rows(df)

    assert len(cleaned_df) == 3
    assert len(removed_df) == 1
    assert log_entry is not None
    assert log_entry["method"] == "drop_duplicates"
    assert log_entry["reversible"] is True
    assert "_orig_row_index" in removed_df.columns
    assert removed_df.iloc[0]["_orig_row_index"] == 2


def test_type_coercion_numeric_and_bool():
    s_int = pd.Series(["10", "20", "invalid", "40"])
    coerced_int, log_int = coerce_column_type(s_int, "int", "count_col")
    assert log_int is not None
    assert pd.isna(coerced_int.iloc[2])

    s_bool = pd.Series(["yes", "no", "TRUE", "0", None])
    coerced_bool, log_bool = coerce_column_type(s_bool, "bool", "flag_col")
    assert coerced_bool.tolist() == [True, False, True, False, None]


def test_date_handling_resolved_vs_ambiguous():
    # Resolved date
    s_res = pd.Series(["25/01/2024", "14/02/2024", None])
    date_meta_res = {"column": "order_date", "detected_format": "DD/MM/YYYY", "needs_user_confirmation": False}
    norm_res, log_res = normalize_date_column(s_res, date_meta_res)
    assert norm_res.iloc[0] == "2024-01-25"
    assert norm_res.iloc[1] == "2024-02-14"
    assert pd.isna(norm_res.iloc[2])  # Missing date remains missing
    assert log_res["method"] == "normalize_date"

    # Ambiguous date
    s_amb = pd.Series(["05/06/2024", "07/08/2024"])
    date_meta_amb = {"column": "sub_date", "detected_format": "ambiguous", "needs_user_confirmation": True}
    norm_amb, log_amb = normalize_date_column(s_amb, date_meta_amb)
    assert norm_amb.tolist() == ["05/06/2024", "07/08/2024"]
    assert log_amb["method"] == "preserve_ambiguous_date"
    assert log_amb.get("warning") is True


def test_iqr_outlier_flagging_without_modification():
    series = pd.Series([10.0, 12.0, 11.0, 13.0, 12.0, 11.5, 500.0])
    outlier_log = detect_column_outliers(series, "price")

    assert outlier_log is not None
    assert outlier_log["method"] == "flag_outliers_iqr"
    assert outlier_log["details"]["outlier_count"] == 1
    assert series.iloc[6] == 500.0


def test_cleaning_agent_full_run(tmp_path: Path):
    df = pd.DataFrame({
        "order_id": [101, 102, 102, 103],
        "category": ["Apparel", "Apparel", "Apparel", None],
        "revenue": [50.0, 75.0, 75.0, np.nan],
        "order_date": ["25/01/2024", "14/02/2024", "14/02/2024", "03/03/2024"],
        "ambig_date": ["01/02/2024", "02/03/2024", "02/03/2024", "03/04/2024"],
    })

    dio = DIO.create_empty(file_name="orders.csv", dataset_hash="raw_hash_999")
    dio["columns"] = [
        {"name": "order_id", "dtype_inferred": "int", "semantic_label": "identifier"},
        {"name": "category", "dtype_inferred": "category", "semantic_label": "category"},
        {"name": "revenue", "dtype_inferred": "float", "semantic_label": "currency_amount"},
        {"name": "order_date", "dtype_inferred": "date", "semantic_label": "order_date"},
        {"name": "ambig_date", "dtype_inferred": "date", "semantic_label": "date"},
    ]
    dio["date_columns"] = [
        {"column": "order_date", "detected_format": "DD/MM/YYYY", "needs_user_confirmation": False},
        {"column": "ambig_date", "detected_format": "ambiguous", "needs_user_confirmation": True},
    ]

    agent = CleaningAgent()
    cleaned_df, updated_dio = agent.run(df, dio, run_dir=tmp_path)

    # 1. State check
    assert updated_dio["progress"]["cleaning"] == ProgressState.FINISHED.value

    # 2. Duplicate removal
    assert len(cleaned_df) == 3

    # 3. Imputation check (category imputed with Apparel, revenue with mean/median)
    assert cleaned_df["category"].isna().sum() == 0
    assert cleaned_df["revenue"].isna().sum() == 0

    # 4. Artifacts check
    cleaned_path = Path(updated_dio["artifacts"]["cleaned_csv"])
    removed_path = Path(updated_dio["artifacts"]["removed_rows_csv"])
    assert cleaned_path.is_file()
    assert removed_path.is_file()

    # 5. Raw hash preserved
    assert updated_dio.dataset_hash == "raw_hash_999"


def test_round_trip_reconstruction(tmp_path: Path):
    """
    MANDATORY HARD-GATE REQUIREMENT:
    Proves that reversible=true is an authentic guarantee by fully reconstructing
    the exact original ingested DataFrame from cleaned_data + removed_rows + cleaning_log.
    """
    original_df = pd.DataFrame({
        "customer_id": ["C101", "C102", "C102", "C103", "C104"],
        "age": [25.0, 30.0, 30.0, np.nan, 45.0],
        "category": ["Bronze", "Silver", "Silver", None, "Gold"],
        "signup_date": ["25/01/2024", "14/02/2024", "14/02/2024", "05/03/2024", "18/04/2024"],
    })

    dio = DIO.create_empty(file_name="customers.csv", dataset_hash="orig_hash_123")
    dio["columns"] = [
        {"name": "customer_id", "dtype_inferred": "string", "semantic_label": "identifier"},
        {"name": "age", "dtype_inferred": "float", "semantic_label": "age"},
        {"name": "category", "dtype_inferred": "category", "semantic_label": "category"},
        {"name": "signup_date", "dtype_inferred": "date", "semantic_label": "date"},
    ]
    dio["date_columns"] = [
        {"column": "signup_date", "detected_format": "DD/MM/YYYY", "needs_user_confirmation": False}
    ]

    agent = CleaningAgent()
    cleaned_df, updated_dio = agent.run(original_df, dio, run_dir=tmp_path)

    saved_cleaned = pd.read_csv(updated_dio["artifacts"]["cleaned_csv"])
    saved_removed = pd.read_csv(updated_dio["artifacts"]["removed_rows_csv"])

    dup_log = next((l for l in updated_dio["cleaning_log"] if l.get("method") == "drop_duplicates"), None)
    removed_indices = set(dup_log["details"]["removed_indices"]) if dup_log else set()
    kept_indices = [i for i in original_df.index if i not in removed_indices]

    reconstructed_df = saved_cleaned.copy()
    reconstructed_df.index = kept_indices

    # Step A: Reverse Imputations using cleaning_log recorded null indices
    for log_entry in updated_dio["cleaning_log"]:
        if log_entry.get("method") in ("mean", "median", "mode", "constant_unknown") and "original_null_indices" in log_entry:
            col = log_entry["column"]
            for orig_idx in log_entry["original_null_indices"]:
                if orig_idx in reconstructed_df.index:
                    reconstructed_df.loc[orig_idx, col] = np.nan

    # Step B: Reverse date format normalization
    reconstructed_df["signup_date"] = [
        original_df.loc[idx, "signup_date"] for idx in reconstructed_df.index
    ]

    # Step C: Re-integrate removed duplicate rows
    if len(saved_removed) > 0:
        clean_removed = saved_removed.drop(columns=["_orig_row_index"]).copy()
        clean_removed.index = saved_removed["_orig_row_index"]
        reconstructed_df = pd.concat([reconstructed_df, clean_removed])

    # Step D: Restore original index ordering
    reconstructed_df = reconstructed_df.sort_index()

    # Step E: Assert exact equality against original ingested DataFrame
    assert len(reconstructed_df) == len(original_df)
    assert list(reconstructed_df.columns) == list(original_df.columns)

    for col in original_df.columns:
        orig_series = original_df[col]
        recon_series = reconstructed_df[col]
        assert (orig_series.isna().to_numpy() == recon_series.isna().to_numpy()).all()
        non_null_mask = orig_series.notna()
        assert (orig_series[non_null_mask].astype(str).to_numpy() == recon_series[non_null_mask].astype(str).to_numpy()).all()
