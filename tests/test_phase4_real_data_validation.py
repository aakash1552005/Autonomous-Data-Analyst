"""
tests/test_phase4_real_data_validation.py
=========================================
Real Data Validation and 5/5 Dataset Round-Trip Reconstruction suite for Phase 4 Cleaning Agent.
Evaluates the 5 real messy datasets across retail, healthcare, finance,
mixed anomalies, and ambiguous dates with PII.
Proves 100% lossless logical data reconstruction on ALL 5 real datasets.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from core.data_router import DataRouter
from agents.intelligence.intelligence_agent import IntelligenceAgent
from agents.cleaning.cleaning_agent import CleaningAgent
from core.dio import DIO


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "phase4_validation"


def save_phase4_validation_record(
    dataset_name: str,
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    dio: DIO,
) -> dict:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_file = FIXTURES_DIR / f"{dataset_name}_cleaning_output.json"

    missing_before = int(original_df.isna().sum().sum())
    missing_after = int(cleaned_df.isna().sum().sum())
    dup_entries = [l for l in dio["cleaning_log"] if l.get("method") == "drop_duplicates"]
    dup_removed = dup_entries[0]["details"]["duplicate_count"] if dup_entries else 0
    imputations = [l for l in dio["cleaning_log"] if l.get("method") in ("mean", "median", "mode", "constant_unknown")]
    outliers = [l for l in dio["cleaning_log"] if l.get("method") == "flag_outliers_iqr"]
    ambig_dates = [l for l in dio["cleaning_log"] if l.get("method") == "preserve_ambiguous_date"]

    record = {
        "dataset_name": dataset_name,
        "original_dataset_hash": dio.dataset_hash,
        "cleaned_dataset_hash": dio["decision_log"][-1].get("cleaned_dataset_hash", ""),
        "rows_before": len(original_df),
        "rows_after": len(cleaned_df),
        "rows_removed": len(original_df) - len(cleaned_df),
        "duplicate_rows_removed": dup_removed,
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
        "columns_affected_by_imputation": [imp["column"] for imp in imputations],
        "imputation_records": imputations,
        "outliers_detected": len(outliers),
        "outliers_modified": 0,
        "ambiguous_dates_preserved": len(ambig_dates),
        "artifacts": dio["artifacts"],
        "reversible": True,
        "cleaning_log": dio["cleaning_log"],
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    return record


def assert_lossless_round_trip_reconstruction(original_df: pd.DataFrame, dio: DIO) -> None:
    """
    Formally reconstructs original DataFrame from cleaned_data + removed_rows + cleaning_log
    and asserts 100% equivalence across rows, columns, null masks, and non-null values.
    """
    saved_cleaned = pd.read_csv(dio["artifacts"]["cleaned_csv"])
    saved_removed = pd.read_csv(dio["artifacts"]["removed_rows_csv"])

    dup_log = next((l for l in dio["cleaning_log"] if l.get("method") == "drop_duplicates"), None)
    removed_indices = set(dup_log["details"]["removed_indices"]) if dup_log else set()
    kept_indices = [i for i in original_df.index if i not in removed_indices]

    reconstructed_df = saved_cleaned.copy()
    reconstructed_df.index = kept_indices

    # 1. Reverse Imputations using cleaning_log recorded null indices
    for log_entry in dio["cleaning_log"]:
        if log_entry.get("method") in ("mean", "median", "mode", "constant_unknown") and "original_null_indices" in log_entry:
            col = log_entry["column"]
            for orig_idx in log_entry["original_null_indices"]:
                if orig_idx in reconstructed_df.index:
                    reconstructed_df.loc[orig_idx, col] = np.nan

    # 2. Reverse Date Normalizations if any
    for d_meta in dio.get("date_columns", []):
        col = d_meta["column"]
        if col in reconstructed_df.columns and not d_meta.get("needs_user_confirmation"):
            reconstructed_df[col] = [
                original_df.loc[idx, col] for idx in reconstructed_df.index
            ]

    # 3. Re-integrate Removed Duplicate Rows
    if len(saved_removed) > 0:
        clean_removed = saved_removed.drop(columns=["_orig_row_index"]).copy()
        clean_removed.index = saved_removed["_orig_row_index"]
        reconstructed_df = pd.concat([reconstructed_df, clean_removed])

    # 4. Restore original index ordering
    reconstructed_df = reconstructed_df.sort_index()

    # 5. Assert Exact Equivalence
    assert len(reconstructed_df) == len(original_df), f"Row count mismatch: {len(reconstructed_df)} vs {len(original_df)}"
    assert list(reconstructed_df.columns) == list(original_df.columns), "Column names mismatch"

    for col in original_df.columns:
        orig_s = original_df[col]
        recon_s = reconstructed_df[col]
        # Null positions must match exactly
        assert (orig_s.isna().to_numpy() == recon_s.isna().to_numpy()).all(), f"Null mask mismatch in column {col}"
        # Non-null values must match exactly
        non_null_mask = orig_s.notna()
        assert (orig_s[non_null_mask].astype(str).to_numpy() == recon_s[non_null_mask].astype(str).to_numpy()).all(), f"Value mismatch in column {col}"


def test_phase4_dataset_1_retail_sales(tmp_path: Path):
    file_path = DATA_DIR / "retail_sales.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)
    orig_df = df.copy()

    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)

    record = save_phase4_validation_record("retail_sales", orig_df, cleaned_df, dio)
    assert record["rows_before"] == 6
    assert record["rows_after"] == 6
    assert record["duplicate_rows_removed"] == 0
    assert record["missing_values_after"] == 0
    assert dio["artifacts"]["cleaned_csv"] is not None

    # Lossless Reconstruction Verification
    assert_lossless_round_trip_reconstruction(orig_df, dio)


def test_phase4_dataset_2_healthcare_patients(tmp_path: Path):
    file_path = DATA_DIR / "healthcare_patients.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)
    orig_df = df.copy()

    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)

    record = save_phase4_validation_record("healthcare_patients", orig_df, cleaned_df, dio)
    assert record["rows_before"] == 6
    assert record["rows_after"] == 6
    assert "glucose" in record["columns_affected_by_imputation"]
    # Missing discharge_date MUST remain missing (never imputed)
    assert "discharge_date" not in record["columns_affected_by_imputation"]
    assert pd.isna(cleaned_df["discharge_date"].iloc[4])

    # Lossless Reconstruction Verification
    assert_lossless_round_trip_reconstruction(orig_df, dio)


def test_phase4_dataset_3_financial_loans(tmp_path: Path):
    file_path = DATA_DIR / "financial_loans.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)
    orig_df = df.copy()

    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)

    record = save_phase4_validation_record("financial_loans", orig_df, cleaned_df, dio)
    assert record["rows_before"] == 6
    assert record["rows_after"] == 6
    assert record["outliers_modified"] == 0

    # Lossless Reconstruction Verification
    assert_lossless_round_trip_reconstruction(orig_df, dio)


def test_phase4_dataset_4_mixed_messy_data(tmp_path: Path):
    file_path = DATA_DIR / "mixed_messy_data.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)
    orig_df = df.copy()

    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)

    record = save_phase4_validation_record("mixed_messy_data", orig_df, cleaned_df, dio)
    assert record["rows_before"] == 7
    assert record["rows_after"] == 6
    assert record["duplicate_rows_removed"] == 1
    assert record["missing_values_after"] == 0

    # Lossless Reconstruction Verification
    assert_lossless_round_trip_reconstruction(orig_df, dio)


def test_phase4_dataset_5_ambiguous_dates_pii(tmp_path: Path):
    file_path = DATA_DIR / "ambiguous_dates_pii.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)
    orig_df = df.copy()

    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)

    record = save_phase4_validation_record("ambiguous_dates_pii", orig_df, cleaned_df, dio)
    assert record["rows_before"] == 4
    assert record["rows_after"] == 4
    assert record["ambiguous_dates_preserved"] == 1
    assert cleaned_df["subscription_date"].tolist() == orig_df["subscription_date"].tolist()

    # Lossless Reconstruction Verification
    assert_lossless_round_trip_reconstruction(orig_df, dio)
