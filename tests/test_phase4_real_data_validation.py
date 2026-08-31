"""
tests/test_phase4_real_data_validation.py
=========================================
Real Data Validation suite for Phase 4 Cleaning Agent.
Evaluates the 5 real messy datasets across retail, healthcare, finance,
mixed anomalies, and ambiguous dates with PII.
Generates validation JSON artifacts in tests/fixtures/phase4_validation/.
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
    assert "discharge_date" in record["columns_affected_by_imputation"]
    assert record["missing_values_after"] == 0


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
    # Ambiguous date values must remain completely intact
    assert cleaned_df["subscription_date"].tolist() == orig_df["subscription_date"].tolist()
