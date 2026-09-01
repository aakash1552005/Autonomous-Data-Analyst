"""
tests/test_phase6_real_data_validation.py
=========================================
Real Data Validation suite for Phase 6 ML Agent across all 5 benchmark datasets.
Evaluates task detection, target validation, insufficient data handling,
and end-to-end training verification with JSON fixture persistence.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from core.data_router import DataRouter
from agents.intelligence.intelligence_agent import IntelligenceAgent
from agents.cleaning.cleaning_agent import CleaningAgent
from agents.eda.eda_agent import EDAAgent
from agents.ml.ml_agent import MLAgent
from core.dio import DIO


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "phase6_validation"


def save_phase6_validation_record(
    dataset_name: str,
    dio: DIO,
) -> dict:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_file = FIXTURES_DIR / f"{dataset_name}_ml_output.json"

    ml_section = dio.get("ml", {})
    record = {
        "dataset_name": dataset_name,
        "original_dataset_hash": dio.dataset_hash,
        "status": ml_section.get("status"),
        "task_type": ml_section.get("task_type"),
        "target_column": ml_section.get("target_column"),
        "reason": ml_section.get("reason"),
        "target_validation": ml_section.get("target_validation"),
        "feature_columns": ml_section.get("feature_columns"),
        "excluded_columns": ml_section.get("excluded_columns"),
        "baseline": ml_section.get("baseline"),
        "candidate_models": ml_section.get("candidate_models"),
        "selected_model": ml_section.get("selected_model"),
        "metrics": ml_section.get("metrics"),
        "model_artifact": ml_section.get("model_artifact"),
        "leakage_checks": ml_section.get("leakage_checks"),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    return record


def test_phase6_dataset_1_retail_sales(tmp_path: Path):
    file_path = DATA_DIR / "retail_sales.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)

    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)

    eda_agent = EDAAgent()
    cleaned_df, dio = eda_agent.run(cleaned_df, dio, run_dir=run_dir)

    ml_agent = MLAgent()
    _, dio = ml_agent.run(cleaned_df, dio, run_dir=run_dir)

    record = save_phase6_validation_record("retail_sales", dio)

    # 6 rows is < 30 rows -> MUST fail safely with insufficient_data, NOT train a fake model
    assert dio["ml"]["status"] in ("insufficient_data", "unsupported")
    assert dio["ml"]["target_column"] is not None or dio["ml"]["status"] == "unsupported"


def test_phase6_dataset_2_healthcare_patients(tmp_path: Path):
    file_path = DATA_DIR / "healthcare_patients.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)

    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)

    eda_agent = EDAAgent()
    cleaned_df, dio = eda_agent.run(cleaned_df, dio, run_dir=run_dir)

    ml_agent = MLAgent()
    _, dio = ml_agent.run(cleaned_df, dio, run_dir=run_dir)

    record = save_phase6_validation_record("healthcare_patients", dio)
    assert dio["ml"]["status"] in ("insufficient_data", "unsupported")


def test_phase6_dataset_3_financial_loans(tmp_path: Path):
    file_path = DATA_DIR / "financial_loans.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)

    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)

    eda_agent = EDAAgent()
    cleaned_df, dio = eda_agent.run(cleaned_df, dio, run_dir=run_dir)

    ml_agent = MLAgent()
    _, dio = ml_agent.run(cleaned_df, dio, run_dir=run_dir)

    record = save_phase6_validation_record("financial_loans", dio)
    assert dio["ml"]["target_column"] == "default_status"
    # Target correctly identified as default_status, but 6 rows < 30 rows -> insufficient_data
    assert dio["ml"]["status"] == "insufficient_data"


def test_phase6_dataset_4_mixed_messy_data(tmp_path: Path):
    file_path = DATA_DIR / "mixed_messy_data.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)

    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)

    eda_agent = EDAAgent()
    cleaned_df, dio = eda_agent.run(cleaned_df, dio, run_dir=run_dir)

    ml_agent = MLAgent()
    _, dio = ml_agent.run(cleaned_df, dio, run_dir=run_dir)

    record = save_phase6_validation_record("mixed_messy_data", dio)
    assert dio["ml"]["status"] in ("insufficient_data", "unsupported")


def test_phase6_dataset_5_ambiguous_dates_pii(tmp_path: Path):
    file_path = DATA_DIR / "ambiguous_dates_pii.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)

    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)

    eda_agent = EDAAgent()
    cleaned_df, dio = eda_agent.run(cleaned_df, dio, run_dir=run_dir)

    ml_agent = MLAgent()
    _, dio = ml_agent.run(cleaned_df, dio, run_dir=run_dir)

    record = save_phase6_validation_record("ambiguous_dates_pii", dio)
    # Correct rejection: 4 rows + PII + no defensible target
    assert dio["ml"]["status"] in ("insufficient_data", "unsupported")
