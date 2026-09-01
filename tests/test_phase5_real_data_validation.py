"""
tests/test_phase5_real_data_validation.py
=========================================
Real Data Validation suite for Phase 5 EDA Agent across all 5 benchmark datasets.
Evaluates summary statistics, Pearson/Spearman correlations, and Plotly+Kaleido PNG generation
for Retail, Healthcare, Financial, Mixed Messy, and Ambiguous Dates datasets.
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
from core.dio import DIO


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "phase5_validation"


def save_phase5_validation_record(
    dataset_name: str,
    dio: DIO,
) -> dict:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_file = FIXTURES_DIR / f"{dataset_name}_eda_output.json"

    eda_section = dio.get("eda", {})
    chart_paths = dio.get("artifacts", {}).get("chart_paths", [])

    record = {
        "dataset_name": dataset_name,
        "original_dataset_hash": dio.dataset_hash,
        "cleaned_dataset_hash": dio["decision_log"][-2].get("cleaned_dataset_hash", ""),
        "summary_stats_columns": list(eda_section.get("summary_stats", {}).keys()),
        "summary_stats": eda_section.get("summary_stats", {}),
        "correlations": eda_section.get("correlations", {}),
        "charts_generated_count": len(chart_paths),
        "chart_paths": chart_paths,
        "eda_metrics": [m for m in dio["agent_metrics"] if m.get("agent") == "eda"],
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    return record


def test_phase5_dataset_1_retail_sales(tmp_path: Path):
    file_path = DATA_DIR / "retail_sales.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)

    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)

    eda_agent = EDAAgent()
    _, dio = eda_agent.run(cleaned_df, dio, run_dir=run_dir)

    record = save_phase5_validation_record("retail_sales", dio)

    # 1. Assert summary statistics populated
    stats = dio["eda"]["summary_stats"]
    assert "revenue" in stats
    assert stats["revenue"]["mean"] > 0
    assert "quantity" in stats

    # 2. Assert correlations
    corr = dio["eda"]["correlations"]
    assert len(corr["top_correlations"]) > 0

    # 3. Assert chart files exist and are valid PNGs
    chart_paths = dio["artifacts"]["chart_paths"]
    assert len(chart_paths) > 0
    for p in chart_paths:
        chart_file = Path(p)
        assert chart_file.is_file()
        assert chart_file.stat().st_size > 1000  # Valid non-trivial PNG


def test_phase5_dataset_2_healthcare_patients(tmp_path: Path):
    file_path = DATA_DIR / "healthcare_patients.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)

    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)

    eda_agent = EDAAgent()
    _, dio = eda_agent.run(cleaned_df, dio, run_dir=run_dir)

    record = save_phase5_validation_record("healthcare_patients", dio)

    stats = dio["eda"]["summary_stats"]
    assert "glucose" in stats
    assert stats["glucose"]["count"] == 6  # 5 valid + 1 imputed value

    chart_paths = dio["artifacts"]["chart_paths"]
    assert len(chart_paths) > 0
    for p in chart_paths:
        assert Path(p).is_file()


def test_phase5_dataset_3_financial_loans(tmp_path: Path):
    file_path = DATA_DIR / "financial_loans.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)

    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)

    eda_agent = EDAAgent()
    _, dio = eda_agent.run(cleaned_df, dio, run_dir=run_dir)

    record = save_phase5_validation_record("financial_loans", dio)

    stats = dio["eda"]["summary_stats"]
    assert "loan_amount" in stats
    assert "account_balance" in stats

    chart_paths = dio["artifacts"]["chart_paths"]
    assert len(chart_paths) > 0


def test_phase5_dataset_4_mixed_messy_data(tmp_path: Path):
    file_path = DATA_DIR / "mixed_messy_data.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)

    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)

    eda_agent = EDAAgent()
    _, dio = eda_agent.run(cleaned_df, dio, run_dir=run_dir)

    record = save_phase5_validation_record("mixed_messy_data", dio)

    stats = dio["eda"]["summary_stats"]
    assert "metric_score" in stats
    assert any("user_category" in k for k in stats)

    chart_paths = dio["artifacts"]["chart_paths"]
    assert len(chart_paths) > 0


def test_phase5_dataset_5_ambiguous_dates_pii(tmp_path: Path):
    file_path = DATA_DIR / "ambiguous_dates_pii.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)

    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    cleaning_agent = CleaningAgent()
    cleaned_df, dio = cleaning_agent.run(df, dio, run_dir=run_dir)

    eda_agent = EDAAgent()
    _, dio = eda_agent.run(cleaned_df, dio, run_dir=run_dir)

    record = save_phase5_validation_record("ambiguous_dates_pii", dio)

    stats = dio["eda"]["summary_stats"]
    assert "amount" in stats

    chart_paths = dio["artifacts"]["chart_paths"]
    assert len(chart_paths) > 0
