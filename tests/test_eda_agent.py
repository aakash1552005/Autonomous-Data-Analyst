"""
tests/test_eda_agent.py
=======================
Comprehensive test suite for Agent 3 (Exploratory Data Analysis Agent).
Verifies numeric and categorical summary statistics, Pearson/Spearman correlations,
deterministic chart selection, configurable chart count limits, Plotly/Kaleido PNG rendering,
zero-LLM execution guarantee, and graceful degradation on narrow dataset fixtures.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from agents.eda.eda_agent import EDAAgent
from agents.eda.summary_stats import compute_numeric_summary, compute_categorical_summary, compute_dataset_summary_stats
from agents.eda.correlations import compute_correlations
from agents.eda.chart_generator import select_charts_deterministically, generate_all_eda_charts
from core.config import AppConfig, EDAConfig
from core.dio import DIO
from core.base_agent import ProgressState


def test_numeric_summary_statistics():
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, np.nan])
    res = compute_numeric_summary(series, "salary")

    assert res["column"] == "salary"
    assert res["type"] == "numeric"
    assert res["count"] == 5
    assert res["null_count"] == 1
    assert res["null_pct"] == 16.67
    assert res["mean"] == 30.0
    assert res["min"] == 10.0
    assert res["q25"] == 20.0
    assert res["median"] == 30.0
    assert res["q75"] == 40.0
    assert res["max"] == 50.0
    assert res["iqr"] == 20.0
    assert res["skewness"] == 0.0


def test_categorical_summary_statistics():
    series = pd.Series(["Gold", "Silver", "Gold", "Bronze", "Gold", None])
    res = compute_categorical_summary(series, "tier")

    assert res["column"] == "tier"
    assert res["type"] == "categorical"
    assert res["count"] == 5
    assert res["unique_count"] == 3
    assert res["null_count"] == 1
    assert res["null_pct"] == 16.67
    assert res["top_category"] == "Gold"
    assert res["top_category_freq"] == 3
    assert res["top_categories"]["Gold"] == 3
    assert res["top_categories"]["Silver"] == 1


def test_pearson_and_spearman_correlations():
    # Perfectly correlated x and y
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],
        "z": [5.0, 4.0, 3.0, 2.0, 1.0],
    })
    columns_info = [
        {"name": "x", "dtype_inferred": "float"},
        {"name": "y", "dtype_inferred": "float"},
        {"name": "z", "dtype_inferred": "float"},
    ]
    res = compute_correlations(df, columns_info=columns_info)

    assert "x" in res["pearson"]
    assert res["pearson"]["x"]["y"] == 1.0
    assert res["pearson"]["x"]["z"] == -1.0
    assert len(res["top_correlations"]) >= 1
    assert res["top_correlations"][0]["abs_r"] == 1.0


def test_deterministic_chart_selection_order():
    """
    CRITICAL REQUIREMENT:
    Assert that repeated runs against identical data produce identical chart selections and ordering.
    """
    df = pd.DataFrame({
        "sales": [100.0, 200.0, 300.0, 400.0],
        "profit": [10.0, 20.0, 30.0, 40.0],
        "discount": [0.05, 0.10, 0.15, 0.20],
        "region": ["North", "South", "East", "West"],
        "order_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
    })
    columns_info = [
        {"name": "sales", "dtype_inferred": "float"},
        {"name": "profit", "dtype_inferred": "float"},
        {"name": "discount", "dtype_inferred": "float"},
        {"name": "region", "dtype_inferred": "string"},
        {"name": "order_date", "dtype_inferred": "date"},
    ]
    date_columns_info = [{"column": "order_date", "detected_format": "YYYY-MM-DD"}]

    plans_run1 = select_charts_deterministically(df, columns_info, date_columns_info, max_charts=6)
    plans_run2 = select_charts_deterministically(df, columns_info, date_columns_info, max_charts=6)

    assert len(plans_run1) == len(plans_run2)
    for p1, p2 in zip(plans_run1, plans_run2):
        assert p1["chart_id"] == p2["chart_id"]
        assert p1["chart_type"] == p2["chart_type"]
        assert p1["title"] == p2["title"]


def test_configurable_max_charts():
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4],
        "f2": [10, 20, 30, 40],
        "f3": [100, 200, 300, 400],
        "cat1": ["A", "B", "A", "B"],
        "cat2": ["X", "Y", "X", "Y"],
    })
    columns_info = [{"name": c, "dtype_inferred": "float" if "f" in c else "category"} for c in df.columns]

    plans_max2 = select_charts_deterministically(df, columns_info, max_charts=2)
    assert len(plans_max2) == 2

    plans_max4 = select_charts_deterministically(df, columns_info, max_charts=4)
    assert len(plans_max4) == 4


def test_png_chart_artifacts_rendered(tmp_path: Path):
    """
    CRITICAL REQUIREMENT:
    Assert that Plotly + Kaleido renders actual valid non-empty PNG files.
    """
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "revenue": [50.0, 75.0, 120.0, 200.0],
        "cost": [30.0, 45.0, 70.0, 110.0],
        "margin": [0.4, 0.4, 0.41, 0.45],
        "category": ["Hardware", "Software", "Hardware", "Services"],
        "sale_date": ["2024-01-10", "2024-01-15", "2024-01-20", "2024-01-25"],
    })

    dio = DIO.create_empty(file_name="sales.csv", dataset_hash="h123")
    dio["columns"] = [
        {"name": "order_id", "dtype_inferred": "int"},
        {"name": "revenue", "dtype_inferred": "float"},
        {"name": "cost", "dtype_inferred": "float"},
        {"name": "margin", "dtype_inferred": "float"},
        {"name": "category", "dtype_inferred": "category"},
        {"name": "sale_date", "dtype_inferred": "date"},
    ]
    dio["date_columns"] = [{"column": "sale_date", "detected_format": "YYYY-MM-DD", "needs_user_confirmation": False}]

    agent = EDAAgent()
    returned_df, updated_dio = agent.run(df, dio, run_dir=tmp_path)

    # Check progress
    assert updated_dio["progress"]["eda"] == ProgressState.FINISHED.value

    # Check charts generated
    chart_paths = updated_dio["artifacts"]["chart_paths"]
    assert len(chart_paths) > 0
    assert len(chart_paths) <= 6

    for p_str in chart_paths:
        p = Path(p_str)
        assert p.suffix == ".png"
        assert p.is_file()
        assert p.stat().st_size > 0  # Non-empty rendered image


def test_zero_llm_eda_guarantee(tmp_path: Path):
    """
    CRITICAL REQUIREMENT:
    Asserts that EDA execution operates 100% deterministically without making any LLM calls.
    """
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    dio = DIO.create_empty(file_name="simple.csv", dataset_hash="h999")
    dio["columns"] = [{"name": "a", "dtype_inferred": "float"}, {"name": "b", "dtype_inferred": "float"}]

    agent = EDAAgent()
    _, updated_dio = agent.run(df, dio, run_dir=tmp_path)

    # Ensure LLM usage remains 0
    assert updated_dio["llm_usage"]["total_calls"] == 0
    assert updated_dio["llm_usage"]["total_tokens"] == 0


def test_graceful_degradation_narrow_fixtures(tmp_path: Path):
    """
    CRITICAL REQUIREMENT:
    Verify graceful degradation across narrow fixtures without raising exceptions.
    """
    agent = EDAAgent()

    # Fixture 1: Single numeric column
    df_single = pd.DataFrame({"metric": [10.0, 20.0, 30.0, 40.0]})
    dio_single = DIO.create_empty(file_name="single.csv", dataset_hash="h1")
    dio_single["columns"] = [{"name": "metric", "dtype_inferred": "float"}]
    _, dio_single = agent.run(df_single, dio_single, run_dir=tmp_path / "f1")
    assert dio_single["progress"]["eda"] == ProgressState.FINISHED.value
    assert len(dio_single["artifacts"]["chart_paths"]) >= 1

    # Fixture 2: All-categorical columns (no numerics)
    df_cats = pd.DataFrame({"city": ["London", "Paris", "Berlin"], "status": ["Active", "Active", "Pending"]})
    dio_cats = DIO.create_empty(file_name="cats.csv", dataset_hash="h2")
    dio_cats["columns"] = [{"name": "city", "dtype_inferred": "string"}, {"name": "status", "dtype_inferred": "string"}]
    _, dio_cats = agent.run(df_cats, dio_cats, run_dir=tmp_path / "f2")
    assert dio_cats["progress"]["eda"] == ProgressState.FINISHED.value
    assert len(dio_cats["artifacts"]["chart_paths"]) >= 1

    # Fixture 3: Numeric columns with NO date column
    df_nodate = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    dio_nodate = DIO.create_empty(file_name="nodate.csv", dataset_hash="h3")
    dio_nodate["columns"] = [{"name": "x", "dtype_inferred": "float"}, {"name": "y", "dtype_inferred": "float"}]
    _, dio_nodate = agent.run(df_nodate, dio_nodate, run_dir=tmp_path / "f3")
    assert dio_nodate["progress"]["eda"] == ProgressState.FINISHED.value

    # Fixture 4: Fewer than 3 numeric columns (2 numeric columns -> no heatmap)
    df_2num = pd.DataFrame({"n1": [1.0, 2.0, 3.0], "n2": [10.0, 20.0, 30.0]})
    plans_2num = select_charts_deterministically(df_2num, [{"name": "n1", "dtype_inferred": "float"}, {"name": "n2", "dtype_inferred": "float"}])
    assert not any(p["chart_type"] == "correlation_heatmap" for p in plans_2num)


def test_dio_section_isolation(tmp_path: Path):
    """
    CRITICAL REQUIREMENT:
    Verify that EDA agent mutates ONLY dio['eda'] and dio['artifacts']['chart_paths'].
    """
    df = pd.DataFrame({"val": [10, 20, 30]})
    dio = DIO.create_empty(file_name="iso.csv", dataset_hash="h_iso")
    dio["ingestion"] = {"n_rows": 3, "n_columns": 1, "file_type": "csv", "encoding": "utf-8"}
    dio["columns"] = [{"name": "val", "dtype_inferred": "float"}]
    dio["quality"] = {"score": 100, "issues": []}
    dio["cleaning_log"] = [{"column": "val", "method": "none"}]

    agent = EDAAgent()
    _, updated_dio = agent.run(df, dio, run_dir=tmp_path)

    # Original sections MUST remain untouched
    assert updated_dio["ingestion"]["n_rows"] == 3
    assert updated_dio["quality"]["score"] == 100
    assert len(updated_dio["cleaning_log"]) == 1
    # Future sections MUST remain untouched/empty
    assert updated_dio["ml"]["problem_type"] == "none"
    assert len(updated_dio["insights"]) == 0
    assert updated_dio["reports"]["pdf_path"] is None
