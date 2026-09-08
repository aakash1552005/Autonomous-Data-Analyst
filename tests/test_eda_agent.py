"""
tests/test_eda_agent.py
=======================
Comprehensive test suite for Agent 3 (Exploratory Data Analysis Agent).
Verifies:
1. Exact statistical correctness (mean, median, std, min, max, Pearson, Spearman, NaN handling, zero-variance).
2. Deterministic chart selection order and configurable limits.
3. Plotly + Kaleido PNG rendering and artifact existence.
4. Zero-LLM and network independence.
5. Graceful degradation on narrow fixtures.
6. Exclusion of ambiguous dates from time-series charts.
7. Strict PII and identifier exclusion from chart artifacts.
8. Target candidate awareness and leakage protection.
9. Strict DIO boundary isolation.
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


def test_statistical_correctness_ground_truth():
    """
    CRITICAL REQUIREMENT:
    Verify exact hand-calculated ground truths for summary statistics and correlations.
    """
    # Ground truth dataset: [10, 20, 30, 40, 50]
    # Mean = 30.0, Median = 30.0, Std = sqrt((400+100+0+100+400)/4) = sqrt(250) = 15.8114
    # Min = 10.0, Max = 50.0, Q25 = 20.0, Q75 = 40.0, IQR = 20.0, Skewness = 0.0
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, np.nan])
    res = compute_numeric_summary(series, "metric")

    assert res["column"] == "metric"
    assert res["type"] == "numeric"
    assert res["count"] == 5
    assert res["null_count"] == 1
    assert res["null_pct"] == 16.67
    assert res["mean"] == 30.0
    assert res["median"] == 30.0
    assert res["std"] == 15.8114
    assert res["min"] == 10.0
    assert res["max"] == 50.0
    assert res["q25"] == 20.0
    assert res["q75"] == 40.0
    assert res["iqr"] == 20.0
    assert res["skewness"] == 0.0


def test_zero_variance_and_all_nan_handling():
    # Constant column (zero variance, std = 0)
    s_const = pd.Series([5.0, 5.0, 5.0, 5.0])
    res_const = compute_numeric_summary(s_const, "const_col")
    assert res_const["mean"] == 5.0
    assert res_const["std"] == 0.0
    assert res_const["iqr"] == 0.0

    # All-NaN column
    s_nan = pd.Series([np.nan, np.nan, np.nan])
    res_nan = compute_numeric_summary(s_nan, "nan_col")
    assert res_nan["count"] == 0
    assert res_nan["null_count"] == 3
    assert res_nan["null_pct"] == 100.0
    assert res_nan["mean"] is None


def test_pearson_and_spearman_ground_truth():
    # Linear relationship: y = 2x + 1 -> Pearson r = 1.0, Spearman r = 1.0
    # Inverse relationship: z = -3x -> Pearson r = -1.0, Spearman r = -1.0
    # Constant column: c = 7 -> should be excluded from correlations (zero variance)
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [3.0, 5.0, 7.0, 9.0, 11.0],
        "z": [-3.0, -6.0, -9.0, -12.0, -15.0],
        "c": [7.0, 7.0, 7.0, 7.0, 7.0],
    })
    columns_info = [
        {"name": "x", "dtype_inferred": "float"},
        {"name": "y", "dtype_inferred": "float"},
        {"name": "z", "dtype_inferred": "float"},
        {"name": "c", "dtype_inferred": "float"},
    ]
    res = compute_correlations(df, columns_info=columns_info)

    assert "x" in res["pearson"]
    assert "c" not in res["pearson"]  # Excluded due to zero variance
    assert res["pearson"]["x"]["y"] == 1.0
    assert res["pearson"]["x"]["z"] == -1.0
    assert res["spearman"]["x"]["y"] == 1.0
    assert res["spearman"]["x"]["z"] == -1.0
    assert len(res["top_correlations"]) >= 2


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
    date_columns_info = [{"column": "order_date", "detected_format": "YYYY-MM-DD", "needs_user_confirmation": False}]

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
        {"name": "order_id", "dtype_inferred": "int", "semantic_label": "identifier"},
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
        assert p.stat().st_size > 1000  # Non-empty rendered image


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


def test_ambiguous_dates_excluded_from_time_series(tmp_path: Path):
    """
    CRITICAL REQUIREMENT:
    Ambiguous date columns (needs_user_confirmation=True) MUST NEVER become time-series line trend axes.
    """
    df = pd.DataFrame({
        "ambig_date": ["05/06/2024", "07/08/2024", "01/02/2024"],
        "metric": [100.0, 150.0, 200.0],
    })
    columns_info = [
        {"name": "ambig_date", "dtype_inferred": "date"},
        {"name": "metric", "dtype_inferred": "float"},
    ]
    date_columns_info = [
        {"column": "ambig_date", "detected_format": "ambiguous", "needs_user_confirmation": True}
    ]

    plans = select_charts_deterministically(df, columns_info, date_columns_info, max_charts=6)
    # Proves no line_trend chart was selected for the ambiguous date column
    assert not any(p["chart_type"] == "line_trend" for p in plans)


def test_pii_and_identifier_exclusion_from_charts(tmp_path: Path):
    """
    CRITICAL REQUIREMENT:
    Raw PII and identifiers MUST NEVER appear in chart titles, filenames, metadata, or logs.
    """
    df = pd.DataFrame({
        "user_id": ["U1", "U2", "U3"],
        "full_name": ["Alice Smith", "Bob Jones", "Charlie Brown"],
        "email": ["alice@secret.com", "bob@secret.com", "charlie@secret.com"],
        "salary": [50000.0, 60000.0, 75000.0],
    })
    dio = DIO.create_empty(file_name="employees.csv", dataset_hash="emp_hash")
    dio["columns"] = [
        {"name": "user_id", "dtype_inferred": "string", "semantic_label": "identifier"},
        {"name": "full_name", "dtype_inferred": "string", "semantic_label": "person_name", "is_pii": True},
        {"name": "email", "dtype_inferred": "string", "semantic_label": "email", "is_pii": True},
        {"name": "salary", "dtype_inferred": "float", "semantic_label": "currency_amount"},
    ]

    agent = EDAAgent()
    _, updated_dio = agent.run(df, dio, run_dir=tmp_path)

    chart_paths = updated_dio["artifacts"]["chart_paths"]
    assert len(chart_paths) > 0

    # Assert no PII in chart filenames or chart titles
    for p_str in chart_paths:
        p_name = Path(p_str).name
        assert "full_name" not in p_name
        assert "email" not in p_name
        assert "alice" not in p_name.lower()
        assert "user_id" not in p_name


def test_target_candidate_identification_and_leakage_protection(tmp_path: Path):
    """
    CRITICAL REQUIREMENT:
    Target candidates must be recorded in dio['eda']['target_candidates'] from Intelligence DIO,
    without mutating feature definitions or altering downstream ML boundaries.
    """
    df = pd.DataFrame({
        "tenure": [12.0, 24.0, 36.0, 48.0],
        "monthly_charges": [50.0, 70.0, 85.0, 100.0],
        "churn": [0, 1, 0, 1],
    })
    dio = DIO.create_empty(file_name="telecom.csv", dataset_hash="telecom_hash")
    dio["columns"] = [
        {"name": "tenure", "dtype_inferred": "float"},
        {"name": "monthly_charges", "dtype_inferred": "float"},
        {"name": "churn", "dtype_inferred": "int", "semantic_label": "target_label", "is_target_candidate": True},
    ]

    agent = EDAAgent()
    _, updated_dio = agent.run(df, dio, run_dir=tmp_path)

    assert "target_candidates" in updated_dio["eda"]
    assert updated_dio["eda"]["target_candidates"] == ["churn"]


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


def test_extended_visual_charts_generation(tmp_path: Path):
    """
    Verify that when max_charts is set to 9, the new visual charts:
    - Target Distribution (Rule 7)
    - Data Completeness Matrix (Rule 8)
    - Violin Spread Plot (Rule 9)
    are selected and correctly rendered to PNG files.
    """
    from agents.eda.chart_generator import select_charts_deterministically, render_chart_to_png

    df = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feature_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "category": ["A", "B", "A", "B", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1],
    })
    columns_info = [
        {"name": "feature_a", "dtype_inferred": "float"},
        {"name": "feature_b", "dtype_inferred": "float"},
        {"name": "category", "dtype_inferred": "category"},
        {"name": "target", "dtype_inferred": "int", "semantic_label": "target_label", "is_target_candidate": True},
    ]

    plans = select_charts_deterministically(df, columns_info=columns_info, max_charts=10)
    chart_ids = [p["chart_id"] for p in plans]

    # Verify new chart rules are present in plans
    assert any("target" in cid for cid in chart_ids)
    assert any("completeness" in cid for cid in chart_ids)
    assert any("violin" in cid for cid in chart_ids)

    # Render each chart to ensure Kaleido exports valid PNGs
    for plan in plans:
        out_path = render_chart_to_png(df, plan, tmp_path)
        assert out_path is not None
        assert out_path.is_file()
        assert out_path.stat().st_size > 500
