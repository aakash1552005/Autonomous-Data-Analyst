"""
tests/test_chat_agent.py
========================
Comprehensive unit and integration test suite for Agent 7 (Chat Agent).

Verifies:
1. Exact mathematical correctness for all 7 whitelisted operations:
   - mean
   - sum
   - count (column and total rows)
   - min
   - max
   - value_counts
   - groupby_mean
2. Handling of non-numeric columns for arithmetic aggregations.
3. Handling of nonexistent/unsupported columns (strictly unavailable message without hallucination or fuzzy guessing).
4. Tier 2 DIO contextual retrieval and offline heuristic question answering.
5. Analytical DIO immutability (analytical sections must never be mutated by chat).
6. BaseAgent contract compliance and decision logging.
"""

import copy
import pandas as pd
import pytest

from agents.chat.chat_agent import ChatAgent
from agents.chat.query_classifier import classify_query, ClassifiedQuery
from agents.chat.whitelist_executor import execute_whitelisted_operation
from core.config import AppConfig, load_config
from core.dio import DIO


@pytest.fixture
def sample_chat_dataset():
    df = pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0, 400.0],
        "quantity": [1, 2, 3, 4],
        "region": ["North", "North", "South", "South"],
        "customer_name": ["Alice", "Bob", "Charlie", "David"],
        "account_id": ["ACC_001", "ACC_002", "ACC_003", "ACC_004"],
    })
    dio = {
        "dataset_id": "test_chat_run",
        "ingestion": {
            "n_rows": 4,
            "n_columns": 5,
        },
        "columns": [
            {"name": "revenue", "dtype_inferred": "float", "is_pii": False, "semantic_label": "monetary"},
            {"name": "quantity", "dtype_inferred": "int", "is_pii": False, "semantic_label": "quantity"},
            {"name": "region", "dtype_inferred": "string", "is_pii": False, "semantic_label": "categorical"},
            {"name": "customer_name", "dtype_inferred": "string", "is_pii": True, "semantic_label": "name", "pii_type": "name"},
            {"name": "account_id", "dtype_inferred": "string", "is_pii": False, "semantic_label": "identifier"},
        ],
        "domain_guess": {
            "domain": "retail",
            "confidence": 0.95,
        },
        "quality": {
            "score": 92.0,
            "issues": ["Low row count (<30 rows)"],
        },
        "eda": {
            "summary_stats": {
                "revenue": {"type": "numeric", "mean": 250.0, "min": 100.0, "max": 400.0},
                "quantity": {"type": "numeric", "mean": 2.5, "min": 1, "max": 4},
                "region": {"type": "categorical", "unique_count": 2, "top_category": "North"},
                "customer_name": {"type": "categorical", "unique_count": 4, "count": 4},
                "account_id": {"type": "categorical", "unique_count": 4, "count": 4},
            },
            "correlations": {},
            "charts": [],
            "target_candidates": ["revenue"],
        },
        "ml": {
            "model_training_status": "SKIPPED_INSUFFICIENT_DATA",
            "best_model_name": None,
            "metrics": {},
        },
        "insights": [
            {
                "id": "ins_001",
                "category": "quality",
                "text": "Analyzed dataset of 4 rows and 5 columns with an overall data quality score of 92.0/100.",
                "confidence": 1.0,
                "evidence": "dio.quality.score",
                "recommendation": "Expand dataset size to enable machine learning.",
            }
        ],
        "reports": {},
        "decision_log": [],
        "errors": [],
    }
    return df, dio


def test_classifier_all_eleven_operations(sample_chat_dataset):
    df, dio = sample_chat_dataset
    columns = [c["name"] for c in dio["columns"]]

    # 1. mean
    q1 = classify_query("what is the average revenue", columns)
    assert q1.operation == "mean"
    assert q1.column == "revenue"

    # 2. sum
    q2 = classify_query("calculate total quantity", columns)
    assert q2.operation == "sum"
    assert q2.column == "quantity"

    # 3. count
    q3 = classify_query("how many rows in this dataset", columns)
    assert q3.operation == "count"
    assert q3.column is None

    q3_col = classify_query("count of revenue", columns)
    assert q3_col.operation == "count"
    assert q3_col.column == "revenue"

    # 4. min
    q4 = classify_query("what is the minimum revenue", columns)
    assert q4.operation == "min"
    assert q4.column == "revenue"

    # 5. max
    q5 = classify_query("highest quantity recorded", columns)
    assert q5.operation == "max"
    assert q5.column == "quantity"

    # 6. value_counts
    q6 = classify_query("distribution of region", columns)
    assert q6.operation == "value_counts"
    assert q6.column == "region"

    # 7. groupby_mean
    q7 = classify_query("average revenue by region", columns)
    assert q7.operation == "groupby_mean"
    assert q7.column == "revenue"
    assert q7.group_column == "region"

    # 8. median
    q8 = classify_query("what is the median revenue", columns)
    assert q8.operation == "median"
    assert q8.column == "revenue"

    # 9. std
    q9 = classify_query("what is the standard deviation of revenue", columns)
    assert q9.operation == "std"
    assert q9.column == "revenue"

    # 10. variance
    q10 = classify_query("what is the variance of revenue", columns)
    assert q10.operation == "variance"
    assert q10.column == "revenue"

    # 11. groupby_sum
    q11 = classify_query("total revenue by region", columns)
    assert q11.operation == "groupby_sum"
    assert q11.column == "revenue"
    assert q11.group_column == "region"


def test_executor_eleven_operations_exact_math(sample_chat_dataset):
    df, dio = sample_chat_dataset
    columns_info = dio["columns"]

    # 1. mean
    res_mean = execute_whitelisted_operation(df, columns_info, "mean", "revenue")
    assert res_mean.success is True
    assert res_mean.numeric_value == 250.0
    assert "250.0" in res_mean.result_text

    # 2. sum
    res_sum = execute_whitelisted_operation(df, columns_info, "sum", "quantity")
    assert res_sum.success is True
    assert res_sum.numeric_value == 10.0
    assert "10.0" in res_sum.result_text

    # 3. count (total rows and column)
    res_count_rows = execute_whitelisted_operation(df, columns_info, "count", None)
    assert res_count_rows.success is True
    assert res_count_rows.numeric_value == 4.0

    res_count_col = execute_whitelisted_operation(df, columns_info, "count", "revenue")
    assert res_count_col.success is True
    assert res_count_col.numeric_value == 4.0

    # 4. min
    res_min = execute_whitelisted_operation(df, columns_info, "min", "revenue")
    assert res_min.success is True
    assert res_min.numeric_value == 100.0

    # 5. max
    res_max = execute_whitelisted_operation(df, columns_info, "max", "revenue")
    assert res_max.success is True
    assert res_max.numeric_value == 400.0

    # 6. value_counts
    res_vc = execute_whitelisted_operation(df, columns_info, "value_counts", "region")
    assert res_vc.success is True
    assert res_vc.data == {"North": 2, "South": 2}

    # 7. groupby_mean
    res_gb = execute_whitelisted_operation(df, columns_info, "groupby_mean", "revenue", group_column="region")
    assert res_gb.success is True
    assert res_gb.data == {"North": 150.0, "South": 350.0}

    # 8. median
    res_median = execute_whitelisted_operation(df, columns_info, "median", "revenue")
    assert res_median.success is True
    assert res_median.numeric_value == 250.0
    assert "250.0" in res_median.result_text

    # 9. std
    res_std = execute_whitelisted_operation(df, columns_info, "std", "revenue")
    assert res_std.success is True
    assert res_std.numeric_value == 129.0994
    assert "129.0994" in res_std.result_text

    # 10. variance
    res_var = execute_whitelisted_operation(df, columns_info, "variance", "revenue")
    assert res_var.success is True
    assert res_var.numeric_value == 16666.6667
    assert "16666.6667" in res_var.result_text

    # 11. groupby_sum
    res_gb_sum = execute_whitelisted_operation(df, columns_info, "groupby_sum", "revenue", group_column="region")
    assert res_gb_sum.success is True
    assert res_gb_sum.data == {"North": 300.0, "South": 700.0}
    assert "Total of 'revenue' grouped by 'region'" in res_gb_sum.result_text


def test_nonexistent_column_returns_unavailable(sample_chat_dataset):
    df, dio = sample_chat_dataset
    agent = ChatAgent()

    res = agent.run_query("what is the average salary", df, dio)
    assert res["status"] == "unavailable"
    assert "The requested information is not available" in res["response"]


def test_non_numeric_aggregation_fails_gracefully(sample_chat_dataset):
    df, dio = sample_chat_dataset
    agent = ChatAgent()

    # Region is string/categorical; mean cannot be computed
    res = agent.run_query("what is the mean region", df, dio)
    assert res["status"] == "error"
    assert "non-numeric" in res["response"]


def test_tier2_offline_heuristic_qa(sample_chat_dataset):
    df, dio = sample_chat_dataset
    agent = ChatAgent()

    # Quality score
    res_q = agent.run_query("what is the data quality score?", df, dio)
    assert res_q["tier"] == 2
    assert "92.0/100" in res_q["response"]

    # Domain
    res_d = agent.run_query("what domain does this data belong to?", df, dio)
    assert res_d["tier"] == 2
    assert "Retail" in res_d["response"]

    # ML training outcome
    res_ml = agent.run_query("what is the machine learning model status?", df, dio)
    assert res_ml["tier"] == 2
    assert "insufficient data" in res_ml["response"].lower()

    # Executive insights
    res_ins = agent.run_query("what are the key analytical insights?", df, dio)
    assert res_ins["tier"] == 2
    assert "92.0/100" in res_ins["response"]


def test_analytical_dio_immutability(sample_chat_dataset):
    df, dio = sample_chat_dataset
    agent = ChatAgent()

    # Deepcopy analytical sections to verify zero mutation
    original_domain = copy.deepcopy(dio["domain_guess"])
    original_quality = copy.deepcopy(dio["quality"])
    original_columns = copy.deepcopy(dio["columns"])
    original_eda = copy.deepcopy(dio["eda"])
    original_ml = copy.deepcopy(dio["ml"])
    original_insights = copy.deepcopy(dio["insights"])
    original_reports = copy.deepcopy(dio["reports"])

    # Run multiple queries through run() and run_query()
    agent.run_query("what is the average revenue?", df, dio)
    agent.run_query("what is the data quality score?", df, dio)
    agent.run(df, dio, query="total quantity")

    # Assert analytical sections remain 100% identical
    assert dio["domain_guess"] == original_domain
    assert dio["quality"] == original_quality
    assert dio["columns"] == original_columns
    assert dio["eda"] == original_eda
    assert dio["ml"] == original_ml
    assert dio["insights"] == original_insights
    assert dio["reports"] == original_reports

    # Assert decision log records the chat event
    assert len(dio["decision_log"]) >= 1
    assert dio["decision_log"][-1]["agent"] == "chat"
