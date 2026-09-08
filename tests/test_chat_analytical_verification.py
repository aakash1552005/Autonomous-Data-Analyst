"""
tests/test_chat_analytical_verification.py
==========================================
Phase 12: Chat Analytical Verification.
Validates mathematical correctness of all 11 whitelisted operations,
query classification, schema-bound column resolution, missing-column handling,
empty data handling, and deterministic responses.
"""

import math
import numpy as np
import pandas as pd
import pytest

from agents.chat.whitelist_executor import (
    execute_whitelisted_operation,
    is_column_sensitive,
    ALLOWED_OPERATIONS,
)
from agents.chat.query_classifier import classify_query, ClassifiedQuery


# ─── Fixture Datasets ───────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Known-value dataset for exact mathematical verification."""
    return pd.DataFrame({
        "revenue": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cost": [5.0, 10.0, 15.0, 20.0, 25.0],
        "category": ["A", "B", "A", "B", "A"],
        "region": ["East", "West", "East", "West", "East"],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    })


@pytest.fixture
def columns_info():
    """Safe columns info (no PII)."""
    return [
        {"name": "revenue", "dtype": "float", "is_pii": False, "semantic_label": "metric"},
        {"name": "cost", "dtype": "float", "is_pii": False, "semantic_label": "metric"},
        {"name": "category", "dtype": "category", "is_pii": False, "semantic_label": "category"},
        {"name": "region", "dtype": "category", "is_pii": False, "semantic_label": "category"},
        {"name": "name", "dtype": "string", "is_pii": True, "semantic_label": "person_name"},
    ]


@pytest.fixture
def empty_df():
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame({"revenue": pd.Series([], dtype="float64"), "category": pd.Series([], dtype="object")})


@pytest.fixture
def empty_columns_info():
    return [
        {"name": "revenue", "dtype": "float", "is_pii": False, "semantic_label": "metric"},
        {"name": "category", "dtype": "category", "is_pii": False, "semantic_label": "category"},
    ]


# ─── Phase 12.1: All 11 Operations Present ─────────────────────────────────

class TestOperationWhitelistCompleteness:
    """Verify the whitelist contains exactly the approved 11 operations."""

    def test_exactly_11_operations(self):
        assert len(ALLOWED_OPERATIONS) == 11

    def test_all_approved_operations_present(self):
        expected = {
            "mean", "sum", "count", "min", "max", "value_counts",
            "groupby_mean", "median", "std", "variance", "groupby_sum",
        }
        assert ALLOWED_OPERATIONS == expected


# ─── Phase 12.2: Mathematical Correctness ───────────────────────────────────

class TestMathematicalCorrectness:
    """Verify exact mathematical output for each operation."""

    def test_mean_correctness(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "mean", "revenue")
        assert result.success is True
        assert result.status == "success"
        # mean([10,20,30,40,50]) = 30.0
        assert result.numeric_value == 30.0

    def test_sum_correctness(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "sum", "revenue")
        assert result.success is True
        assert result.numeric_value == 150.0

    def test_count_column_correctness(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "count", "revenue")
        assert result.success is True
        assert result.numeric_value == 5.0

    def test_count_rows_correctness(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "count", None)
        assert result.success is True
        assert result.numeric_value == 5.0

    def test_min_correctness(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "min", "revenue")
        assert result.success is True
        assert result.numeric_value == 10.0

    def test_max_correctness(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "max", "revenue")
        assert result.success is True
        assert result.numeric_value == 50.0

    def test_median_correctness(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "median", "revenue")
        assert result.success is True
        # median([10,20,30,40,50]) = 30.0
        assert result.numeric_value == 30.0

    def test_value_counts_correctness(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "value_counts", "category")
        assert result.success is True
        assert result.status == "success"
        assert result.data is not None
        assert result.data.get("A") == 3
        assert result.data.get("B") == 2

    def test_groupby_mean_correctness(self, sample_df, columns_info):
        result = execute_whitelisted_operation(
            sample_df, columns_info, "groupby_mean", "revenue", group_column="category"
        )
        assert result.success is True
        assert result.data is not None
        # A: mean(10,30,50) = 30.0; B: mean(20,40) = 30.0
        assert result.data.get("A") == 30.0
        assert result.data.get("B") == 30.0

    def test_groupby_sum_correctness(self, sample_df, columns_info):
        result = execute_whitelisted_operation(
            sample_df, columns_info, "groupby_sum", "revenue", group_column="category"
        )
        assert result.success is True
        assert result.data is not None
        # A: sum(10,30,50) = 90.0; B: sum(20,40) = 60.0
        assert result.data.get("A") == 90.0
        assert result.data.get("B") == 60.0


# ─── Phase 12.3: std and variance ddof=1 Verification ──────────────────────

class TestSampleStatisticsSemantics:
    """Verify std and variance use ddof=1 (sample statistics, not population)."""

    def test_std_uses_ddof1(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "std", "revenue")
        assert result.success is True
        # Sample std of [10,20,30,40,50] with ddof=1
        expected_std = round(float(pd.Series([10, 20, 30, 40, 50]).std(ddof=1)), 4)
        assert result.numeric_value == expected_std
        # Verify it's NOT population std
        pop_std = round(float(pd.Series([10, 20, 30, 40, 50]).std(ddof=0)), 4)
        assert result.numeric_value != pop_std

    def test_variance_uses_ddof1(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "variance", "revenue")
        assert result.success is True
        # Sample variance of [10,20,30,40,50] with ddof=1
        expected_var = round(float(pd.Series([10, 20, 30, 40, 50]).var(ddof=1)), 4)
        assert result.numeric_value == expected_var
        # Verify it's NOT population variance
        pop_var = round(float(pd.Series([10, 20, 30, 40, 50]).var(ddof=0)), 4)
        assert result.numeric_value != pop_var

    def test_std_single_element_returns_zero(self, columns_info):
        """Single-element std should be 0.0 (no variation)."""
        df = pd.DataFrame({"revenue": [42.0], "category": ["A"]})
        result = execute_whitelisted_operation(df, columns_info, "std", "revenue")
        assert result.success is True
        assert result.numeric_value == 0.0

    def test_variance_single_element_returns_zero(self, columns_info):
        """Single-element variance should be 0.0."""
        df = pd.DataFrame({"revenue": [42.0], "category": ["A"]})
        result = execute_whitelisted_operation(df, columns_info, "variance", "revenue")
        assert result.success is True
        assert result.numeric_value == 0.0

    def test_std_variance_consistency(self, sample_df, columns_info):
        """std^2 should equal variance."""
        std_result = execute_whitelisted_operation(sample_df, columns_info, "std", "revenue")
        var_result = execute_whitelisted_operation(sample_df, columns_info, "variance", "revenue")
        assert std_result.success is True and var_result.success is True
        # Allow small rounding tolerance
        assert abs(std_result.numeric_value ** 2 - var_result.numeric_value) < 0.01


# ─── Phase 12.4: Missing Column Handling ────────────────────────────────────

class TestMissingColumnHandling:
    """Verify graceful handling of nonexistent columns."""

    def test_mean_missing_column(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "mean", "nonexistent_col")
        assert result.success is False
        assert result.status == "unavailable"

    def test_sum_missing_column(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "sum", "nonexistent_col")
        assert result.success is False
        assert result.status == "unavailable"

    def test_groupby_missing_target_col(self, sample_df, columns_info):
        result = execute_whitelisted_operation(
            sample_df, columns_info, "groupby_mean", "nonexistent", group_column="category"
        )
        assert result.success is False
        assert result.status == "unavailable"

    def test_groupby_missing_group_col(self, sample_df, columns_info):
        result = execute_whitelisted_operation(
            sample_df, columns_info, "groupby_mean", "revenue", group_column="nonexistent"
        )
        assert result.success is False
        assert result.status == "unavailable"


# ─── Phase 12.5: Empty DataFrame Handling ───────────────────────────────────

class TestEmptyDataFrameHandling:
    """Verify operations handle zero-row DataFrames gracefully."""

    def test_mean_empty_df(self, empty_df, empty_columns_info):
        result = execute_whitelisted_operation(empty_df, empty_columns_info, "mean", "revenue")
        assert result.success is True
        assert result.numeric_value is None

    def test_sum_empty_df(self, empty_df, empty_columns_info):
        result = execute_whitelisted_operation(empty_df, empty_columns_info, "sum", "revenue")
        # sum of empty should be 0
        assert result.success is True

    def test_count_empty_df(self, empty_df, empty_columns_info):
        result = execute_whitelisted_operation(empty_df, empty_columns_info, "count", None)
        assert result.success is True
        assert result.numeric_value == 0.0

    def test_std_empty_df(self, empty_df, empty_columns_info):
        result = execute_whitelisted_operation(empty_df, empty_columns_info, "std", "revenue")
        assert result.success is True
        assert result.numeric_value is None

    def test_variance_empty_df(self, empty_df, empty_columns_info):
        result = execute_whitelisted_operation(empty_df, empty_columns_info, "variance", "revenue")
        assert result.success is True
        assert result.numeric_value is None


# ─── Phase 12.6: Non-Numeric Column Handling ────────────────────────────────

class TestNonNumericColumnHandling:
    """Verify numeric operations on non-numeric columns return proper errors."""

    def test_mean_on_string_column(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "mean", "category")
        assert result.success is False
        assert result.status == "error"

    def test_std_on_string_column(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "std", "category")
        assert result.success is False
        assert result.status == "error"

    def test_variance_on_string_column(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "variance", "category")
        assert result.success is False
        assert result.status == "error"

    def test_groupby_sum_non_numeric_target(self, sample_df, columns_info):
        result = execute_whitelisted_operation(
            sample_df, columns_info, "groupby_sum", "category", group_column="region"
        )
        assert result.success is False
        assert result.status == "error"


# ─── Phase 12.7: Unsupported Operation Handling ────────────────────────────

class TestUnsupportedOperations:
    """Verify that operations outside the whitelist are rejected."""

    def test_unsupported_operation_rejected(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "correlation", "revenue")
        assert result.success is False
        assert result.status == "refusal"

    def test_arbitrary_string_rejected(self, sample_df, columns_info):
        result = execute_whitelisted_operation(sample_df, columns_info, "exec", "revenue")
        assert result.success is False
        assert result.status == "refusal"


# ─── Phase 12.8: PII Shield on All Operations ──────────────────────────────

class TestPIIShieldAllOperations:
    """Verify PII shield blocks access to sensitive columns for all operations."""

    @pytest.mark.parametrize("operation", [
        "mean", "sum", "count", "min", "max", "median", "std", "variance", "value_counts",
    ])
    def test_pii_column_blocked(self, sample_df, columns_info, operation):
        result = execute_whitelisted_operation(sample_df, columns_info, operation, "name")
        assert result.status == "refusal"
        assert "restricted" in result.result_text.lower() or "refusal" in result.result_text.lower()


# ─── Phase 12.9: Query Classification Verification ─────────────────────────

class TestQueryClassification:
    """Verify deterministic query classification for all 11 operations."""

    def test_mean_classification(self):
        cols = ["revenue", "cost", "category"]
        result = classify_query("what is the average revenue", cols)
        assert result.operation == "mean"
        assert result.column == "revenue"

    def test_sum_classification(self):
        cols = ["revenue", "cost", "category"]
        result = classify_query("what is the total revenue", cols)
        assert result.operation == "sum"
        assert result.column == "revenue"

    def test_count_classification(self):
        cols = ["revenue", "cost", "category"]
        result = classify_query("how many rows", cols)
        assert result.operation == "count"

    def test_min_classification(self):
        cols = ["revenue", "cost", "category"]
        result = classify_query("what is the minimum cost", cols)
        assert result.operation == "min"
        assert result.column == "cost"

    def test_max_classification(self):
        cols = ["revenue", "cost", "category"]
        result = classify_query("what is the maximum revenue", cols)
        assert result.operation == "max"
        assert result.column == "revenue"

    def test_median_classification(self):
        cols = ["revenue", "cost", "category"]
        result = classify_query("what is the median revenue", cols)
        assert result.operation == "median"
        assert result.column == "revenue"

    def test_std_classification(self):
        cols = ["revenue", "cost", "category"]
        result = classify_query("what is the standard deviation of revenue", cols)
        assert result.operation == "std"
        assert result.column == "revenue"

    def test_variance_classification(self):
        cols = ["revenue", "cost", "category"]
        result = classify_query("what is the variance of cost", cols)
        assert result.operation == "variance"
        assert result.column == "cost"

    def test_value_counts_classification(self):
        cols = ["revenue", "cost", "category"]
        result = classify_query("what is the distribution of category", cols)
        assert result.operation == "value_counts"
        assert result.column == "category"

    def test_groupby_mean_classification(self):
        cols = ["revenue", "cost", "category"]
        result = classify_query("average revenue by category", cols)
        assert result.operation == "groupby_mean"
        assert result.column == "revenue"
        assert result.group_column == "category"

    def test_groupby_sum_classification(self):
        cols = ["revenue", "cost", "category"]
        result = classify_query("total revenue by category", cols)
        assert result.operation == "groupby_sum"
        assert result.column == "revenue"
        assert result.group_column == "category"

    def test_unrecognized_query_unclassified(self):
        cols = ["revenue", "cost", "category"]
        result = classify_query("tell me about the weather", cols)
        assert result.operation == "unclassified"

    def test_empty_query_unclassified(self):
        cols = ["revenue", "cost"]
        result = classify_query("", cols)
        assert result.operation == "unclassified"


# ─── Phase 12.10: Deterministic Response Verification ──────────────────────

class TestDeterministicResponses:
    """Verify same input always produces same output."""

    def test_repeated_mean_is_deterministic(self, sample_df, columns_info):
        results = [
            execute_whitelisted_operation(sample_df, columns_info, "mean", "revenue")
            for _ in range(5)
        ]
        values = [r.numeric_value for r in results]
        assert all(v == values[0] for v in values)

    def test_repeated_std_is_deterministic(self, sample_df, columns_info):
        results = [
            execute_whitelisted_operation(sample_df, columns_info, "std", "revenue")
            for _ in range(5)
        ]
        values = [r.numeric_value for r in results]
        assert all(v == values[0] for v in values)

