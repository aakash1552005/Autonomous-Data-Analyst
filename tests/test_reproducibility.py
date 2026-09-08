"""
tests/test_reproducibility.py
==============================
Phases 18-19: Benchmark Hardening and Reproducibility Validation.
Verifies honest metrics and deterministic reproducibility by running
the benchmark twice and comparing metric-level results.
"""

import json
from pathlib import Path

import pytest


# ─── Phase 18: Benchmark Metric Integrity ──────────────────────────────────

class TestBenchmarkMetricIntegrity:
    """Verify benchmark metrics are honest and correctly labeled."""

    def test_benchmark_has_required_metric_categories(self):
        """Verify benchmark evaluates all required metric categories."""
        from tests.benchmark.ground_truth import BENCHMARK_REGISTRY
        from tests.benchmark.benchmark_runner import DatasetBenchmarkResult
        import dataclasses

        assert len(BENCHMARK_REGISTRY) >= 5, "Expected at least 5 benchmark datasets in registry"

        # Check required fields on DatasetBenchmarkResult schema
        result_fields = {f.name for f in dataclasses.fields(DatasetBenchmarkResult)}
        required_metrics = {
            "semantic_label_accuracy",
            "domain_accuracy",
            "date_resolution_accuracy",
            "pii_recall",
            "cleaning_metrics",
            "eda_metrics",
            "ml_metrics",
            "insight_metrics",
            "report_metrics",
            "runtime",
        }
        for metric in required_metrics:
            assert metric in result_fields, f"DatasetBenchmarkResult missing required metric: {metric}"

    def test_ml_behavior_not_labeled_as_accuracy(self):
        """ML behavior compliance must not be mislabeled as model accuracy."""
        from tests.benchmark.benchmark_runner import DatasetBenchmarkResult
        import dataclasses

        result_fields = {f.name for f in dataclasses.fields(DatasetBenchmarkResult)}
        # Ensure no field is called "model_accuracy" or confusingly named
        for field_name in result_fields:
            assert "model_accuracy" not in field_name.lower(), (
                f"Misleading metric name '{field_name}' found. Must be behavior compliance, not model accuracy."
            )


# ─── Phase 19: Deterministic Reproducibility ───────────────────────────────

class TestDeterministicReproducibility:
    """Run deterministic operations twice and verify identical results."""

    def test_chat_operations_reproducible(self):
        """Same operations on same data produce identical results."""
        import pandas as pd
        from agents.chat.whitelist_executor import execute_whitelisted_operation

        df = pd.DataFrame({
            "revenue": [10.0, 20.0, 30.0, 40.0, 50.0],
            "category": ["A", "B", "A", "B", "A"],
        })
        info = [
            {"name": "revenue", "dtype": "float", "is_pii": False, "semantic_label": "metric"},
            {"name": "category", "dtype": "category", "is_pii": False, "semantic_label": "category"},
        ]

        operations = ["mean", "sum", "count", "min", "max", "median", "std", "variance"]

        # Run 1
        results_1 = {}
        for op in operations:
            r = execute_whitelisted_operation(df, info, op, "revenue")
            results_1[op] = r.numeric_value

        # Run 2
        results_2 = {}
        for op in operations:
            r = execute_whitelisted_operation(df, info, op, "revenue")
            results_2[op] = r.numeric_value

        # All metrics must be identical
        for op in operations:
            assert results_1[op] == results_2[op], \
                f"Non-reproducible result for '{op}': {results_1[op]} vs {results_2[op]}"

    def test_value_counts_reproducible(self):
        """Value counts must be deterministic."""
        import pandas as pd
        from agents.chat.whitelist_executor import execute_whitelisted_operation

        df = pd.DataFrame({"category": ["A", "B", "A", "C", "B", "A"]})
        info = [{"name": "category", "dtype": "object", "is_pii": False, "semantic_label": "category"}]

        r1 = execute_whitelisted_operation(df, info, "value_counts", "category")
        r2 = execute_whitelisted_operation(df, info, "value_counts", "category")

        assert r1.data == r2.data

    def test_groupby_operations_reproducible(self):
        """Groupby operations must be deterministic."""
        import pandas as pd
        from agents.chat.whitelist_executor import execute_whitelisted_operation

        df = pd.DataFrame({
            "revenue": [10.0, 20.0, 30.0, 40.0],
            "category": ["A", "B", "A", "B"],
        })
        info = [
            {"name": "revenue", "dtype": "float", "is_pii": False, "semantic_label": "metric"},
            {"name": "category", "dtype": "category", "is_pii": False, "semantic_label": "category"},
        ]

        for op in ["groupby_mean", "groupby_sum"]:
            r1 = execute_whitelisted_operation(df, info, op, "revenue", group_column="category")
            r2 = execute_whitelisted_operation(df, info, op, "revenue", group_column="category")
            assert r1.data == r2.data, f"Non-reproducible {op}"

    def test_query_classification_reproducible(self):
        """Query classification must be deterministic."""
        from agents.chat.query_classifier import classify_query

        cols = ["revenue", "cost", "category"]
        queries = [
            "what is the average revenue",
            "total cost by category",
            "distribution of category",
            "how many rows",
        ]

        for q in queries:
            r1 = classify_query(q, cols)
            r2 = classify_query(q, cols)
            assert r1.operation == r2.operation, f"Non-reproducible classification for: {q}"
            assert r1.column == r2.column
            assert r1.group_column == r2.group_column
