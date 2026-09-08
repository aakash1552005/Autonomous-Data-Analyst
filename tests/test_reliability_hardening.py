"""
tests/test_reliability_hardening.py
===================================
Phase 15: Reliability Hardening.
Validates graceful handling of None values, empty DataFrames, missing DIO sections,
partial failures, malformed files, duplicate requests, and unsupported data types.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from core.config import load_config
from core.dio import DIO
from agents.chat.chat_agent import ChatAgent
from agents.chat.whitelist_executor import execute_whitelisted_operation


# ─── Phase 15.1: None/Empty DataFrame Handling ─────────────────────────────

class TestNoneDataFrameHandling:
    """Verify chat agent handles None DataFrame gracefully."""

    def test_chat_with_none_df(self):
        agent = ChatAgent()
        dio = {
            "columns": [],
            "quality": {"score": 0},
            "domain_guess": {"domain": "unknown", "confidence": 0},
            "ingestion": {"n_rows": 0, "n_columns": 0},
            "insights": [], "ml": {}, "eda": {},
            "decision_log": [], "progress": [], "errors": [], "agent_metrics": {},
        }
        result = agent.run_query("what is the average revenue", None, dio)
        assert result["status"] in ("unavailable", "success")
        # Must NOT crash

    def test_chat_with_empty_df(self):
        agent = ChatAgent()
        df = pd.DataFrame()
        dio = {
            "columns": [],
            "quality": {"score": 0},
            "domain_guess": {"domain": "unknown", "confidence": 0},
            "ingestion": {"n_rows": 0, "n_columns": 0},
            "insights": [], "ml": {}, "eda": {},
            "decision_log": [], "progress": [], "errors": [], "agent_metrics": {},
        }
        result = agent.run_query("what is the average revenue", df, dio)
        assert result is not None
        assert "status" in result

    def test_empty_query(self):
        agent = ChatAgent()
        result = agent.run_query("", None, {"columns": [], "decision_log": []})
        assert result["status"] == "unavailable"


# ─── Phase 15.2: Missing DIO Sections ──────────────────────────────────────

class TestMissingDIOSections:
    """Verify chat agent handles missing DIO sections without crashing."""

    def test_missing_columns_key(self):
        agent = ChatAgent()
        dio = {"decision_log": [], "progress": [], "errors": [], "agent_metrics": {}}
        result = agent.run_query("quality score", None, dio)
        assert result is not None

    def test_missing_quality_key(self):
        agent = ChatAgent()
        dio = {"columns": [], "decision_log": [], "progress": [], "errors": [], "agent_metrics": {}}
        result = agent.run_query("what is the quality score", None, dio)
        assert result is not None

    def test_missing_ml_key(self):
        agent = ChatAgent()
        dio = {"columns": [], "decision_log": [], "progress": [], "errors": [], "agent_metrics": {}}
        result = agent.run_query("what model was trained", None, dio)
        assert result is not None


# ─── Phase 15.3: Partial Failure Resilience ─────────────────────────────────

class TestPartialFailureResilience:
    """Verify pipeline stages handle downstream failures gracefully."""

    def test_executor_with_nan_only_column(self):
        """Column with all NaN values should not crash."""
        df = pd.DataFrame({"revenue": [float("nan"), float("nan"), float("nan")]})
        info = [{"name": "revenue", "dtype": "float", "is_pii": False, "semantic_label": "metric"}]
        result = execute_whitelisted_operation(df, info, "mean", "revenue")
        assert result.success is True
        assert result.numeric_value is None

    def test_executor_with_inf_values(self):
        """Column with infinity values should handle gracefully."""
        import numpy as np
        df = pd.DataFrame({"revenue": [1.0, 2.0, float("inf")]})
        info = [{"name": "revenue", "dtype": "float", "is_pii": False, "semantic_label": "metric"}]
        result = execute_whitelisted_operation(df, info, "mean", "revenue")
        # Should not crash — inf in pandas mean is inf
        assert result.success is True

    def test_executor_with_mixed_types(self):
        """Object column that can be coerced should not crash numeric ops."""
        df = pd.DataFrame({"revenue": [1, 2, "three", 4]})
        info = [{"name": "revenue", "dtype": "object", "is_pii": False, "semantic_label": "metric"}]
        # This is non-numeric dtype, so should return error
        result = execute_whitelisted_operation(df, info, "mean", "revenue")
        assert result is not None
        assert result.status in ("error", "success")


# ─── Phase 15.4: Unsupported Data Types ────────────────────────────────────

class TestUnsupportedDataTypes:
    """Verify operations on complex/unsupported dtypes don't crash."""

    def test_value_counts_on_empty_strings(self):
        df = pd.DataFrame({"category": ["", "", ""]})
        info = [{"name": "category", "dtype": "object", "is_pii": False, "semantic_label": "category"}]
        result = execute_whitelisted_operation(df, info, "value_counts", "category")
        assert result.success is True

    def test_groupby_with_single_group(self):
        df = pd.DataFrame({"revenue": [10.0, 20.0], "category": ["A", "A"]})
        info = [
            {"name": "revenue", "dtype": "float", "is_pii": False, "semantic_label": "metric"},
            {"name": "category", "dtype": "category", "is_pii": False, "semantic_label": "category"},
        ]
        result = execute_whitelisted_operation(
            df, info, "groupby_mean", "revenue", group_column="category"
        )
        assert result.success is True
        assert result.data.get("A") == 15.0
