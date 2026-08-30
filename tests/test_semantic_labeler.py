"""
tests/test_semantic_labeler.py
==============================
Tests for 3-tiered semantic column labeling (Rules, Value Patterns, LLM fallback).
Verifies confidence hierarchies and bounded LLM call restrictions.
"""

from unittest.mock import MagicMock
import pandas as pd
import pytest

from agents.intelligence.semantic_labeler import SemanticLabeler
from llm.base import LLMProvider, LLMResponse, TokenGovernor


class DummyTrackingLLM(LLMProvider):
    def __init__(self, governor: TokenGovernor | None = None, return_label: str = "custom_metric"):
        super().__init__(governor=governor)
        self.return_label = return_label
        self.calls = 0

    def _call_provider(self, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=self.return_label,
            prompt_tokens=15,
            completion_tokens=2,
            total_tokens=17,
            model="mock",
            provider="mock",
        )


def test_semantic_labeler_tier1_rules():
    labeler = SemanticLabeler()

    # Rule checks on column names
    label, conf, method = labeler.label_column("annual_revenue", pd.Series([100000, 250000]))
    assert label == "currency_amount"
    assert conf == 0.90
    assert method == "rule"

    label, conf, method = labeler.label_column("is_churn", pd.Series([0, 1]))
    assert label == "target_label"
    assert conf == 0.90
    assert method == "rule"

    label, conf, method = labeler.label_column("cust_id", pd.Series([1, 2, 3]))
    assert label == "identifier"
    assert conf == 0.90
    assert method == "rule"


def test_semantic_labeler_tier2_patterns():
    labeler = SemanticLabeler()

    # UUID values in generic column name "code_val"
    uuids = [
        "c9a646d3-9c61-4cc9-bc12-ae963f6d41a3",
        "d8b757e4-0d72-4dd0-cd23-bf074a7e52b4",
    ]
    label, conf, method = labeler.label_column("unnamed_field", pd.Series(uuids))
    assert label == "identifier"
    assert conf == 0.85
    assert method == "pattern"

    # Currency symbols in non-descriptive header
    curr_values = ["$10.50", "$25.00", "$99.99"]
    label, conf, method = labeler.label_column("metric_x", pd.Series(curr_values))
    assert label == "currency_amount"
    assert conf == 0.80
    assert method == "pattern"


def test_semantic_labeler_tier3_llm_fallback():
    mock_llm = DummyTrackingLLM(return_label="product_category")
    labeler = SemanticLabeler(llm_provider=mock_llm)

    # Column that does not match Tier 1 or Tier 2
    obscure_series = pd.Series(["Hardware", "Electronics", "Apparel", "Home"])
    label, conf, method = labeler.label_column("field_q99", obscure_series)

    assert label == "product_category"
    assert conf == 0.60
    assert method == "llm"
    assert mock_llm.calls == 1


def test_semantic_labeler_llm_called_only_for_unresolved():
    mock_llm = DummyTrackingLLM(return_label="something")
    labeler = SemanticLabeler(llm_provider=mock_llm)

    # Clear rule-based column -> LLM should NOT be called
    labeler.label_column("customer_email", pd.Series(["a@b.com", "c@d.com"]))
    assert mock_llm.calls == 0
