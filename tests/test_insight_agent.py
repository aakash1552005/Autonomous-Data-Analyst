"""
tests/test_insight_agent.py
===========================
Unit tests for the Insight & Narrative Agent (Agent 7).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd

from core.dio import DIO
from core.config import AppConfig
from llm.base import LLMProvider, LLMResponse
from agents.insight.insight_agent import InsightAgent
from agents.insight.evidence_collector import extract_all_dio_numbers, collect_evidence_summary
from agents.insight.hallucination_guard import (
    extract_numbers_from_text,
    is_number_grounded,
    verify_insight_grounding,
)
from agents.insight.deterministic_engine import generate_deterministic_insights


class MockLLMProvider(LLMProvider):
    def __init__(self, mock_response_text: str):
        super().__init__()
        self.mock_response_text = mock_response_text

    def _call_provider(self, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        return LLMResponse(
            text=self.mock_response_text,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="mock_model",
            provider="mock",
        )


@pytest.fixture
def sample_dio() -> DIO:
    dio = DIO.create_empty(file_name="sample_sales.csv", dataset_hash="h_sales")
    dio["ingestion"] = {"n_rows": 100, "n_columns": 5, "file_type": "csv", "encoding": "utf-8"}
    dio["quality"] = {"score": 92, "issues": ["2.0% nulls in column 'region'"]}
    dio["domain_guess"] = {"domain": "retail", "confidence": 0.85}
    dio["columns"] = [
        {"name": "sales", "dtype_inferred": "float", "semantic_label": "currency_amount", "is_pii": False},
        {"name": "profit", "dtype_inferred": "float", "semantic_label": "currency_amount", "is_pii": False},
        {"name": "region", "dtype_inferred": "category", "semantic_label": "location", "is_pii": False},
    ]
    dio["eda"]["summary_stats"] = {
        "numeric": {
            "sales": {"mean": 250.50, "median": 200.0, "std": 50.0, "min": 100.0, "max": 500.0},
            "profit": {"mean": 45.20, "median": 40.0, "std": 15.0, "min": 5.0, "max": 120.0},
        },
        "categorical": {
            "region": {"unique_count": 4, "top_categories": {"East": 40, "West": 30, "North": 20, "South": 10}},
        },
    }
    dio["eda"]["correlations"] = {
        "pearson": {
            "sales": {"sales": 1.0, "profit": 0.82},
            "profit": {"sales": 0.82, "profit": 1.0},
        },
        "top_correlations": [
            {"col1": "sales", "col2": "profit", "pearson": 0.82, "spearman": 0.79},
        ],
    }
    dio["ml"] = {
        "status": "trained",
        "task_type": "regression",
        "target_column": "profit",
        "selected_model": "random_forest",
        "metrics": {"rmse": 12.35, "mae": 9.40, "r2": 0.78},
        "feature_importance": {"sales": 0.65},
    }
    return dio


def test_extract_all_dio_numbers(sample_dio: DIO):
    ints, floats = extract_all_dio_numbers(sample_dio)
    assert 100 in ints      # n_rows
    assert 92 in ints       # quality score
    assert 250.50 in floats # sales mean
    assert 0.82 in floats   # correlation
    assert 12.35 in floats  # ML RMSE
    assert 0.78 in floats   # ML R2


def test_extract_numbers_from_text():
    text = "Average profit increased by 14.5% from $100 to 250.50 across 1,000 customers."
    nums = extract_numbers_from_text(text)
    assert nums == [14.5, 100.0, 250.50, 1000.0]


def test_is_number_grounded():
    grounded_ints = {100}
    grounded_floats = {250.50, 45.20, 0.82}
    assert is_number_grounded(250.50, grounded_ints, grounded_floats, tolerance=0.05) is True
    assert is_number_grounded(250.0, grounded_ints, grounded_floats, tolerance=0.05) is True  # within 5%
    assert is_number_grounded(82.0, grounded_ints, grounded_floats, tolerance=0.05) is True   # percentage equivalence to 0.82
    assert is_number_grounded(999.0, grounded_ints, grounded_floats, tolerance=0.05) is False
    assert is_number_grounded(99.9, grounded_ints, grounded_floats, tolerance=0.05) is False


def test_verify_insight_grounding_success(sample_dio: DIO):
    grounded_ints, grounded_floats = extract_all_dio_numbers(sample_dio)
    valid_insight = {
        "text": "Average sales were 250.50 with an overall correlation of 0.82 to profit across 100 records.",
    }
    is_ok, nums, reason = verify_insight_grounding(
        valid_insight,
        grounded_ints=grounded_ints,
        grounded_floats=grounded_floats,
    )
    assert is_ok is True
    assert nums == [250.50, 0.82, 100.0]
    assert reason == ""


def test_verify_insight_grounding_failure_ungrounded(sample_dio: DIO):
    grounded_ints, grounded_floats = extract_all_dio_numbers(sample_dio)
    invalid_insight = {
        "text": "Average sales reached 999.99 with 99.9% customer retention.",
    }
    is_ok, nums, reason = verify_insight_grounding(
        invalid_insight,
        grounded_ints=grounded_ints,
        grounded_floats=grounded_floats,
    )
    assert is_ok is False
    assert "Ungrounded numerical claim" in reason


def test_verify_insight_grounding_failure_no_numbers():
    invalid_insight = {
        "text": "Sales and profits appear to be very good in all regions.",
    }
    is_ok, nums, reason = verify_insight_grounding(
        invalid_insight,
        grounded_ints={100},
        grounded_floats=set(),
    )
    assert is_ok is False
    assert "zero numbers" in reason


def test_deterministic_insights_generation(sample_dio: DIO):
    insights = generate_deterministic_insights(sample_dio, max_insights=5)
    assert len(insights) >= 3
    assert all("id" in ins for ins in insights)
    assert all("text" in ins for ins in insights)
    assert all("confidence" in ins for ins in insights)
    assert all("evidence" in ins for ins in insights)
    assert all("recommendation" in ins for ins in insights)


def test_full_insight_agent_with_mock_llm(sample_dio: DIO, tmp_path: Path):
    mock_llm_json = json.dumps([
        {
            "category": "distribution",
            "text": "Average sales reached 250.50 with a median of 200.0 across 100 records.",
            "confidence": 0.95,
            "evidence": "eda.summary_stats.sales.mean",
            "recommendation": "Target promotions around the 250.50 average sales range.",
        },
        {
            "category": "correlation",
            "text": "Strong linear correlation of 0.82 observed between sales and profit.",
            "confidence": 0.90,
            "evidence": "eda.correlations.pearson.sales.profit",
            "recommendation": "Leverage sales volume expansion to directly improve profit.",
        },
        {
            "category": "machine_learning",
            "text": "Random forest regression model achieved an RMSE of 12.35 predicting profit.",
            "confidence": 0.85,
            "evidence": "ml.metrics.rmse",
            "recommendation": "Utilize model forecasts for margin management.",
        }
    ])

    mock_llm = MockLLMProvider(mock_llm_json)
    agent = InsightAgent(llm_provider=mock_llm)
    df = pd.DataFrame({"sales": [250.5] * 100, "profit": [45.2] * 100})

    _, updated_dio = agent.run(df, sample_dio, run_dir=tmp_path)

    assert len(updated_dio["insights"]) >= 3
    assert updated_dio["progress"]["insight"] == "FINISHED"
    assert updated_dio["insights"][0]["id"] == "ins_001"
    assert updated_dio["insights"][0]["category"] == "distribution"
    assert "250.50" in updated_dio["insights"][0]["text"]
