"""
tests/test_metrics.py
=====================
Tests for agent metrics tracking and `metrics.json` persistence.
"""

from pathlib import Path
from core.dio import DIO
from core.persistence import save_metrics_json, load_metrics_json


def test_agent_metrics_in_dio():
    dio = DIO.create_empty(file_name="financials.csv")

    # Simulate sequential agents appending their metrics
    dio.agent_metrics.append({
        "agent": "intelligence",
        "runtime_seconds": 1.8,
        "warnings": 0,
        "errors": 0,
    })
    dio.agent_metrics.append({
        "agent": "cleaning",
        "runtime_seconds": 0.9,
        "warnings": 1,
        "errors": 0,
    })
    dio.agent_metrics.append({
        "agent": "eda",
        "runtime_seconds": 3.4,
        "warnings": 0,
        "errors": 0,
    })

    assert len(dio.agent_metrics) == 3
    assert dio.agent_metrics[0]["agent"] == "intelligence"
    assert dio.agent_metrics[1]["runtime_seconds"] == 0.9
    assert dio.agent_metrics[2]["agent"] == "eda"


def test_save_and_load_metrics_json(tmp_path: Path):
    metrics_summary = {
        "dataset": "financials.csv",
        "runtime_seconds": 6.1,
        "agents": {
            "intelligence": 1.8,
            "cleaning": 0.9,
            "eda": 3.4,
        },
        "ml": {
            "problem_type": "regression",
            "best_model": "RandomForestRegressor",
            "rmse": 124.5,
        },
        "llm_tokens": 850,
    }

    metrics_file = tmp_path / "metrics.json"
    save_metrics_json(metrics_summary, metrics_file)
    assert metrics_file.is_file()

    loaded = load_metrics_json(metrics_file)
    assert loaded["dataset"] == "financials.csv"
    assert loaded["runtime_seconds"] == 6.1
    assert loaded["ml"]["best_model"] == "RandomForestRegressor"
    assert loaded["llm_tokens"] == 850
