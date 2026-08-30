"""
tests/test_dio.py
=================
Tests for the Dataset Intelligence Object (DIO).
Verifies creation, section mutation, JSON serialization/deserialization round-trip,
dictionary-like access, and schema validation.
"""

import json
import uuid
import pytest
from core.dio import DIO


def test_dio_creation():
    dataset_id = str(uuid.uuid4())
    dio = DIO.create_empty(file_name="sales.csv", dataset_hash="abc123hash", dataset_id=dataset_id)
    assert dio.file_name == "sales.csv"
    assert dio.dataset_hash == "abc123hash"
    assert dio.dataset_id == dataset_id
    assert dio.schema_version == "1.0.0"
    assert isinstance(dio.columns, list)
    assert isinstance(dio.progress, dict)
    assert isinstance(dio.errors, list)


def test_dio_dict_like_access():
    dio = DIO.create_empty(file_name="test.csv")
    dio["progress"]["intelligence"] = "FINISHED"
    assert dio["progress"]["intelligence"] == "FINISHED"
    assert "progress" in dio

    with pytest.raises(KeyError):
        _ = dio["non_existent_section"]


def test_dio_json_round_trip():
    """
    Mandatory test: DIO -> JSON -> DIO preserves every top-level section
    and nested metadata without loss.
    """
    dio = DIO.create_empty(file_name="customers.xlsx", dataset_hash="deadbeef1234")
    dio.columns.append({
        "name": "email",
        "dtype_raw": "object",
        "dtype_inferred": "string",
        "semantic_label": "email",
        "confidence": 0.95,
        "is_pii": True,
        "pii_type": "email",
        "null_pct": 0.02,
        "unique_count": 980,
    })
    dio.date_columns.append({
        "column": "created_at",
        "detected_format": "YYYY-MM-DD",
        "confidence": 1.0,
        "evidence": ["day > 12"],
        "needs_user_confirmation": False,
    })
    dio.domain_guess = {"domain": "retail", "confidence": 0.85}
    dio.quality = {"score": 92, "issues": ["2% missing emails"]}
    dio.cleaning_log.append({
        "column": "age",
        "action": "impute_median",
        "reason": "skew > 1",
        "replacement_value": 34,
        "reversible": True,
    })
    dio.eda = {
        "summary_stats": {"age": {"mean": 34.5, "median": 34.0}},
        "correlations": {"age_income": 0.45},
        "charts": ["runs/test_run/charts/histogram_age.png"],
    }
    dio.ml = {
        "problem_type": "classification",
        "target_column": "churn",
        "target_score": 5.0,
        "models_tried": [{"name": "RandomForest", "metric_name": "f1", "metric_value": 0.88}],
        "best_model": "RandomForest",
        "feature_importance": {"age": 0.35, "tenure": 0.65},
        "feature_explanations": ["Higher tenure correlates with lower churn"],
        "class_imbalance_warning": False,
    }
    dio.insights.append({
        "text": "Churn decreased by 12% in Q4.",
        "confidence": 0.9,
        "evidence": "eda.summary_stats",
    })
    dio.reports = {"pdf_path": "runs/test/report.pdf", "pptx_path": "runs/test/presentation.pptx"}
    dio.progress = {"intelligence": "FINISHED", "cleaning": "FINISHED", "eda": "RUNNING"}
    dio.errors.append({"agent": "ml", "code": "ML_001", "message": "Low variance target"})
    dio.agent_metrics.append({"agent": "intelligence", "runtime_seconds": 2.4, "warnings": 0, "errors": 0})
    dio.decision_log.append({
        "agent": "intelligence",
        "action": "date_resolved",
        "reason": "day > 12 found in row 3",
        "confidence": 1.0,
        "timestamp": "2026-08-30T12:00:00Z",
    })
    dio.llm_usage = {
        "total_tokens": 450,
        "total_calls": 2,
        "prompt_tokens": 300,
        "completion_tokens": 150,
        "calls": [{"call_index": 1, "model": "llama3.1:8b", "tokens": 225}],
    }

    # Serialize
    json_str = dio.to_json()
    assert isinstance(json_str, str)

    # Deserialize
    reconstructed = DIO.from_json(json_str)

    # Validate exact equality of all sections
    assert reconstructed.to_dict() == dio.to_dict()
    assert reconstructed.dataset_id == dio.dataset_id
    assert reconstructed.dataset_hash == "deadbeef1234"
    assert reconstructed.columns[0]["name"] == "email"
    assert reconstructed.ml["best_model"] == "RandomForest"
    assert reconstructed.progress["eda"] == "RUNNING"
    assert reconstructed.llm_usage["total_tokens"] == 450


def test_dio_validation():
    dio = DIO.create_empty(file_name="valid.csv")
    issues = dio.validate()
    assert len(issues) == 0

    dio.schema_version = ""
    dio.columns = "not-a-list"  # type: ignore[assignment]
    issues = dio.validate()
    assert len(issues) >= 2
