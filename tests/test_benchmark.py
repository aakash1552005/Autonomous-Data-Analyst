"""
tests/test_benchmark.py
=======================
Comprehensive test suite for Phase 10 Benchmark & Evaluation Layer.
Verifies:
  1. Benchmark ground truth schema & dataset registry
  2. All 11 evaluators (Semantic, Domain, Date, PII, Cleaning, EDA, ML, Insight, Report, Runtime, Token)
  3. Baseline comparator handling (YDATA_PROFILING_UNAVAILABLE)
  4. 17 Adversarial Scenarios (fail safely without crashing evaluation)
  5. JSON schema validation for evaluation_results.json
  6. HTML report generation with zero raw PII leakage
  7. Benchmark cross-run reproducibility (identical metrics across runs)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from typing import Any
import pandas as pd
import pytest

from core.dio import DIO
from tests.benchmark.ground_truth import (
    BENCHMARK_REGISTRY,
    ColumnGroundTruth,
    CleaningGroundTruth,
    EDAGroundTruth,
    MLGroundTruth,
    InsightGroundTruth,
    DatasetGroundTruth,
    get_ground_truth,
)
from tests.benchmark.evaluators import (
    SemanticEvaluator,
    DomainEvaluator,
    DateEvaluator,
    PIIEvaluator,
    CleaningEvaluator,
    EDAEvaluator,
    MLEvaluator,
    InsightEvaluator,
    ReportEvaluator,
    RuntimeEvaluator,
    TokenEvaluator,
    BaselineComparator,
)
from tests.benchmark.benchmark_runner import BenchmarkRunner, BenchmarkSuiteResult, DatasetBenchmarkResult
from tests.benchmark.report_generator import ReportGenerator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample"
SCHEMA_FILE = PROJECT_ROOT / "tests" / "benchmark" / "expected" / "evaluation_schema.json"


# ==============================================================================
# 1. GROUND TRUTH REGISTRY & SCHEMA TESTS
# ==============================================================================

def test_ground_truth_registry_completeness():
    """Verify all 5 required benchmark datasets are registered and valid."""
    required_datasets = [
        "retail_sales",
        "healthcare_patients",
        "financial_loans",
        "mixed_messy_data",
        "ambiguous_dates_pii",
        "customer_churn_ml",
    ]
    for ds_name in required_datasets:
        assert ds_name in BENCHMARK_REGISTRY, f"Dataset '{ds_name}' not in registry"
        gt = get_ground_truth(ds_name)
        assert isinstance(gt, DatasetGroundTruth)
        assert (SAMPLE_DATA_DIR / gt.file_name).exists(), f"Sample file '{gt.file_name}' not found"
        assert len(gt.columns) > 0
        assert gt.expected_domain != ""
        assert 0.0 <= gt.min_domain_confidence <= 1.0


def test_ground_truth_unknown_dataset_error():
    """Verify requesting an unregistered dataset raises KeyError."""
    with pytest.raises(KeyError, match="No benchmark ground truth defined"):
        get_ground_truth("non_existent_dataset")


# ==============================================================================
# 2. INDIVIDUAL EVALUATOR UNIT TESTS
# ==============================================================================

def test_semantic_evaluator_perfect_and_error_scoring():
    """Test SemanticEvaluator scores correct matches and identifies errors and low confidence."""
    gt = get_ground_truth("retail_sales")
    evaluator = SemanticEvaluator()

    # Case A: Perfect match
    mock_dio = DIO.create_empty("test_hash")
    mock_dio["columns"] = [
        {"name": "order_id", "semantic_label": "identifier", "confidence": 0.90},
        {"name": "sku", "semantic_label": "identifier", "confidence": 0.90},
        {"name": "order_date", "semantic_label": "unknown", "confidence": 0.40},
        {"name": "customer_id", "semantic_label": "identifier", "confidence": 0.90},
        {"name": "price", "semantic_label": "currency_amount", "confidence": 0.90},
        {"name": "quantity", "semantic_label": "quantity", "confidence": 0.90},
        {"name": "discount_pct", "semantic_label": "percentage", "confidence": 0.90},
        {"name": "revenue", "semantic_label": "currency_amount", "confidence": 0.90},
        {"name": "status", "semantic_label": "category", "confidence": 0.90},
    ]

    res = evaluator.evaluate(mock_dio, gt)
    assert res.semantic_label_accuracy == 1.0
    assert res.semantic_label_correct == 9
    assert res.semantic_label_total == 9
    assert len(res.semantic_label_errors) == 0
    assert res.low_confidence_count == 1  # order_date confidence 0.40

    # Case B: Introduced errors
    mock_dio["columns"][0]["semantic_label"] = "health_metric"  # order_id wrong
    res_err = evaluator.evaluate(mock_dio, gt)
    assert res_err.semantic_label_accuracy == round(8 / 9, 4)
    assert len(res_err.semantic_label_errors) == 1
    assert res_err.semantic_label_errors[0]["column"] == "order_id"


def test_domain_evaluator_accuracy():
    """Test DomainEvaluator against expected and incorrect domains."""
    gt = get_ground_truth("retail_sales")
    evaluator = DomainEvaluator()

    # Exact match
    dio_correct = DIO.create_empty("hash")
    dio_correct["domain_guess"] = {"domain": "retail", "confidence": 0.95}
    res = evaluator.evaluate(dio_correct, gt)
    assert res.domain_accuracy == 1.0
    assert res.domain_correct is True

    # Mismatch
    dio_wrong = DIO.create_empty("hash")
    dio_wrong["domain_guess"] = {"domain": "healthcare", "confidence": 0.90}
    res_wrong = evaluator.evaluate(dio_wrong, gt)
    assert res_wrong.domain_accuracy == 0.0
    assert res_wrong.domain_correct is False


def test_date_evaluator_unambiguous_and_silent_guessing():
    """Test DateEvaluator detects unambiguous formats and catches silent date guesses."""
    evaluator = DateEvaluator()

    # Case 1: Unambiguous date (retail_sales)
    gt_retail = get_ground_truth("retail_sales")
    dio_retail = DIO.create_empty("hash")
    dio_retail["date_columns"] = [
        {"column": "order_date", "detected_format": "DD/MM/YYYY", "confidence": 1.0, "needs_user_confirmation": False}
    ]
    res_ret = evaluator.evaluate(dio_retail, gt_retail)
    assert res_ret.date_resolution_accuracy == 1.0
    assert res_ret.silent_date_guess_count == 0

    # Case 2: Ambiguous date correctly flagged (ambiguous_dates_pii)
    gt_ambig = get_ground_truth("ambiguous_dates_pii")
    dio_ambig_ok = DIO.create_empty("hash")
    dio_ambig_ok["date_columns"] = [
        {"column": "subscription_date", "detected_format": "ambiguous", "confidence": 0.50, "needs_user_confirmation": True}
    ]
    res_ambig_ok = evaluator.evaluate(dio_ambig_ok, gt_ambig)
    assert res_ambig_ok.date_resolution_accuracy == 1.0
    assert res_ambig_ok.ambiguous_dates_correctly_flagged == 1
    assert res_ambig_ok.silent_date_guess_count == 0

    # Case 3: Silent guess failure on ambiguous date!
    dio_ambig_silent = DIO.create_empty("hash")
    dio_ambig_silent["date_columns"] = [
        {"column": "subscription_date", "detected_format": "MM/DD/YYYY", "confidence": 1.0, "needs_user_confirmation": False}
    ]
    res_silent = evaluator.evaluate(dio_ambig_silent, gt_ambig)
    assert res_silent.date_resolution_accuracy == 0.0
    assert res_silent.silent_date_guess_count == 1
    assert len(res_silent.date_errors) == 1
    assert res_silent.date_errors[0]["silent_guess"] is True


def test_pii_evaluator_recall_precision_and_leakage():
    """Test PIIEvaluator computes TP, FP, FN, Recall, Precision, and flags raw leakage."""
    gt = get_ground_truth("ambiguous_dates_pii")
    evaluator = PIIEvaluator()

    # Case A: Perfect PII detection
    dio_pii = DIO.create_empty("hash")
    dio_pii["columns"] = [
        {"name": "user_id", "is_pii": False},
        {"name": "full_name", "is_pii": True},
        {"name": "email", "is_pii": True},
        {"name": "phone_number", "is_pii": True},
        {"name": "credit_card", "is_pii": True},
        {"name": "subscription_date", "is_pii": False},
        {"name": "amount", "is_pii": False},
    ]
    res = evaluator.evaluate(dio_pii, gt)
    assert res.pii_recall == 1.0
    assert res.pii_precision == 1.0
    assert res.pii_true_positive == 4
    assert res.pii_false_negative == 0
    assert res.pii_false_positive == 0
    assert res.raw_pii_leakage_count == 0

    # Case B: False Negative (missed credit_card)
    dio_pii["columns"][4]["is_pii"] = False
    res_fn = evaluator.evaluate(dio_pii, gt)
    assert res_fn.pii_true_positive == 3
    assert res_fn.pii_false_negative == 1
    assert res_fn.pii_recall == 0.75

    # Case C: Raw PII Leakage Detection
    dio_leaked = DIO.create_empty("hash")
    dio_leaked["decision_log"] = [{"message": "Found user johndoe@email.com"}]
    res_leak = evaluator.evaluate(dio_leaked, gt)
    assert res_leak.raw_pii_leakage_count == 1
    assert res_leak.leakage_detected is True


def test_cleaning_evaluator():
    """Test CleaningEvaluator verifies row counts and duplicate removal."""
    gt = get_ground_truth("mixed_messy_data")
    evaluator = CleaningEvaluator()

    dio = DIO.create_empty("hash")
    dio["ingestion"] = {"n_rows": 7, "n_columns": 5}
    dio["cleaning_log"] = [
        {"method": "drop_duplicates", "details": {"duplicate_count": 1}}
    ]

    cleaned_df = pd.DataFrame({
        "record_id": ["REC-01", "REC-02", "REC-03", "REC-04", "REC-05", "REC-06"],
        "metric_score": [88.5, 60.0, 92.0, 45.0, 74.2, 61.0],
    })

    res = evaluator.evaluate(dio, gt, cleaned_df=cleaned_df)
    assert res.cleaning_accuracy == 1.0
    assert res.duplicates_removed == 1
    assert res.row_count_before == 7
    assert res.row_count_after == 6
    assert res.cleaning_passed is True


def test_eda_evaluator_tolerances_and_charts(tmp_path: Path):
    """Test EDAEvaluator verifies statistics within tolerance and checks chart file sizes."""
    gt = get_ground_truth("retail_sales")
    evaluator = EDAEvaluator()

    chart_file = tmp_path / "revenue_dist.png"
    chart_file.write_bytes(b"X" * 1200)  # > 500 bytes

    dio = DIO.create_empty("hash")
    dio["eda"] = {
        "summary_stats": {
            "price": {"min": 8.75, "max": 120.0},
            "quantity": {"min": 1.0, "max": 5.0},
            "revenue": {"min": 15.50, "max": 408.0},
        },
        "chart_paths": [str(chart_file)],
    }

    res = evaluator.evaluate(dio, gt, run_dir=tmp_path)
    assert res.stats_verified_count == 6
    assert res.charts_verified is True
    assert res.eda_accuracy == 1.0


def test_ml_evaluator_applicable_and_skipped():
    """Test MLEvaluator handles both ML applicable (customer_churn_ml) and ML skipped (<30 rows)."""
    evaluator = MLEvaluator()

    # Case 1: ML Applicable (customer_churn_ml: 60 rows > 30)
    gt_churn = get_ground_truth("customer_churn_ml")
    dio_churn = DIO.create_empty("hash")
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        f.write(b"model_binary" * 50)
        model_path = f.name

    dio_churn["ml"] = {
        "status": "trained",
        "task_type": "classification",
        "target_column": "churn",
        "metrics": {"accuracy": 0.85, "f1": 0.82},
    }
    dio_churn["artifacts"] = {"model_pkl": model_path}

    res_ml = evaluator.evaluate(dio_churn, gt_churn)
    assert res_ml.ml_accuracy == 1.0
    assert res_ml.ml_ran is True
    assert res_ml.target_matched is True
    assert res_ml.task_type_matched is True

    # Case 2: ML Safely Skipped (ambiguous_dates_pii: 4 rows < 5)
    gt_ambig = get_ground_truth("ambiguous_dates_pii")
    dio_ambig = DIO.create_empty("hash")
    dio_ambig["ml"] = {
        "status": "skipped",
        "skip_reason": "Insufficient dataset size (4 rows < 5 minimum required)",
    }
    res_skip = evaluator.evaluate(dio_ambig, gt_ambig)
    assert res_skip.ml_accuracy == 1.0
    assert res_skip.ml_ran is False
    assert res_skip.ml_applicable_expected is False


def test_insight_evaluator_grounding_and_hallucination():
    """Test InsightEvaluator checks factual grounding and catches fabricated numbers."""
    gt = get_ground_truth("retail_sales")
    evaluator = InsightEvaluator()

    dio = DIO.create_empty("hash")
    dio["ingestion"] = {"n_rows": 6, "n_columns": 9}
    dio["eda"] = {
        "summary_stats": {
            "revenue": {"max": 408.0, "min": 15.50},
        }
    }

    # Case 1: Grounded insight
    dio["insights"] = [
        {
            "text": "Dataset has 6 records across 9 columns with peak revenue at 408.0.",
            "evidence": [{"metric": "Revenue", "value": 408.0}],
        }
    ]
    res_grounded = evaluator.evaluate(dio, gt)
    assert res_grounded.insight_grounding_accuracy == 1.0
    assert res_grounded.fabricated_numeric_claim_count == 0

    # Case 2: Hallucinated / fabricated number
    dio["insights"] = [
        {
            "text": "Average revenue was unexpectedly high at 999999.99 dollars.",
            "evidence": [],
        }
    ]
    res_hallucinated = evaluator.evaluate(dio, gt)
    assert res_hallucinated.fabricated_numeric_claim_count == 1
    assert res_hallucinated.insight_grounding_accuracy == 0.0


def test_baseline_comparator_unavailable():
    """Test BaselineComparator handles ydata-profiling cleanly without raising error."""
    gt = get_ground_truth("retail_sales")
    comparator = BaselineComparator()
    df = pd.DataFrame({"a": [1, 2, 3]})
    dio = DIO.create_empty("hash")
    dio["ingestion"] = {"n_rows": 3, "n_columns": 1}

    res = comparator.evaluate(df, dio, gt)
    assert res.status in ("AVAILABLE", "YDATA_PROFILING_UNAVAILABLE")
    assert "ydata-profiling" in res.framework
    assert isinstance(res.limitation_note, str)


def test_runtime_and_token_evaluators():
    """Test RuntimeEvaluator and TokenEvaluator produce structured metrics."""
    run_eval = RuntimeEvaluator()
    tok_eval = TokenEvaluator()

    dio = DIO.create_empty("hash")
    dio["agent_metrics"] = [
        {"agent": "intelligence", "runtime_seconds": 0.02},
        {"agent": "cleaning", "runtime_seconds": 0.03},
    ]
    dio["llm_usage"] = {
        "prompt_tokens": 150,
        "completion_tokens": 50,
        "total_tokens": 200,
        "calls_count": 2,
    }

    res_run = run_eval.evaluate(dio)
    assert res_run.total_runtime_seconds > 0
    assert res_run.intelligence_runtime == 0.02

    res_tok = tok_eval.evaluate(dio)
    assert res_tok.status == "actual"
    assert res_tok.total_tokens == 200


# ==============================================================================
# 3. 17 ADVERSARIAL SCENARIO TESTS (Section 20)
# ==============================================================================

def test_adversarial_01_incorrect_semantic_label():
    """Adversarial 1: Evaluator correctly scores down incorrect semantic labels."""
    gt = get_ground_truth("retail_sales")
    dio = DIO.create_empty("hash")
    dio["columns"] = [
        {"name": "price", "semantic_label": "person_name", "confidence": 0.90},
    ]
    res = SemanticEvaluator().evaluate(dio, gt)
    assert res.semantic_label_accuracy < 1.0
    assert any(e["column"] == "price" for e in res.semantic_label_errors)


def test_adversarial_02_incorrect_domain():
    """Adversarial 2: Evaluator reports domain mismatch when domain is wrong."""
    gt = get_ground_truth("healthcare_patients")
    dio = DIO.create_empty("hash")
    dio["domain_guess"] = {"domain": "gaming", "confidence": 0.99}
    res = DomainEvaluator().evaluate(dio, gt)
    assert res.domain_correct is False
    assert res.domain_accuracy == 0.0


def test_adversarial_03_ambiguous_date_preservation():
    """Adversarial 3: Ambiguous dates must be flagged without silent guessing."""
    gt = get_ground_truth("ambiguous_dates_pii")
    dio = DIO.create_empty("hash")
    dio["date_columns"] = [
        {"column": "subscription_date", "detected_format": "ambiguous", "confidence": 0.5, "needs_user_confirmation": True}
    ]
    res = DateEvaluator().evaluate(dio, gt)
    assert res.ambiguous_dates_correctly_flagged == 1
    assert res.silent_date_guess_count == 0


def test_adversarial_04_invalid_date():
    """Adversarial 4: Invalid date strings do not cause evaluator crash."""
    gt = get_ground_truth("retail_sales")
    dio = DIO.create_empty("hash")
    dio["date_columns"] = [
        {"column": "order_date", "detected_format": "INVALID_CORRUPT_FORMAT", "confidence": 0.0}
    ]
    res = DateEvaluator().evaluate(dio, gt)
    assert res.date_resolution_accuracy == 0.0
    assert len(res.date_errors) > 0


def test_adversarial_05_synthetic_pii_detection():
    """Adversarial 5: Synthetic PII must be detected with 100% recall on ambiguous_dates_pii."""
    gt = get_ground_truth("ambiguous_dates_pii")
    dio = DIO.create_empty("hash")
    dio["columns"] = [
        {"name": "full_name", "is_pii": True},
        {"name": "email", "is_pii": True},
        {"name": "phone_number", "is_pii": True},
        {"name": "credit_card", "is_pii": True},
        {"name": "amount", "is_pii": False},
    ]
    res = PIIEvaluator().evaluate(dio, gt)
    assert res.pii_recall == 1.0
    assert res.pii_false_negative == 0


def test_adversarial_06_identifier_columns():
    """Adversarial 6: Identifier columns should be correctly mapped as identifiers."""
    gt = get_ground_truth("retail_sales")
    dio = DIO.create_empty("hash")
    dio["columns"] = [
        {"name": "order_id", "semantic_label": "identifier", "confidence": 0.9},
        {"name": "customer_id", "semantic_label": "identifier", "confidence": 0.9},
    ]
    res = SemanticEvaluator().evaluate(dio, gt)
    matched = [e for e in res.semantic_label_errors if e["column"] in ("order_id", "customer_id")]
    assert len(matched) == 0  # Both should be recognized as correct


def test_adversarial_07_duplicate_rows():
    """Adversarial 7: Duplicate removal on mixed_messy_data must be detected and scored."""
    gt = get_ground_truth("mixed_messy_data")
    dio = DIO.create_empty("hash")
    dio["ingestion"] = {"n_rows": 7}
    dio["cleaning_log"] = [{"method": "drop_duplicates", "details": {"duplicate_count": 1}}]
    cleaned_df = pd.DataFrame({"record_id": ["R1", "R2", "R3", "R4", "R5", "R6"]})

    res = CleaningEvaluator().evaluate(dio, gt, cleaned_df=cleaned_df)
    assert res.duplicates_removed == 1
    assert res.row_count_after == 6


def test_adversarial_08_all_missing_column():
    """Adversarial 8: Column with 100% missing values is handled gracefully."""
    gt = get_ground_truth("mixed_messy_data")
    dio = DIO.create_empty("hash")
    dio["columns"] = [{"name": "all_null", "null_pct": 1.0, "semantic_label": "unknown", "confidence": 0.4}]
    res = SemanticEvaluator().evaluate(dio, gt)
    assert isinstance(res.semantic_label_accuracy, float)


def test_adversarial_09_zero_variance_column():
    """Adversarial 9: Zero-variance column handled in EDA without divide-by-zero."""
    gt = get_ground_truth("retail_sales")
    dio = DIO.create_empty("hash")
    dio["eda"] = {
        "summary_stats": {
            "constant_col": {"min": 5.0, "max": 5.0, "std": 0.0},
        },
        "chart_paths": [],
    }
    res = EDAEvaluator().evaluate(dio, gt)
    assert isinstance(res.eda_accuracy, float)


def test_adversarial_10_insufficient_ml_data():
    """Adversarial 10: ML stage skipped safely when data < 5 rows."""
    gt = get_ground_truth("ambiguous_dates_pii")  # 4 rows
    dio = DIO.create_empty("hash")
    dio["ml"] = {"status": "skipped", "skip_reason": "Too few rows"}
    res = MLEvaluator().evaluate(dio, gt)
    assert res.ml_accuracy == 1.0
    assert res.ml_ran is False


def test_adversarial_11_ml_leakage_protection():
    """Adversarial 11: PII columns are excluded from ML features."""
    gt = get_ground_truth("ambiguous_dates_pii")
    dio = DIO.create_empty("hash")
    dio["ml"] = {
        "status": "completed",
        "target_column": "amount",
        "features": ["full_name", "credit_card"],  # Leakage!
    }
    # ML should not run on ambiguous_dates_pii per ground truth
    res = MLEvaluator().evaluate(dio, gt)
    assert res.ml_accuracy == 0.0


def test_adversarial_12_unsupported_target():
    """Adversarial 12: When ML chooses wrong target, accuracy reflects failure."""
    gt = get_ground_truth("customer_churn_ml")  # Expected churn
    dio = DIO.create_empty("hash")
    dio["ml"] = {
        "status": "trained",
        "target_column": "wrong_column",
        "task_type": "classification",
        "metrics": {"accuracy": 0.5},
    }
    res = MLEvaluator().evaluate(dio, gt)
    assert res.target_matched is False
    assert res.ml_accuracy < 1.0


def test_adversarial_13_insight_fabricated_number():
    """Adversarial 13: Grounding evaluator detects fabricated numbers in insights."""
    gt = get_ground_truth("retail_sales")
    dio = DIO.create_empty("hash")
    dio["ingestion"] = {"n_rows": 6, "n_columns": 9}
    dio["insights"] = [
        {"text": "Profits surged by 88888.88 percent this quarter.", "evidence": []}
    ]
    res = InsightEvaluator().evaluate(dio, gt)
    assert res.fabricated_numeric_claim_count >= 1
    assert res.insight_grounding_accuracy == 0.0


def test_adversarial_14_broken_report_artifact(tmp_path: Path):
    """Adversarial 14: Report evaluator detects missing/corrupt deliverable files."""
    gt = get_ground_truth("retail_sales")
    dio = DIO.create_empty("hash")
    dio["artifacts"] = {
        "pdf_report": str(tmp_path / "non_existent.pdf"),
        "pptx_report": str(tmp_path / "non_existent.pptx"),
    }
    res = ReportEvaluator().evaluate(dio, gt, run_dir=tmp_path)
    assert res.pdf_generated is False
    assert res.pptx_generated is False
    assert res.report_fidelity_accuracy == 0.0


def test_adversarial_15_missing_optional_dependency():
    """Adversarial 15: Missing optional ydata-profiling does not crash evaluation."""
    gt = get_ground_truth("retail_sales")
    comparator = BaselineComparator()
    df = pd.DataFrame({"x": [1, 2]})
    dio = DIO.create_empty("hash")
    res = comparator.evaluate(df, dio, gt)
    assert res.status in ("AVAILABLE", "YDATA_PROFILING_UNAVAILABLE")


def test_adversarial_16_unavailable_token_metrics():
    """Adversarial 16: When tokens are not available, report 'unavailable', never fabricate."""
    dio = DIO.create_empty("hash")
    dio["llm_usage"] = {}  # Empty
    res = TokenEvaluator().evaluate(dio)
    assert res.status == "unavailable"
    assert res.total_tokens is None


def test_adversarial_17_empty_or_corrupted_dio():
    """Adversarial 17: Evaluators handle an empty/corrupted DIO without crashing."""
    empty_dio = DIO.create_empty("empty_hash")
    gt = get_ground_truth("retail_sales")

    sem_res = SemanticEvaluator().evaluate(empty_dio, gt)
    assert sem_res.semantic_label_accuracy == 0.0

    dom_res = DomainEvaluator().evaluate(empty_dio, gt)
    assert dom_res.domain_accuracy == 0.0

    date_res = DateEvaluator().evaluate(empty_dio, gt)
    assert date_res.date_resolution_accuracy == 0.0


# ==============================================================================
# 4. JSON SCHEMA & HTML GENERATION TESTS
# ==============================================================================

def test_evaluation_json_schema_validation(tmp_path: Path):
    """Verify evaluation_results.json validates against the official JSON schema."""
    schema_text = SCHEMA_FILE.read_text(encoding="utf-8")
    schema_dict = json.loads(schema_text)
    assert schema_dict["title"] == "AutonomousDataAnalystEvaluationResults"

    # Create dummy suite result
    suite_res = BenchmarkSuiteResult(
        benchmark_version="1.0",
        run_timestamp="2026-09-05T12:00:00Z",
        datasets=[
            DatasetBenchmarkResult(
                dataset="retail_sales",
                domain="retail",
                rows=6,
                columns=9,
                pipeline_status="completed",
                semantic_label_accuracy=1.0,
                domain_accuracy=1.0,
                date_resolution_accuracy=1.0,
                pii_recall=1.0,
                pii_metrics={"raw_pii_leakage_count": 0, "leakage_detected": False},
                cleaning_metrics={"cleaning_accuracy": 1.0},
                eda_metrics={"eda_accuracy": 1.0},
                ml_metrics={"ml_accuracy": 1.0},
                insight_metrics={"insight_grounding_accuracy": 1.0},
                report_metrics={"report_fidelity_accuracy": 1.0},
                runtime={"total_runtime_seconds": 1.25},
                token_usage={"status": "unavailable"},
                baseline_comparison={"status": "YDATA_PROFILING_UNAVAILABLE"},
                errors=[],
            )
        ],
        aggregate_metrics={
            "total_datasets": 1,
            "successful_datasets": 1,
            "mean_semantic_accuracy": 1.0,
            "mean_domain_accuracy": 1.0,
            "mean_date_accuracy": 1.0,
            "mean_pii_recall": 1.0,
            "mean_cleaning_accuracy": 1.0,
            "mean_eda_accuracy": 1.0,
            "mean_ml_accuracy": 1.0,
            "mean_insight_accuracy": 1.0,
            "mean_report_fidelity": 1.0,
            "total_runtime_seconds": 1.25,
            "total_pii_leakages": 0,
        },
        limitations=["Testing limitation note"],
    )

    gen = ReportGenerator(output_dir=tmp_path)
    json_path = gen.generate_json(suite_res)
    assert json_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    for req_key in schema_dict["required"]:
        assert req_key in data, f"Key '{req_key}' missing from generated JSON"


def test_evaluation_html_generation_and_zero_pii(tmp_path: Path):
    """Verify evaluation.html generation produces valid HTML with absolute zero PII leakage."""
    suite_res = BenchmarkSuiteResult(
        benchmark_version="1.0",
        run_timestamp="2026-09-05T12:00:00Z",
        datasets=[
            DatasetBenchmarkResult(
                dataset="ambiguous_dates_pii",
                domain="generic",
                rows=4,
                columns=7,
                pipeline_status="completed",
                semantic_label_accuracy=0.85,
                domain_accuracy=1.0,
                date_resolution_accuracy=1.0,
                pii_recall=1.0,
                pii_metrics={"raw_pii_leakage_count": 0, "leakage_detected": False},
                cleaning_metrics={"cleaning_accuracy": 1.0},
                eda_metrics={"eda_accuracy": 1.0},
                ml_metrics={"ml_accuracy": 1.0, "ml_ran": False},
                insight_metrics={"insight_grounding_accuracy": 1.0},
                report_metrics={"report_fidelity_accuracy": 1.0},
                runtime={"total_runtime_seconds": 0.85},
                token_usage={"status": "unavailable"},
                baseline_comparison={"status": "YDATA_PROFILING_UNAVAILABLE"},
                errors=[],
            )
        ],
        aggregate_metrics={
            "total_datasets": 1,
            "successful_datasets": 1,
            "mean_semantic_accuracy": 0.85,
            "mean_domain_accuracy": 1.0,
            "mean_date_accuracy": 1.0,
            "mean_pii_recall": 1.0,
            "mean_cleaning_accuracy": 1.0,
            "mean_eda_accuracy": 1.0,
            "mean_ml_accuracy": 1.0,
            "mean_insight_accuracy": 1.0,
            "mean_report_fidelity": 1.0,
            "total_runtime_seconds": 0.85,
            "total_pii_leakages": 0,
        },
        limitations=["No PII in HTML"],
    )

    gen = ReportGenerator(output_dir=tmp_path)
    html_path = gen.generate_html(suite_res)
    assert html_path.exists()

    html_text = html_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text
    assert "Autonomous Data Analyst — Benchmark & Evaluation Report" in html_text
    assert "YDATA_PROFILING_UNAVAILABLE" in html_text

    # Zero PII Check in HTML
    gt_pii = get_ground_truth("ambiguous_dates_pii")
    for sensitive_val in gt_pii.sensitive_raw_values:
        assert sensitive_val not in html_text, f"Raw PII '{sensitive_val}' leaked into evaluation.html!"


# ==============================================================================
# 5. BENCHMARK REPRODUCIBILITY TEST (Section 21)
# ==============================================================================

def test_benchmark_cross_run_reproducibility():
    """
    Run evaluation twice under identical configuration.
    Assert deterministic metrics are strictly identical across both runs.
    """
    evaluator_sem = SemanticEvaluator()
    evaluator_dom = DomainEvaluator()
    evaluator_date = DateEvaluator()
    evaluator_pii = PIIEvaluator()

    gt = get_ground_truth("retail_sales")

    mock_dio = DIO.create_empty("deterministic_hash")
    mock_dio["columns"] = [
        {"name": "order_id", "semantic_label": "identifier", "confidence": 0.90},
        {"name": "price", "semantic_label": "currency_amount", "confidence": 0.90},
    ]
    mock_dio["domain_guess"] = {"domain": "retail", "confidence": 0.95}
    mock_dio["date_columns"] = [
        {"column": "order_date", "detected_format": "DD/MM/YYYY", "confidence": 1.0, "needs_user_confirmation": False}
    ]

    # Run 1
    sem1 = evaluator_sem.evaluate(mock_dio, gt)
    dom1 = evaluator_dom.evaluate(mock_dio, gt)
    date1 = evaluator_date.evaluate(mock_dio, gt)
    pii1 = evaluator_pii.evaluate(mock_dio, gt)

    # Run 2
    sem2 = evaluator_sem.evaluate(mock_dio, gt)
    dom2 = evaluator_dom.evaluate(mock_dio, gt)
    date2 = evaluator_date.evaluate(mock_dio, gt)
    pii2 = evaluator_pii.evaluate(mock_dio, gt)

    # Assert 100% equivalence on deterministic metrics
    assert sem1.semantic_label_accuracy == sem2.semantic_label_accuracy
    assert sem1.semantic_label_correct == sem2.semantic_label_correct
    assert dom1.domain_accuracy == dom2.domain_accuracy
    assert dom1.domain_confidence == dom2.domain_confidence
    assert date1.date_resolution_accuracy == date2.date_resolution_accuracy
    assert pii1.pii_recall == pii2.pii_recall
    assert pii1.pii_precision == pii2.pii_precision


# ==============================================================================
# 6. PHASE 10.1 REMEDIATION REGRESSION TESTS
# ==============================================================================

def test_remediation_pii_aggregation_test_a_leak_detected():
    """Test A: A dataset with one detected raw PII leak produces leakage_count=1 and suite reports total=1."""
    res = DatasetBenchmarkResult(
        dataset="test_leak_ds",
        domain="generic",
        rows=10,
        columns=5,
        pipeline_status="completed",
        semantic_label_accuracy=1.0,
        domain_accuracy=1.0,
        date_resolution_accuracy=1.0,
        pii_recall=1.0,
        pii_metrics={"raw_pii_leakage_count": 1, "leakage_detected": True},
        cleaning_metrics={"cleaning_accuracy": 1.0},
        eda_metrics={"eda_accuracy": 1.0},
        ml_metrics={"ml_behavior_compliance_score": 1.0},
        insight_metrics={"insight_grounding_accuracy": 1.0},
        report_metrics={"report_fidelity_accuracy": 1.0},
        runtime={"total_runtime_seconds": 1.0},
        token_usage={},
        baseline_comparison={},
    )
    assert res.pii_metrics_leakage() == 1

    # In suite aggregation:
    total_leak = sum(r.pii_metrics_leakage() for r in [res])
    assert total_leak == 1


def test_remediation_pii_aggregation_test_b_zero_leak():
    """Test B: A dataset with zero leaks produces 0."""
    res = DatasetBenchmarkResult(
        dataset="test_clean_ds",
        domain="generic",
        rows=10,
        columns=5,
        pipeline_status="completed",
        semantic_label_accuracy=1.0,
        domain_accuracy=1.0,
        date_resolution_accuracy=1.0,
        pii_recall=1.0,
        pii_metrics={"raw_pii_leakage_count": 0, "leakage_detected": False},
        cleaning_metrics={"cleaning_accuracy": 1.0},
        eda_metrics={"eda_accuracy": 1.0},
        ml_metrics={"ml_behavior_compliance_score": 1.0},
        insight_metrics={"insight_grounding_accuracy": 1.0},
        report_metrics={"report_fidelity_accuracy": 1.0},
        runtime={"total_runtime_seconds": 1.0},
        token_usage={},
        baseline_comparison={},
    )
    assert res.pii_metrics_leakage() == 0


def test_remediation_pii_aggregation_test_c_mixed_suite():
    """Test C: Mixed suite with leaking and non-leaking datasets aggregates correctly."""
    r1 = DatasetBenchmarkResult(
        dataset="clean",
        domain="generic",
        rows=10,
        columns=5,
        pipeline_status="completed",
        semantic_label_accuracy=1.0,
        domain_accuracy=1.0,
        date_resolution_accuracy=1.0,
        pii_recall=1.0,
        pii_metrics={"raw_pii_leakage_count": 0},
        cleaning_metrics={},
        eda_metrics={},
        ml_metrics={},
        insight_metrics={},
        report_metrics={},
        runtime={"total_runtime_seconds": 1.0},
        token_usage={},
        baseline_comparison={},
    )
    r2 = DatasetBenchmarkResult(
        dataset="leaking",
        domain="generic",
        rows=10,
        columns=5,
        pipeline_status="completed",
        semantic_label_accuracy=1.0,
        domain_accuracy=1.0,
        date_resolution_accuracy=1.0,
        pii_recall=1.0,
        pii_metrics={"raw_pii_leakage_count": 3},
        cleaning_metrics={},
        eda_metrics={},
        ml_metrics={},
        insight_metrics={},
        report_metrics={},
        runtime={"total_runtime_seconds": 1.0},
        token_usage={},
        baseline_comparison={},
    )
    assert sum(r.pii_metrics_leakage() for r in [r1, r2]) == 3


def test_remediation_pii_aggregation_test_d_missing_metric_fails_loudly():
    """Test D: A missing metric cannot silently become a false PASS; must raise ValueError."""
    r_missing = DatasetBenchmarkResult(
        dataset="missing_metrics",
        domain="generic",
        rows=10,
        columns=5,
        pipeline_status="completed",
        semantic_label_accuracy=1.0,
        domain_accuracy=1.0,
        date_resolution_accuracy=1.0,
        pii_recall=1.0,
        pii_metrics={},  # Missing raw_pii_leakage_count!
        cleaning_metrics={},
        eda_metrics={},
        ml_metrics={},
        insight_metrics={},
        report_metrics={},
        runtime={"total_runtime_seconds": 1.0},
        token_usage={},
        baseline_comparison={},
    )
    with pytest.raises(ValueError, match="missing"):
        r_missing.pii_metrics_leakage()


def test_remediation_insight_grounding_tests_1_to_6():
    """Test InsightEvaluator strictly penalizes fabricated numbers across all 6 required cases."""
    gt = get_ground_truth("retail_sales")
    evaluator = InsightEvaluator()

    dio_base = DIO.create_empty("hash")
    dio_base["ingestion"] = {"n_rows": 6, "n_columns": 9}
    dio_base["eda"] = {"summary_stats": {"revenue": {"max": 408.0, "min": 15.50}}}

    # Case 1: All claims supported -> 100%
    dio1 = DIO.create_empty("h1")
    dio1["ingestion"] = {"n_rows": 6, "n_columns": 9}
    dio1["insights"] = [
        {"text": "Total records are 6.", "evidence": [{"metric": "Rows", "value": 6}]},
        {"text": "Total columns are 9.", "evidence": [{"metric": "Cols", "value": 9}]},
    ]
    res1 = evaluator.evaluate(dio1, gt)
    assert res1.insight_grounding_accuracy == 1.0
    assert res1.grounding_accuracy == 1.0
    assert res1.supported_claim_count == 2
    assert res1.unsupported_claim_count == 0
    assert res1.fabricated_numeric_claim_count == 0

    # Case 2: Unsupported claim -> score decreases
    dio2 = DIO.create_empty("h2")
    dio2["ingestion"] = {"n_rows": 6, "n_columns": 9}
    dio2["insights"] = [
        {"text": "Total records are 6.", "evidence": [{"metric": "Rows", "value": 6}]},
        {"text": "No evidence provided for this statement.", "evidence": []},
    ]
    res2 = evaluator.evaluate(dio2, gt)
    assert res2.insight_grounding_accuracy < 1.0
    assert res2.supported_claim_count == 1
    assert res2.unsupported_claim_count == 1

    # Case 3: Fabricated numeric claim -> score decreases
    dio3 = DIO.create_empty("h3")
    dio3["ingestion"] = {"n_rows": 6, "n_columns": 9}
    dio3["insights"] = [
        {"text": "Total records are 6.", "evidence": [{"metric": "Rows", "value": 6}]},
        {"text": "Revenue reached 777777.77 dollars.", "evidence": []},
    ]
    res3 = evaluator.evaluate(dio3, gt)
    assert res3.insight_grounding_accuracy < 1.0
    assert res3.fabricated_numeric_claim_count == 1

    # Case 4: Supported + fabricated claim -> CANNOT report 100%
    dio4 = DIO.create_empty("h4")
    dio4["ingestion"] = {"n_rows": 6, "n_columns": 9}
    dio4["insights"] = [
        {"text": "Records are 6 with fake number 999999.0.", "evidence": [{"metric": "Rows", "value": 6}]},
    ]
    res4 = evaluator.evaluate(dio4, gt)
    assert res4.insight_grounding_accuracy < 1.0, "Insight with fabricated numbers must not report 100%!"
    assert res4.fabricated_numeric_claim_count == 1

    # Case 5: Multiple fabricated numbers -> all are accounted for
    dio5 = DIO.create_empty("h5")
    dio5["ingestion"] = {"n_rows": 6, "n_columns": 9}
    dio5["insights"] = [
        {"text": "Fabricated 8888.0 and 9999.0 and 55555.0 numbers.", "evidence": []},
    ]
    res5 = evaluator.evaluate(dio5, gt)
    assert res5.fabricated_numeric_claim_count == 3
    assert res5.insight_grounding_accuracy < 0.5

    # Case 6: Zero evaluated claims -> explicitly defined behavior
    dio6 = DIO.create_empty("h6")
    dio6["insights"] = []
    res6 = evaluator.evaluate(dio6, gt)
    assert res6.insight_grounding_accuracy == 1.0
    assert res6.total_insights_evaluated == 0
    assert res6.supported_claim_count == 0
    assert res6.unsupported_claim_count == 0
    assert res6.fabricated_numeric_claim_count == 0


def test_remediation_ml_semantics_compliance_and_status():
    """Test ML evaluation semantics: ml_behavior_compliance_score, SKIPPED_INSUFFICIENT_DATA vs TRAINED."""
    evaluator = MLEvaluator()

    # Case 1: Skipped model (< 30 rows)
    gt_skip = get_ground_truth("retail_sales")
    dio_skip = DIO.create_empty("skip")
    dio_skip["ml"] = {"status": "skipped", "skip_reason": "6 rows < 30"}
    res_skip = evaluator.evaluate(dio_skip, gt_skip)

    assert res_skip.ml_behavior_compliance_score == 1.0
    assert res_skip.model_training_status == "SKIPPED_INSUFFICIENT_DATA"
    assert res_skip.metrics_present is False
    assert res_skip.model_persisted is False
    assert res_skip.predictive_metrics is None

    # Case 2: Trained model (customer_churn_ml)
    gt_churn = get_ground_truth("customer_churn_ml")
    dio_churn = DIO.create_empty("churn")
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        f.write(b"model_binary" * 50)
        m_path = f.name

    dio_churn["ml"] = {
        "status": "trained",
        "task_type": "classification",
        "target_column": "churn",
        "selected_model": "random_forest",
        "metrics": {"accuracy": 0.85, "f1": 0.82, "roc_auc": 0.88},
        "improved_over_baseline": True,
        "selection_reason": "Best F1",
    }
    dio_churn["artifacts"] = {"model_pkl": m_path}

    res_trained = evaluator.evaluate(dio_churn, gt_churn)
    assert res_trained.ml_behavior_compliance_score == 1.0
    assert res_trained.model_training_status == "TRAINED"
    assert res_trained.metrics_present is True
    assert res_trained.model_persisted is True
    assert res_trained.predictive_metrics is not None
    assert res_trained.predictive_metrics["f1"] == 0.82


def test_remediation_eda_pii_exclusion():
    """Test that EDA correlations exclude PII and identifier columns."""
    from agents.eda.correlations import compute_correlations

    df = pd.DataFrame({
        "user_id": ["U1", "U2", "U3", "U4"],
        "credit_card": [4532112233445566, 5412998877665544, 3782822463100051, 4024007188992233],
        "full_name": ["Alice", "Bob", "Charlie", "David"],
        "price": [10.0, 20.0, 30.0, 40.0],
        "quantity": [1.0, 2.0, 3.0, 4.0],
    })

    columns_info = [
        {"name": "user_id", "semantic_label": "identifier", "is_pii": False, "dtype_inferred": "string"},
        {"name": "credit_card", "semantic_label": "credit_card", "is_pii": True, "dtype_inferred": "int"},
        {"name": "full_name", "semantic_label": "person_name", "is_pii": True, "dtype_inferred": "string"},
        {"name": "price", "semantic_label": "currency_amount", "is_pii": False, "dtype_inferred": "float"},
        {"name": "quantity", "semantic_label": "quantity", "is_pii": False, "dtype_inferred": "float"},
    ]

    corr = compute_correlations(df, columns_info=columns_info)
    analyzed = corr["numeric_columns_analyzed"]

    assert "credit_card" not in analyzed, "PII column credit_card must be excluded from correlation analysis!"
    assert "user_id" not in analyzed, "Identifier column user_id must be excluded from correlation analysis!"
    assert "full_name" not in analyzed, "PII column full_name must be excluded from correlation analysis!"
    assert "price" in analyzed
    assert "quantity" in analyzed


def test_remediation_tier_b_datasets_registered_and_ml_execution():
    """Test Tier B realistic datasets exist (>= 1,000 rows) and train genuine ML models."""
    from tests.benchmark.ground_truth import TIER_B_REGISTRY

    assert len(TIER_B_REGISTRY) == 3
    assert "telco_churn_1k" in TIER_B_REGISTRY
    assert "housing_regression_1k" in TIER_B_REGISTRY
    assert "credit_default_1k" in TIER_B_REGISTRY

    sample_dir = Path("data/sample")
    for ds_name, gt in TIER_B_REGISTRY.items():
        csv_file = sample_dir / gt.file_name
        assert csv_file.exists(), f"Tier B dataset file {csv_file} must exist!"
        df = pd.read_csv(csv_file)
        assert len(df) >= 1000, f"Tier B dataset {ds_name} must have >= 1,000 rows, got {len(df)}"
        assert gt.ml.applicable is True

