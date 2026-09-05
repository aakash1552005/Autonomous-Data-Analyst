"""
tests/benchmark/evaluators/pii_evaluator.py
===========================================
Evaluator for Metric 4: PII Recall, Precision & Zero-Leakage Audit.
Measures detector recall:
  PII RECALL = TP / (TP + FN)
  PII PRECISION = TP / (TP + FP)
Ensures raw PII NEVER enters evaluation logs, JSON, or HTML.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from core.dio import DIO
from tests.benchmark.ground_truth import DatasetGroundTruth


@dataclass
class PIIEvaluationResult:
    pii_true_positive: int
    pii_false_negative: int
    pii_false_positive: int
    pii_recall: float
    pii_precision: float
    raw_pii_leakage_count: int
    leakage_detected: bool
    pii_columns_detected: list[str] = field(default_factory=list)
    pii_columns_expected: list[str] = field(default_factory=list)
    pii_errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PIIEvaluator:
    """Evaluates PII detection metrics and verifies zero raw PII leakage."""

    def evaluate(
        self,
        dio: DIO,
        ground_truth: DatasetGroundTruth,
        run_dir: Path | None = None,
    ) -> PIIEvaluationResult:
        columns = dio.get("columns", [])
        dio_cols_by_name = {c.get("name", ""): c for c in columns}

        # Expected PII columns from ground truth
        expected_pii_cols = {
            col_name for col_name, c_gt in ground_truth.columns.items() if c_gt.is_pii
        }

        # Predicted PII columns from DIO
        predicted_pii_cols = {
            c.get("name", "") for c in columns if c.get("is_pii") is True
        }

        tp = len(expected_pii_cols & predicted_pii_cols)
        fn = len(expected_pii_cols - predicted_pii_cols)
        fp = len(predicted_pii_cols - expected_pii_cols)

        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 1.0
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0

        errors: list[dict[str, Any]] = []
        for missed_col in (expected_pii_cols - predicted_pii_cols):
            errors.append({
                "column": missed_col,
                "error": "False Negative: Expected PII column was not flagged is_pii=True",
            })

        for false_col in (predicted_pii_cols - expected_pii_cols):
            errors.append({
                "column": false_col,
                "error": "False Positive: Non-PII column was erroneously flagged is_pii=True",
            })

        # --- Zero PII Leakage Check ---
        leakage_count = 0
        if ground_truth.sensitive_raw_values:
            # Check DIO serialized text
            dio_str = json.dumps(dio.to_dict(), default=str)
            for raw_val in ground_truth.sensitive_raw_values:
                # Column names or masked samples may exist, but raw sensitive values must not appear
                # in decision logs, reports metadata, or prompt strings
                if raw_val in dio_str:
                    leakage_count += 1

            # Check log file if run_dir is provided
            if run_dir is not None:
                log_file = run_dir / "pipeline.log"
                if log_file.exists():
                    log_text = log_file.read_text(encoding="utf-8", errors="ignore")
                    for raw_val in ground_truth.sensitive_raw_values:
                        if raw_val in log_text:
                            leakage_count += 1

        return PIIEvaluationResult(
            pii_true_positive=tp,
            pii_false_negative=fn,
            pii_false_positive=fp,
            pii_recall=recall,
            pii_precision=precision,
            raw_pii_leakage_count=leakage_count,
            leakage_detected=(leakage_count > 0),
            pii_columns_detected=sorted(list(predicted_pii_cols)),
            pii_columns_expected=sorted(list(expected_pii_cols)),
            pii_errors=errors,
        )
