"""
tests/benchmark/evaluators/date_evaluator.py
============================================
Evaluator for Metric 3: Date-Resolution Accuracy.
Evaluates date column identification, format parsing, ambiguous handling, and silent guessing.
Produces:
  - date_resolution_accuracy
  - ambiguous_dates_correctly_flagged
  - silent_date_guess_count
  - date_columns_evaluated
  - date_errors
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.dio import DIO
from tests.benchmark.ground_truth import DatasetGroundTruth


@dataclass
class DateEvaluationResult:
    date_resolution_accuracy: float
    ambiguous_dates_correctly_flagged: int
    silent_date_guess_count: int
    date_columns_evaluated: int
    date_errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DateEvaluator:
    """Evaluates date resolution, ambiguous flag, and silent date guessing."""

    def evaluate(self, dio: DIO, ground_truth: DatasetGroundTruth) -> DateEvaluationResult:
        date_columns_dio = dio.get("date_columns", [])
        dio_dates_by_col = {d.get("column", ""): d for d in date_columns_dio}

        # Identify expected date columns from ground truth
        expected_date_cols = {
            c_name: c_gt for c_name, c_gt in ground_truth.columns.items() if c_gt.is_date
        }

        if not expected_date_cols:
            # No date columns expected in this dataset (e.g. financial_loans, mixed_messy)
            # Check if any false positives were erroneously detected
            fp_dates = [c for c in date_columns_dio if c.get("column") not in ground_truth.columns]
            return DateEvaluationResult(
                date_resolution_accuracy=1.0 if not fp_dates else 0.0,
                ambiguous_dates_correctly_flagged=0,
                silent_date_guess_count=0,
                date_columns_evaluated=0,
                date_errors=[{"column": d["column"], "error": "False positive date column"} for d in fp_dates],
            )

        correct_count = 0
        total_count = len(expected_date_cols)
        ambiguous_flagged_count = 0
        silent_guess_count = 0
        errors: list[dict[str, Any]] = []

        for col_name, c_gt in expected_date_cols.items():
            pred_date = dio_dates_by_col.get(col_name)

            if pred_date is None:
                errors.append({
                    "column": col_name,
                    "error": "Expected date column was not detected in dio['date_columns']",
                })
                continue

            detected_format = pred_date.get("detected_format", "")
            confidence = float(pred_date.get("confidence", 0.0))
            needs_confirmation = bool(pred_date.get("needs_user_confirmation", False))

            if c_gt.is_ambiguous_date:
                # Ambiguous date handling: must be flagged ambiguous with needs_user_confirmation=True
                # If the pipeline guessed a format silently without confirmation, it's a critical failure!
                if detected_format == "ambiguous" and needs_confirmation:
                    ambiguous_flagged_count += 1
                    correct_count += 1
                else:
                    silent_guess_count += 1
                    errors.append({
                        "column": col_name,
                        "error": f"Silent date guess detected! Predicted format '{detected_format}' with confirmation={needs_confirmation}",
                        "silent_guess": True,
                    })
            else:
                # Unambiguous date handling: check expected format and confidence
                if c_gt.expected_date_format:
                    # Normalize separators for comparison (e.g. YYYY-MM-DD vs YYYY/MM/DD)
                    norm_detected = detected_format.replace("/", "-").replace(".", "-").upper()
                    norm_expected = c_gt.expected_date_format.replace("/", "-").replace(".", "-").upper()
                    if norm_detected == norm_expected:
                        correct_count += 1
                    else:
                        errors.append({
                            "column": col_name,
                            "error": f"Format mismatch: detected '{detected_format}' vs expected '{c_gt.expected_date_format}'",
                        })
                else:
                    # Format unspecified, but recognized as date
                    correct_count += 1

        accuracy = round(correct_count / total_count, 4) if total_count > 0 else 1.0

        return DateEvaluationResult(
            date_resolution_accuracy=accuracy,
            ambiguous_dates_correctly_flagged=ambiguous_flagged_count,
            silent_date_guess_count=silent_guess_count,
            date_columns_evaluated=total_count,
            date_errors=errors,
        )
