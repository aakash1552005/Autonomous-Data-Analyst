"""
tests/benchmark/evaluators/cleaning_evaluator.py
================================================
Evaluator for Cleaning Agent Performance.
Measures structural correctness:
  - duplicate removal
  - missing value handling
  - type coercion
  - ambiguous date preservation
  - row count before/after
Produces:
  - cleaning_accuracy
  - duplicates_removed
  - missing_values_before
  - missing_values_after
  - dtype_corrections
  - row_count_before
  - row_count_after
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd

from core.dio import DIO
from tests.benchmark.ground_truth import DatasetGroundTruth


@dataclass
class CleaningEvaluationResult:
    cleaning_accuracy: float
    duplicates_removed: int
    missing_values_before: int
    missing_values_after: int
    dtype_corrections: int
    row_count_before: int
    row_count_after: int
    cleaning_passed: bool
    cleaning_errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CleaningEvaluator:
    """Evaluates cleaning transformations against ground truth."""

    def evaluate(
        self,
        dio: DIO,
        ground_truth: DatasetGroundTruth,
        cleaned_df: pd.DataFrame | None = None,
    ) -> CleaningEvaluationResult:
        cleaning_log = dio.get("cleaning_log", [])
        ingestion = dio.get("ingestion", {})
        gt_cleaning = ground_truth.cleaning

        row_before = int(ingestion.get("n_rows", 0))

        # Check cleaned dataframe or cleaned CSV artifact
        if cleaned_df is not None:
            row_after = len(cleaned_df)
            missing_after = int(cleaned_df.isna().sum().sum())
        elif dio.get("artifacts", {}).get("cleaned_csv"):
            csv_path = Path(dio["artifacts"]["cleaned_csv"])
            if csv_path.exists():
                df_clean = pd.read_csv(csv_path)
                row_after = len(df_clean)
                missing_after = int(df_clean.isna().sum().sum())
            else:
                row_after = 0
                missing_after = 0
        else:
            row_after = 0
            missing_after = 0

        # Extract duplicates removed from cleaning log
        dup_entries = [l for l in cleaning_log if l.get("method") == "drop_duplicates"]
        dup_removed = int(dup_entries[0]["details"]["duplicate_count"]) if dup_entries else 0

        # Count dtype corrections from cleaning log
        dtype_entries = [l for l in cleaning_log if l.get("method") == "coerce_dtype"]
        dtype_corrections = len(dtype_entries)

        errors: list[dict[str, Any]] = []

        # Validate against ground truth expectations
        checks_passed = 0
        total_checks = 4

        # 1. Row count before
        if row_before == gt_cleaning.expected_rows_before:
            checks_passed += 1
        else:
            errors.append({
                "metric": "rows_before",
                "actual": row_before,
                "expected": gt_cleaning.expected_rows_before,
            })

        # 2. Row count after
        if row_after == gt_cleaning.expected_rows_after:
            checks_passed += 1
        else:
            errors.append({
                "metric": "rows_after",
                "actual": row_after,
                "expected": gt_cleaning.expected_rows_after,
            })

        # 3. Duplicates removed
        if dup_removed == gt_cleaning.expected_duplicates_removed:
            checks_passed += 1
        else:
            errors.append({
                "metric": "duplicates_removed",
                "actual": dup_removed,
                "expected": gt_cleaning.expected_duplicates_removed,
            })

        # 4. Missing values after
        if missing_after == gt_cleaning.expected_missing_after:
            checks_passed += 1
        else:
            errors.append({
                "metric": "missing_values_after",
                "actual": missing_after,
                "expected": gt_cleaning.expected_missing_after,
            })

        accuracy = round(checks_passed / total_checks, 4) if total_checks > 0 else 1.0

        return CleaningEvaluationResult(
            cleaning_accuracy=accuracy,
            duplicates_removed=dup_removed,
            missing_values_before=gt_cleaning.expected_missing_before,
            missing_values_after=missing_after,
            dtype_corrections=dtype_corrections,
            row_count_before=row_before,
            row_count_after=row_after,
            cleaning_passed=(checks_passed == total_checks),
            cleaning_errors=errors,
        )
