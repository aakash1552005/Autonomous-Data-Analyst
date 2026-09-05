"""
tests/benchmark/evaluators/baseline_comparator.py
=================================================
Evaluator for Section 15: Baseline Profiler Comparison (ydata-profiling).
Safely checks for optional ydata-profiling dependency.
If unavailable, cleanly records YDATA_PROFILING_UNAVAILABLE without failing the benchmark.
If available, compares EDA metrics against the profiling baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd

from core.dio import DIO
from tests.benchmark.ground_truth import DatasetGroundTruth


@dataclass
class BaselineComparisonResult:
    status: str  # "AVAILABLE" | "YDATA_PROFILING_UNAVAILABLE"
    framework: str = "ydata-profiling"
    comparable_metrics: dict[str, Any] = field(default_factory=dict)
    limitation_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaselineComparator:
    """Compares pipeline EDA output against established profiling baseline."""

    def __init__(self) -> None:
        self.is_available: bool = False
        try:
            import ydata_profiling  # type: ignore # noqa: F401
            self.is_available = True
        except (ImportError, ModuleNotFoundError):
            self.is_available = False

    def evaluate(
        self,
        df: pd.DataFrame,
        dio: DIO,
        ground_truth: DatasetGroundTruth,
    ) -> BaselineComparisonResult:
        if not self.is_available:
            return BaselineComparisonResult(
                status="YDATA_PROFILING_UNAVAILABLE",
                framework="ydata-profiling",
                comparable_metrics={
                    "row_count": dio.get("ingestion", {}).get("n_rows"),
                    "column_count": dio.get("ingestion", {}).get("n_columns"),
                    "quality_score": dio.get("quality", {}).get("score"),
                },
                limitation_note=(
                    "ydata-profiling is not installed in the runtime environment. "
                    "Per Section 15, benchmark comparison records YDATA_PROFILING_UNAVAILABLE "
                    "and continues cleanly without modifying dependencies."
                ),
            )

        # Optional execution if installed
        try:
            from ydata_profiling import ProfileReport  # type: ignore
            profile = ProfileReport(df, minimal=True, progress_bar=False)
            desc = profile.get_description()
            table_stats = desc.table

            return BaselineComparisonResult(
                status="AVAILABLE",
                framework="ydata-profiling",
                comparable_metrics={
                    "profiler_row_count": table_stats.get("n"),
                    "ada_row_count": dio.get("ingestion", {}).get("n_rows"),
                    "profiler_col_count": table_stats.get("n_var"),
                    "ada_col_count": dio.get("ingestion", {}).get("n_columns"),
                    "profiler_missing_cells": table_stats.get("n_cells_missing"),
                },
                limitation_note="Baseline profiling executed successfully.",
            )
        except Exception as e:
            return BaselineComparisonResult(
                status="YDATA_PROFILING_UNAVAILABLE",
                framework="ydata-profiling",
                comparable_metrics={},
                limitation_note=f"ydata-profiling execution encountered an error: {e}",
            )
