"""
tests/benchmark/evaluators/eda_evaluator.py
===========================================
Evaluator for EDA Agent Performance.
Verifies:
  - statistical summary facts within explicit numerical tolerances (+- 1%)
  - chart generation & structural file integrity (> 500 bytes)
  - PII and identifier exclusion from statistical plots
Produces:
  - eda_accuracy
  - stats_verified_count
  - stats_total_count
  - charts_generated_count
  - charts_verified
  - eda_errors
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

from core.dio import DIO
from tests.benchmark.ground_truth import DatasetGroundTruth


@dataclass
class EDAEvaluationResult:
    eda_accuracy: float
    stats_verified_count: int
    stats_total_count: int
    charts_generated_count: int
    charts_verified: bool
    pii_excluded_from_eda: bool
    eda_errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EDAEvaluator:
    """Evaluates statistical facts and chart generation structurally."""

    def evaluate(
        self,
        dio: DIO,
        ground_truth: DatasetGroundTruth,
        run_dir: Path | None = None,
    ) -> EDAEvaluationResult:
        eda_section = dio.get("eda", {})
        summary_stats = eda_section.get("summary_stats", {})
        chart_paths = eda_section.get("chart_paths", []) or dio.get("artifacts", {}).get("chart_paths", [])

        stats_verified = 0
        stats_total = 0
        errors: list[dict[str, Any]] = []

        # 1. Verify Statistical Facts against ground truth with tolerance
        for col_name, facts in ground_truth.eda.statistical_facts.items():
            col_stats = summary_stats.get(col_name)
            if col_stats is None:
                # Column might have been excluded or missing in summary_stats
                errors.append({
                    "column": col_name,
                    "error": f"Statistical summary missing for column '{col_name}'",
                })
                stats_total += len(facts)
                continue

            for stat_key, exp_val in facts.items():
                stats_total += 1
                act_val = col_stats.get(stat_key)
                if act_val is None:
                    errors.append({
                        "column": col_name,
                        "stat": stat_key,
                        "error": f"Metric '{stat_key}' missing from summary stats",
                    })
                    continue

                try:
                    act_float = float(act_val)
                    exp_float = float(exp_val)
                    # Use 1% tolerance or 0.05 absolute tolerance
                    tolerance = max(0.01 * abs(exp_float), 0.05)
                    if math.isclose(act_float, exp_float, abs_tol=tolerance):
                        stats_verified += 1
                    else:
                        errors.append({
                            "column": col_name,
                            "stat": stat_key,
                            "actual": act_float,
                            "expected": exp_float,
                            "error": f"Value mismatch beyond tolerance ({tolerance})",
                        })
                except (ValueError, TypeError) as e:
                    errors.append({
                        "column": col_name,
                        "stat": stat_key,
                        "error": f"Failed to parse stat value: {e}",
                    })

        # 2. Structural Chart Validation
        charts_count = len(chart_paths)
        valid_charts_count = 0
        for cp in chart_paths:
            path_obj = Path(cp)
            if path_obj.exists() and path_obj.stat().st_size > 500:
                valid_charts_count += 1
            else:
                errors.append({
                    "chart_path": str(cp),
                    "error": "Chart artifact missing or empty (< 500 bytes)",
                })

        charts_verified = (valid_charts_count >= ground_truth.eda.min_charts_expected)

        # 3. Excluded PII / Identifier check: Confirm PII columns and identifiers are not featured in chart artifacts or correlations
        pii_excluded = True
        correlations = eda_section.get("correlations", {})
        corr_cols = set(correlations.get("numeric_columns_analyzed", []))
        corr_cols.update(correlations.get("pearson", {}).keys())
        corr_cols.update(correlations.get("spearman", {}).keys())

        for col_name, c_gt in ground_truth.columns.items():
            if c_gt.is_pii or c_gt.is_identifier:
                if col_name in corr_cols or col_name in correlations:
                    errors.append({
                        "column": col_name,
                        "error": f"PII/Identifier column '{col_name}' was not excluded from correlation analysis",
                    })
                    pii_excluded = False
                for cp in chart_paths:
                    cp_name = Path(cp).name.lower()
                    if f"_{col_name.lower()}." in cp_name or f"_{col_name.lower()}_" in cp_name:
                        errors.append({
                            "column": col_name,
                            "error": f"PII/Identifier column '{col_name}' was featured in chart artifact '{cp_name}'",
                        })
                        pii_excluded = False


        # Calculate overall EDA accuracy
        total_eval_points = stats_total + 1 + 1  # stats + charts + pii exclusion
        passed_eval_points = stats_verified + (1 if charts_verified else 0) + (1 if pii_excluded else 0)
        accuracy = round(passed_eval_points / total_eval_points, 4) if total_eval_points > 0 else 1.0

        return EDAEvaluationResult(
            eda_accuracy=accuracy,
            stats_verified_count=stats_verified,
            stats_total_count=stats_total,
            charts_generated_count=charts_count,
            charts_verified=charts_verified,
            pii_excluded_from_eda=pii_excluded,
            eda_errors=errors,
        )
