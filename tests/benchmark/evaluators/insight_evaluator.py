"""
tests/benchmark/evaluators/insight_evaluator.py
==============================================
Evaluator for Insight Agent Numerical Grounding and Hallucination Protection.
Extracts numbers and metrics from generated insight text and evidence objects,
and verifies grounding against DIO factual telemetry and EDA statistics.
Produces:
  - insight_grounding_accuracy
  - supported_claim_count
  - unsupported_claim_count
  - fabricated_numeric_claim_count
  - total_insights_evaluated
  - insight_errors
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from core.dio import DIO
from tests.benchmark.ground_truth import DatasetGroundTruth

# Pattern to extract numeric values from narrative text
NUMBER_REGEX = re.compile(r"\b(?:\$)?(\d+(?:\.\d+)?)(?:%)?\b")


@dataclass
class InsightEvaluationResult:
    insight_grounding_accuracy: float
    supported_claim_count: int
    unsupported_claim_count: int
    fabricated_numeric_claim_count: int
    total_insights_evaluated: int
    insight_errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def grounding_accuracy(self) -> float:
        return self.insight_grounding_accuracy

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["grounding_accuracy"] = self.insight_grounding_accuracy
        return d


class InsightEvaluator:
    """Evaluates factual grounding and hallucination resistance of insights."""

    def evaluate(self, dio: DIO, ground_truth: DatasetGroundTruth) -> InsightEvaluationResult:
        insights = dio.get("insights", [])
        if not insights:
            # If no insights were generated
            return InsightEvaluationResult(
                insight_grounding_accuracy=1.0,
                supported_claim_count=0,
                unsupported_claim_count=0,
                fabricated_numeric_claim_count=0,
                total_insights_evaluated=0,
                insight_errors=[],
            )

        # Collect all grounded known numbers from DIO
        known_numbers: set[float] = set()

        # From ingestion
        ing = dio.get("ingestion", {})
        if "n_rows" in ing:
            known_numbers.add(float(ing["n_rows"]))
        if "n_columns" in ing:
            known_numbers.add(float(ing["n_columns"]))

        # From quality
        if "quality" in dio and "score" in dio["quality"]:
            known_numbers.add(float(dio["quality"]["score"]))

        # From EDA summary stats
        eda_stats = dio.get("eda", {}).get("summary_stats", {})
        for col_name, stats_dict in eda_stats.items():
            if isinstance(stats_dict, dict):
                for val in stats_dict.values():
                    try:
                        known_numbers.add(round(float(val), 2))
                    except (ValueError, TypeError):
                        pass

        # From ML metrics
        ml_metrics = dio.get("ml", {}).get("metrics", {})
        if isinstance(ml_metrics, dict):
            for val in ml_metrics.values():
                try:
                    known_numbers.add(round(float(val), 2))
                except (ValueError, TypeError):
                    pass

        # From ground truth known numbers
        for num in ground_truth.insights.verifiable_numeric_values:
            known_numbers.add(round(float(num), 2))

        supported_claims = 0
        unsupported_claims = 0
        fabricated_count = 0
        errors: list[dict[str, Any]] = []

        for idx, insight in enumerate(insights):
            text = str(insight.get("text", "") if isinstance(insight, dict) else getattr(insight, "text", ""))
            evidence = insight.get("evidence", []) if isinstance(insight, dict) else getattr(insight, "evidence", [])
            # Strip fixed metric scale bounds (e.g. '/100' or 'out of 100') to avoid
            # treating the scale denominator as a factual dataset claim.
            cleaned_text = re.sub(r"(?:/|\bout of\s+)\s*100\b", "", text)
            numbers_found = NUMBER_REGEX.findall(cleaned_text)

            is_insight_grounded = True

            # Check if numbers cited in narrative are grounded
            for num_str in numbers_found:
                try:
                    val = round(float(num_str), 2)
                    # Ignore small index-like or ranking numbers (e.g. 1, 2, 3) unless significant
                    if val in (1.0, 2.0, 3.0):
                        continue

                    # Check if val is close to any known number
                    is_known = any(abs(val - known) <= 0.05 * max(abs(val), abs(known), 1.0) for known in known_numbers)
                    if not is_known:
                        fabricated_count += 1
                        is_insight_grounded = False
                        errors.append({
                            "insight_index": idx,
                            "cited_number": val,
                            "error": f"Cited number {val} is not grounded in DIO facts or summary statistics",
                        })
                except (ValueError, TypeError):
                    pass

            # A claim is supported ONLY IF it has evidence AND zero fabricated numbers
            if evidence and is_insight_grounded:
                supported_claims += 1
            else:
                unsupported_claims += 1

        total_eval_points = supported_claims + unsupported_claims + fabricated_count
        accuracy = round(supported_claims / total_eval_points, 4) if total_eval_points > 0 else 1.0

        return InsightEvaluationResult(
            insight_grounding_accuracy=accuracy,
            supported_claim_count=supported_claims,
            unsupported_claim_count=unsupported_claims,
            fabricated_numeric_claim_count=fabricated_count,
            total_insights_evaluated=len(insights),
            insight_errors=errors,
        )
