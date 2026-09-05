"""
tests/benchmark/evaluators/semantic_evaluator.py
================================================
Evaluator for Metric 1: Semantic Label Accuracy.
Measures Intelligence Agent semantic-label performance against benchmark ground truth.
Produces:
  - semantic_label_accuracy
  - semantic_label_correct
  - semantic_label_total
  - semantic_label_errors
  - low_confidence_count
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.dio import DIO
from tests.benchmark.ground_truth import DatasetGroundTruth


@dataclass
class SemanticEvaluationResult:
    semantic_label_accuracy: float
    semantic_label_correct: int
    semantic_label_total: int
    semantic_label_errors: list[dict[str, Any]] = field(default_factory=list)
    low_confidence_count: int = 0
    macro_accuracy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SemanticEvaluator:
    """Evaluates predicted semantic labels in DIO against ground truth."""

    def evaluate(self, dio: DIO, ground_truth: DatasetGroundTruth) -> SemanticEvaluationResult:
        columns = dio.get("columns", [])
        if not columns:
            return SemanticEvaluationResult(
                semantic_label_accuracy=0.0,
                semantic_label_correct=0,
                semantic_label_total=len(ground_truth.columns),
                semantic_label_errors=[
                    {"column": col_name, "error": "Missing column in DIO"}
                    for col_name in ground_truth.columns
                ],
                low_confidence_count=0,
                macro_accuracy=0.0,
            )

        correct_count = 0
        total_count = 0
        errors: list[dict[str, Any]] = []
        low_confidence_count = 0
        class_correct: dict[str, int] = {}
        class_total: dict[str, int] = {}

        dio_cols_by_name = {c.get("name", "").strip(): c for c in columns}

        for col_name, col_gt in ground_truth.columns.items():
            total_count += 1
            clean_name = col_name.strip()
            pred_col = dio_cols_by_name.get(clean_name) or dio_cols_by_name.get(col_name)

            if pred_col is None:
                errors.append({
                    "column": col_name,
                    "predicted_label": None,
                    "expected_labels": list(col_gt.expected_semantic_labels),
                    "confidence": 0.0,
                    "reason": "Column missing from DIO",
                })
                continue

            pred_label = pred_col.get("semantic_label", "unknown").lower()
            confidence = float(pred_col.get("confidence", 0.0))

            if confidence < 0.70:
                low_confidence_count += 1

            # Match against allowed expected labels / synonyms
            primary_class = col_gt.expected_semantic_labels[0]
            class_total[primary_class] = class_total.get(primary_class, 0) + 1

            is_match = any(
                pred_label == exp.lower() or pred_label in exp.lower() or exp.lower() in pred_label
                for exp in col_gt.expected_semantic_labels
            )

            if is_match:
                correct_count += 1
                class_correct[primary_class] = class_correct.get(primary_class, 0) + 1
            else:
                errors.append({
                    "column": col_name,
                    "predicted_label": pred_label,
                    "expected_labels": list(col_gt.expected_semantic_labels),
                    "confidence": confidence,
                    "reason": f"Predicted '{pred_label}' does not match expected {list(col_gt.expected_semantic_labels)}",
                })

        accuracy = round(correct_count / total_count, 4) if total_count > 0 else 1.0

        # Calculate macro accuracy across expected label classes
        macro_accs = [
            class_correct.get(cls, 0) / class_total[cls]
            for cls in class_total
            if class_total[cls] > 0
        ]
        macro_accuracy = round(sum(macro_accs) / len(macro_accs), 4) if macro_accs else 1.0

        return SemanticEvaluationResult(
            semantic_label_accuracy=accuracy,
            semantic_label_correct=correct_count,
            semantic_label_total=total_count,
            semantic_label_errors=errors,
            low_confidence_count=low_confidence_count,
            macro_accuracy=macro_accuracy,
        )
