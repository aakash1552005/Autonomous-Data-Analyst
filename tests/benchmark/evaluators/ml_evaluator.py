"""
tests/benchmark/evaluators/ml_evaluator.py
==========================================
Evaluator for ML Agent Performance.
Measures:
  - applicability (runs when appropriate, skips safely when insufficient data)
  - target column selection
  - task type matching (classification vs regression)
  - metric presence (R2, RMSE, Accuracy, F1)
  - model artifact persistence
Produces:
  - ml_accuracy
  - ml_ran
  - ml_applicable_expected
  - target_matched
  - task_type_matched
  - metrics_present
  - model_persisted
  - ml_errors
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.dio import DIO
from tests.benchmark.ground_truth import DatasetGroundTruth


@dataclass
class MLEvaluationResult:
    ml_accuracy: float
    ml_ran: bool
    ml_applicable_expected: bool
    target_matched: bool
    task_type_matched: bool
    metrics_present: bool
    model_persisted: bool
    ml_pipeline_compliance_score: float = 1.0
    model_performance: dict[str, Any] | None = None
    ml_errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MLEvaluator:
    """Evaluates ML stage behavior against benchmark ground truth."""

    def evaluate(
        self,
        dio: DIO,
        ground_truth: DatasetGroundTruth,
        run_dir: Path | None = None,
    ) -> MLEvaluationResult:
        ml_section = dio.get("ml", {})
        ml_status = ml_section.get("status", "not_run")
        ml_ran = (ml_status in ("trained", "completed"))
        gt_ml = ground_truth.ml

        errors: list[dict[str, Any]] = []

        if not gt_ml.applicable:
            # ML was expected to be skipped or not applicable (e.g. rows < 30)
            skipped_safely = (ml_status in ("skipped", "insufficient_data", "not_run") or not ml_ran)

            if not skipped_safely:
                errors.append({
                    "error": f"ML was expected to skip, but executed with status '{ml_status}'",
                })

            return MLEvaluationResult(
                ml_accuracy=1.0 if skipped_safely else 0.0,
                ml_ran=ml_ran,
                ml_applicable_expected=False,
                target_matched=True,
                task_type_matched=True,
                metrics_present=True,
                model_persisted=True,
                ml_pipeline_compliance_score=1.0 if skipped_safely else 0.0,
                model_performance=None,
                ml_errors=errors,
            )

        # ML was expected to run
        points = 0
        total_points = 5

        # 1. Pipeline ran ML
        if ml_ran:
            points += 1
        else:
            errors.append({
                "error": f"ML was expected to run, but had status '{ml_status}': {ml_section.get('skip_reason', '')}",
            })

        # 2. Target identification
        act_target = ml_section.get("target_column")
        target_matched = (act_target == gt_ml.expected_target)
        if target_matched:
            points += 1
        else:
            errors.append({
                "metric": "target_column",
                "actual": act_target,
                "expected": gt_ml.expected_target,
            })

        # 3. Task type identification
        act_task = ml_section.get("task_type")
        task_type_matched = (act_task == gt_ml.expected_task_type)
        if task_type_matched:
            points += 1
        else:
            errors.append({
                "metric": "task_type",
                "actual": act_task,
                "expected": gt_ml.expected_task_type,
            })

        # 4. Metric presence
        metrics = ml_section.get("metrics", {})
        metrics_present = bool(metrics and isinstance(metrics, dict))
        if metrics_present:
            points += 1
        else:
            errors.append({
                "error": "ML evaluation metrics dictionary is empty or missing",
            })

        # 5. Model artifact persistence
        model_artifact = dio.get("artifacts", {}).get("model_pkl")
        model_persisted = False
        if model_artifact and Path(model_artifact).exists() and Path(model_artifact).stat().st_size > 100:
            model_persisted = True
            points += 1
        else:
            errors.append({
                "error": "Model artifact (.pkl) missing or empty on disk",
            })

        accuracy = round(points / total_points, 4)

        model_performance = None
        if ml_ran:
            model_performance = {
                "selected_model": ml_section.get("selected_model"),
                "f1": ml_section.get("metrics", {}).get("f1"),
                "roc_auc": ml_section.get("metrics", {}).get("roc_auc"),
                "accuracy": ml_section.get("metrics", {}).get("accuracy"),
                "precision": ml_section.get("metrics", {}).get("precision"),
                "recall": ml_section.get("metrics", {}).get("recall"),
                "improved_over_baseline": ml_section.get("improved_over_baseline", False),
                "selection_reason": ml_section.get("selection_reason", ""),
            }

        return MLEvaluationResult(
            ml_accuracy=accuracy,
            ml_ran=ml_ran,
            ml_applicable_expected=True,
            target_matched=target_matched,
            task_type_matched=task_type_matched,
            metrics_present=metrics_present,
            model_persisted=model_persisted,
            ml_pipeline_compliance_score=accuracy,
            model_performance=model_performance,
            ml_errors=errors,
        )
