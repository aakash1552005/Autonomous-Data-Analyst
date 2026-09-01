"""
agents/ml/target_validator.py
=============================
Target validation and data sufficiency checks before training.
Enforces configurable min_rows_for_ml and min_rows_per_class thresholds.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def validate_target_and_data(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    min_rows_for_ml: int = 30,
    min_rows_per_class: int = 5,
) -> tuple[bool, dict[str, Any], str | None]:
    """
    Validate target and dataset sufficiency.
    Returns (is_valid, validation_details, error_code).
    """
    if target_col not in df.columns:
        return False, {}, "ML_001_TARGET_NOT_FOUND"

    s_target = df[target_col].dropna()
    total_valid_rows = len(s_target)

    # 1. Total row count check vs config threshold
    if total_valid_rows < min_rows_for_ml:
        return False, {
            "total_rows": total_valid_rows,
            "min_rows_for_ml": min_rows_for_ml,
            "reason": f"Dataset has {total_valid_rows} rows, which is below the configured threshold of {min_rows_for_ml}",
        }, "ML_003_INSUFFICIENT_DATA"

    validation_details: dict[str, Any] = {
        "target_column": target_col,
        "task_type": task_type,
        "total_rows": total_valid_rows,
        "null_count": int(df[target_col].isna().sum()),
    }

    if task_type == "classification":
        val_counts = s_target.astype(str).value_counts()
        n_classes = len(val_counts)

        if n_classes < 2:
            return False, validation_details, "ML_002_INVALID_TARGET"

        class_counts = {str(k): int(v) for k, v in val_counts.items()}
        class_percentages = {str(k): round(float(v / total_valid_rows * 100.0), 2) for k, v in val_counts.items()}
        min_class_count = int(val_counts.min())
        max_class_count = int(val_counts.max())
        imbalance_ratio = round(max_class_count / max(1, min_class_count), 2)

        validation_details.update({
            "n_classes": n_classes,
            "class_counts": class_counts,
            "class_percentages": class_percentages,
            "min_class_count": min_class_count,
            "imbalance_ratio": imbalance_ratio,
        })

        # Check min_rows_per_class vs config threshold
        if min_class_count < min_rows_per_class:
            validation_details["reason"] = (
                f"Minority class has {min_class_count} samples, which is below the configured min_rows_per_class of {min_rows_per_class}"
            )
            return False, validation_details, "ML_003_INSUFFICIENT_DATA"

    elif task_type == "regression":
        nums = pd.to_numeric(s_target, errors="coerce").dropna()
        if len(nums) < min_rows_for_ml:
            return False, validation_details, "ML_003_INSUFFICIENT_DATA"

        mean_val = float(nums.mean())
        std_val = float(nums.std()) if len(nums) > 1 else 0.0
        skew_val = float(nums.skew()) if len(nums) > 2 else 0.0

        if std_val == 0.0:
            return False, validation_details, "ML_002_INVALID_TARGET"

        validation_details.update({
            "mean": round(mean_val, 4),
            "median": round(float(nums.median()), 4),
            "std": round(std_val, 4),
            "min": round(float(nums.min()), 4),
            "max": round(float(nums.max()), 4),
            "skewness": round(skew_val, 4) if not np.isnan(skew_val) else 0.0,
        })

    return True, validation_details, None
