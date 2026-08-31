"""
agents/cleaning/outlier_detector.py
===================================
Deterministic IQR-based outlier detector.
FLAG-ONLY policy: Identifies and records distribution anomalies without modifying or deleting data.
Excludes boolean, date, string, and categorical types.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def detect_column_outliers(
    series: pd.Series,
    col_name: str,
) -> dict[str, Any] | None:
    """
    Detect statistical outliers using the Interquartile Range (IQR) rule.
    Flag-only: Returns a structured provenance record without modifying series values.
    """
    if pd.api.types.is_bool_dtype(series):
        return None

    numeric_vals = pd.to_numeric(series.dropna(), errors="coerce").dropna().astype(float)
    if len(numeric_vals) < 4:
        return None  # Insufficient points for meaningful quartiles

    q1 = float(numeric_vals.quantile(0.25))
    q3 = float(numeric_vals.quantile(0.75))
    iqr = q3 - q1

    if iqr == 0.0:
        return None  # Constant or near-constant series

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (numeric_vals < lower_bound) | (numeric_vals > upper_bound)
    outlier_count = int(outlier_mask.sum())

    if outlier_count > 0:
        outlier_pct = round((outlier_count / len(numeric_vals)) * 100.0, 2)
        outlier_indices = numeric_vals[outlier_mask].index.tolist()

        return {
            "column": col_name,
            "missing_count": 0,
            "missing_pct": 0.0,
            "method": "flag_outliers_iqr",
            "reason": f"IQR outlier detection: [{lower_bound:.2f}, {upper_bound:.2f}] (retained and flagged)",
            "replacement_value": None,
            "reversible": True,
            "details": {
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "outlier_count": outlier_count,
                "outlier_pct": outlier_pct,
                "outlier_indices": outlier_indices,
            },
        }

    return None


def detect_dataframe_outliers(
    df: pd.DataFrame,
    columns_info: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Scan all purely numeric columns in a DataFrame and flag outliers.
    """
    outlier_logs: list[dict[str, Any]] = []
    col_type_map = {c["name"]: c.get("dtype_inferred", "string") for c in (columns_info or [])}

    for col in df.columns:
        col_type = col_type_map.get(col, "string")
        # Explicitly skip booleans, dates, strings, and categories
        if col_type in ("bool", "date", "string", "category") or pd.api.types.is_bool_dtype(df[col]):
            continue

        if col_type in ("int", "float") or pd.api.types.is_numeric_dtype(df[col]):
            log_entry = detect_column_outliers(df[col], col)
            if log_entry:
                outlier_logs.append(log_entry)

    return outlier_logs
