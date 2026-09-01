"""
agents/eda/summary_stats.py
===========================
Deterministic summary statistics computation for numeric and categorical columns.
Calculates five-number summaries, skewness, kurtosis, IQR, and category distributions.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def compute_numeric_summary(series: pd.Series, col_name: str) -> dict[str, Any]:
    """
    Compute comprehensive numeric summary statistics for a numeric Series.
    """
    valid_nums = pd.to_numeric(series.dropna(), errors="coerce").dropna().astype(float)
    n_total = len(series)
    null_count = int(series.isna().sum())
    null_pct = round((null_count / n_total * 100.0) if n_total > 0 else 0.0, 2)

    if len(valid_nums) == 0:
        return {
            "column": col_name,
            "type": "numeric",
            "count": 0,
            "null_count": null_count,
            "null_pct": null_pct,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
            "iqr": None,
            "skewness": None,
            "kurtosis": None,
        }

    q25 = float(valid_nums.quantile(0.25))
    q75 = float(valid_nums.quantile(0.75))
    iqr = round(q75 - q25, 4)
    skewness = float(valid_nums.skew()) if len(valid_nums) > 2 else 0.0
    kurtosis = float(valid_nums.kurt()) if len(valid_nums) > 3 else 0.0

    return {
        "column": col_name,
        "type": "numeric",
        "count": int(len(valid_nums)),
        "null_count": null_count,
        "null_pct": null_pct,
        "mean": round(float(valid_nums.mean()), 4),
        "std": round(float(valid_nums.std()), 4) if len(valid_nums) > 1 else 0.0,
        "min": round(float(valid_nums.min()), 4),
        "q25": round(q25, 4),
        "median": round(float(valid_nums.median()), 4),
        "q75": round(q75, 4),
        "max": round(float(valid_nums.max()), 4),
        "iqr": iqr,
        "skewness": round(skewness, 4) if not np.isnan(skewness) else 0.0,
        "kurtosis": round(kurtosis, 4) if not np.isnan(kurtosis) else 0.0,
    }


def compute_categorical_summary(series: pd.Series, col_name: str) -> dict[str, Any]:
    """
    Compute distribution and frequency metrics for a categorical/string/bool Series.
    """
    valid_vals = series.dropna().astype(str).str.strip()
    valid_vals = valid_vals[valid_vals != ""]
    n_total = len(series)
    null_count = int(series.isna().sum())
    null_pct = round((null_count / n_total * 100.0) if n_total > 0 else 0.0, 2)

    if len(valid_vals) == 0:
        return {
            "column": col_name,
            "type": "categorical",
            "count": 0,
            "unique_count": 0,
            "null_count": null_count,
            "null_pct": null_pct,
            "top_category": None,
            "top_category_freq": 0,
            "top_categories": {},
        }

    val_counts = valid_vals.value_counts()
    top_cat = str(val_counts.index[0])
    top_freq = int(val_counts.iloc[0])
    top_5 = {str(k): int(v) for k, v in val_counts.head(5).items()}

    return {
        "column": col_name,
        "type": "categorical",
        "count": int(len(valid_vals)),
        "unique_count": int(valid_vals.nunique()),
        "null_count": null_count,
        "null_pct": null_pct,
        "top_category": top_cat,
        "top_category_freq": top_freq,
        "top_categories": top_5,
    }


def compute_dataset_summary_stats(
    df: pd.DataFrame,
    columns_info: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Compute summary statistics across all columns in DataFrame.
    """
    col_type_map = {c["name"]: c.get("dtype_inferred", "string") for c in (columns_info or [])}
    summaries: dict[str, Any] = {}

    for col in df.columns:
        col_type = col_type_map.get(col, "string")
        series = df[col]

        if (col_type in ("int", "float") or pd.api.types.is_numeric_dtype(series)) and not pd.api.types.is_bool_dtype(series):
            summaries[col] = compute_numeric_summary(series, col)
        else:
            summaries[col] = compute_categorical_summary(series, col)

    return summaries
