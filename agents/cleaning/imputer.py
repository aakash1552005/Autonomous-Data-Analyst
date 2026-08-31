"""
agents/cleaning/imputer.py
==========================
Deterministic missing value imputer for numeric and categorical columns.
Numeric:
  - If abs(skewness) > 1.0 -> median
  - If abs(skewness) <= 1.0 -> mean
Categorical / Date / String:
  - If missing_pct <= 30.0% -> mode
  - If missing_pct > 30.0% -> "Unknown"

Every transformation records full provenance with `reversible=True`.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def impute_numeric_column(
    series: pd.Series,
    col_name: str,
) -> tuple[pd.Series, dict[str, Any] | None]:
    """
    Impute missing values in a numeric series based on distribution skewness.
    Returns: (imputed_series, cleaning_log_entry)
    """
    missing_mask = series.isna()
    missing_count = int(missing_mask.sum())
    if missing_count == 0:
        return series, None

    n_total = len(series)
    missing_pct = round((missing_count / n_total) * 100.0, 2)
    non_null_vals = series.dropna().astype(float)

    if len(non_null_vals) == 0:
        # Edge case: Entire column is null
        replacement_val = 0.0
        imputed_series = series.fillna(replacement_val)
        log_entry = {
            "column": col_name,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "method": "constant",
            "reason": "all values missing, filled with 0.0",
            "replacement_value": replacement_val,
            "reversible": True,
        }
        return imputed_series, log_entry

    # Calculate skewness
    skewness = float(non_null_vals.skew()) if len(non_null_vals) > 2 else 0.0
    if np.isnan(skewness):
        skewness = 0.0

    if abs(skewness) > 1.0:
        replacement_val = float(non_null_vals.median())
        method = "median"
        reason = f"distribution skew = {skewness:.2f} (|skew| > 1), median chosen over mean"
    else:
        replacement_val = float(non_null_vals.mean())
        method = "mean"
        reason = f"distribution skew = {skewness:.2f} (|skew| <= 1), mean chosen"

    if pd.api.types.is_integer_dtype(series) or (non_null_vals % 1 == 0).all():
        replacement_val = round(replacement_val)

    imputed_series = series.fillna(replacement_val)

    log_entry = {
        "column": col_name,
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "method": method,
        "reason": reason,
        "replacement_value": replacement_val,
        "reversible": True,
        "original_null_indices": series[missing_mask].index.tolist(),
    }
    return imputed_series, log_entry


def impute_categorical_column(
    series: pd.Series,
    col_name: str,
) -> tuple[pd.Series, dict[str, Any] | None]:
    """
    Impute missing values in a categorical/string/date series.
    - missing_pct <= 30.0% -> mode
    - missing_pct > 30.0% -> 'Unknown'
    Returns: (imputed_series, cleaning_log_entry)
    """
    missing_mask = series.isna() | (series.astype(str).str.strip() == "") | (series.astype(str).str.lower() == "nan") | (series.astype(str).str.lower() == "none")
    missing_count = int(missing_mask.sum())
    if missing_count == 0:
        return series, None

    n_total = len(series)
    missing_pct = round((missing_count / n_total) * 100.0, 2)
    valid_vals = series[~missing_mask]

    if missing_pct <= 30.0 and len(valid_vals) > 0:
        mode_val = valid_vals.mode().iloc[0]
        replacement_val = str(mode_val)
        method = "mode"
        reason = f"missing percentage = {missing_pct:.1f}% (<= 30%), mode '{replacement_val}' used"
    else:
        replacement_val = "Unknown"
        method = "constant_unknown"
        reason = f"missing percentage = {missing_pct:.1f}% (> 30% or no valid mode), 'Unknown' assigned"

    imputed_series = series.copy()
    imputed_series[missing_mask] = replacement_val

    log_entry = {
        "column": col_name,
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "method": method,
        "reason": reason,
        "replacement_value": replacement_val,
        "reversible": True,
        "original_null_indices": series[missing_mask].index.tolist(),
    }
    return imputed_series, log_entry


def impute_dataframe(
    df: pd.DataFrame,
    columns_info: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Impute all missing values across a DataFrame based on inferred column types.
    """
    cleaned_df = df.copy()
    cleaning_logs: list[dict[str, Any]] = []
    col_type_map = {c["name"]: c.get("dtype_inferred", "string") for c in (columns_info or [])}

    for col in cleaned_df.columns:
        col_type = col_type_map.get(col, "string")
        series = cleaned_df[col]

        if (col_type in ("int", "float") or pd.api.types.is_numeric_dtype(series)) and not pd.api.types.is_bool_dtype(series):
            imputed_series, log_entry = impute_numeric_column(series, col)
            if log_entry:
                cleaned_df[col] = imputed_series
                cleaning_logs.append(log_entry)
        else:
            imputed_series, log_entry = impute_categorical_column(series, col)
            if log_entry:
                cleaned_df[col] = imputed_series
                cleaning_logs.append(log_entry)

    return cleaned_df, cleaning_logs
