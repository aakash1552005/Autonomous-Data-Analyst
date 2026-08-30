"""
agents/intelligence/schema_profiler.py
======================================
Tabular schema profiler.
Calculates dataset dimensions, raw and inferred data types, null percentages,
cardinality, duplicate rows, and structural column anomalies.
"""

from __future__ import annotations

import re
from typing import Any
import pandas as pd
import numpy as np


DATE_LIKE_REGEX = re.compile(
    r"^\s*(\d{1,4})[./\-](\d{1,2})[./\-](\d{1,4})(?:\s+\d{1,2}:\d{1,2}(?::\d{1,2})?)?\s*$"
)


def infer_column_dtype(series: pd.Series) -> str:
    """
    Infer the logical/semantic data type of a pandas Series.
    Returns one of: 'int', 'float', 'string', 'bool', 'date', 'category'.
    """
    # 1. Boolean detection
    if pd.api.types.is_bool_dtype(series):
        return "bool"

    non_null = series.dropna()
    n_non_null = len(non_null)
    if n_non_null == 0:
        return "string"

    # Check for string representations of booleans
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        lower_vals = non_null.astype(str).str.strip().str.lower()
        unique_lower = set(lower_vals.unique())
        if unique_lower.issubset({"true", "false", "0", "1", "yes", "no", "y", "n", "t", "f"}) and len(unique_lower) <= 2:
            return "bool"

    # 2. Datetime detection
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = non_null.astype(str).head(30)
        date_matches = sum(1 for s in sample if DATE_LIKE_REGEX.match(s))
        if date_matches / len(sample) >= 0.75:
            return "date"

    # 3. Numeric detection (int vs float)
    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series):
            return "int"
        # Check if float series actually contains only whole integers
        if (non_null % 1 == 0).all():
            return "int"
        return "float"

    # 4. Try parsing object as numeric
    numeric_converted = pd.to_numeric(non_null, errors="coerce")
    if numeric_converted.notna().sum() / n_non_null >= 0.90:
        if (numeric_converted.dropna() % 1 == 0).all():
            return "int"
        return "float"

    # 5. Categorical vs String
    unique_count = non_null.nunique()
    if unique_count <= 20 or (unique_count / n_non_null <= 0.05 and unique_count < 100):
        return "category"

    return "string"


def profile_schema(df: pd.DataFrame) -> dict[str, Any]:
    """
    Profile the structural schema of a pandas DataFrame.
    """
    n_rows = int(len(df))
    n_columns = int(len(df.columns))
    duplicate_rows_count = int(df.duplicated().sum())

    columns_profile: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = round(null_count / n_rows, 4) if n_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        inferred_type = infer_column_dtype(series)

        columns_profile.append({
            "name": str(col),
            "dtype_raw": str(series.dtype),
            "dtype_inferred": inferred_type,
            "null_pct": null_pct,
            "unique_count": unique_count,
            "is_constant": unique_count <= 1 and null_count == 0,
            "is_empty": null_count == n_rows,
        })

    return {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "duplicate_rows_count": duplicate_rows_count,
        "duplicate_row_pct": round(duplicate_rows_count / n_rows, 4) if n_rows > 0 else 0.0,
        "columns": columns_profile,
    }
