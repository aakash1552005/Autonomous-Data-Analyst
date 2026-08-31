"""
agents/cleaning/type_coercer.py
===============================
Deterministic type coercion utility.
Coerces DataFrame columns to logical types inferred by Intelligence Agent.
Converts unparseable values to null for downstream imputation.
Never mutates identifier or PII columns destructively.
"""

from __future__ import annotations

from typing import Any
import pandas as pd


BOOL_TRUE_SET = {"true", "1", "yes", "t", "y"}
BOOL_FALSE_SET = {"false", "0", "no", "f", "n"}


def coerce_column_type(
    series: pd.Series,
    target_type: str,
    col_name: str,
    is_pii: bool = False,
    semantic_label: str = "",
) -> tuple[pd.Series, dict[str, Any] | None]:
    """
    Coerce a single column to its inferred logical type.
    """
    # Do not coerce identifiers or PII into destructive types
    if is_pii or semantic_label in ("identifier", "email", "phone", "person_name", "credit_card", "ssn_national_id"):
        return series, None

    orig_dtype = str(series.dtype)
    coerced = series.copy()
    coerced_flag = False

    try:
        if target_type == "int":
            # Coerce non-numerics to NaN
            numeric_vals = pd.to_numeric(coerced, errors="coerce")
            coerced = numeric_vals
            coerced_flag = True

        elif target_type == "float":
            numeric_vals = pd.to_numeric(coerced, errors="coerce").astype(float)
            coerced = numeric_vals
            coerced_flag = True

        elif target_type == "bool":
            def parse_bool(v: Any) -> Any:
                if pd.isna(v):
                    return None
                s = str(v).strip().lower()
                if s in BOOL_TRUE_SET:
                    return True
                if s in BOOL_FALSE_SET:
                    return False
                return None

            coerced = coerced.apply(parse_bool)
            coerced_flag = True

        elif target_type == "category":
            coerced = coerced.astype(str).str.strip().replace({"nan": None, "None": None})
            coerced_flag = True

    except Exception:
        # Fall back safely if coercion fails
        return series, None

    if coerced_flag:
        log_entry = {
            "column": col_name,
            "missing_count": 0,
            "missing_pct": 0.0,
            "method": "type_coercion",
            "reason": f"coerced raw dtype '{orig_dtype}' to inferred type '{target_type}'",
            "replacement_value": target_type,
            "reversible": True,
        }
        return coerced, log_entry

    return series, None


def coerce_dataframe_types(
    df: pd.DataFrame,
    columns_info: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Coerce all columns in a DataFrame according to the intelligence columns profile.
    """
    cleaned_df = df.copy()
    cleaning_logs: list[dict[str, Any]] = []

    for col_info in columns_info:
        col_name = col_info.get("name")
        if col_name in cleaned_df.columns:
            target_type = col_info.get("dtype_inferred", "string")
            is_pii = col_info.get("is_pii", False)
            semantic_label = col_info.get("semantic_label", "")

            coerced_series, log_entry = coerce_column_type(
                series=cleaned_df[col_name],
                target_type=target_type,
                col_name=col_name,
                is_pii=is_pii,
                semantic_label=semantic_label,
            )
            cleaned_df[col_name] = coerced_series
            if log_entry:
                cleaning_logs.append(log_entry)

    return cleaned_df, cleaning_logs
