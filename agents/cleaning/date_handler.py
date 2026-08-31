"""
agents/cleaning/date_handler.py
===============================
Deterministic date normalization handler.
Normalizes verified date columns to ISO format (YYYY-MM-DD).
Strictly preserves ambiguous date columns in their original raw state with explicit logging.
"""

from __future__ import annotations

import re
from typing import Any
import pandas as pd


# Format string conversion table
FORMAT_TO_STRFTIME = {
    "DD/MM/YYYY": "%d/%m/%Y",
    "DD-MM-YYYY": "%d-%m-%Y",
    "DD.MM.YYYY": "%d.%m.%Y",
    "MM/DD/YYYY": "%m/%d/%Y",
    "MM-DD-YYYY": "%m-%d-%Y",
    "MM.DD.YYYY": "%m.%d.%Y",
    "YYYY/MM/DD": "%Y/%m/%d",
    "YYYY-MM-DD": "%Y-%m-%d",
    "YYYY.MM.DD": "%Y.%m.%d",
}


def normalize_date_column(
    series: pd.Series,
    date_meta: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Normalize a date column if resolved with high confidence, or preserve if ambiguous.
    """
    col_name = date_meta.get("column", "")
    detected_format = date_meta.get("detected_format", "ambiguous")
    needs_confirmation = date_meta.get("needs_user_confirmation", False)

    # 1. Ambiguous dates protection (HARD REQUIREMENT: NEVER SILENTLY GUESS)
    if needs_confirmation or detected_format == "ambiguous":
        log_entry = {
            "column": col_name,
            "missing_count": int(series.isna().sum()),
            "missing_pct": round((series.isna().sum() / len(series)) * 100.0, 2),
            "method": "preserve_ambiguous_date",
            "reason": "Date format is ambiguous and requires user confirmation; preserved raw values unchanged.",
            "replacement_value": None,
            "reversible": True,
            "warning": True,
        }
        return series, log_entry

    # 2. Resolved dates normalization
    strftime_pattern = FORMAT_TO_STRFTIME.get(detected_format)
    if not strftime_pattern:
        # Fall back to general regex / flexible parsing if exact separator differs
        if "DD" in detected_format and detected_format.startswith("DD"):
            strftime_pattern = "%d/%m/%Y"
        elif "MM" in detected_format and detected_format.startswith("MM"):
            strftime_pattern = "%m/%d/%Y"
        else:
            strftime_pattern = "%Y-%m-%d"

    try:
        # Parse non-null string dates
        parsed_dates = pd.to_datetime(series, format=strftime_pattern, errors="coerce")
        # Format cleanly as ISO YYYY-MM-DD string while preserving NaNs
        formatted_series = parsed_dates.dt.strftime("%Y-%m-%d").where(parsed_dates.notna(), series)

        log_entry = {
            "column": col_name,
            "missing_count": int(formatted_series.isna().sum()),
            "missing_pct": round((formatted_series.isna().sum() / len(series)) * 100.0, 2),
            "method": "normalize_date",
            "reason": f"Normalized date format from '{detected_format}' to ISO 'YYYY-MM-DD'",
            "replacement_value": "YYYY-MM-DD",
            "reversible": True,
        }
        return formatted_series, log_entry
    except Exception:
        # If parsing fails on unexpected values, keep raw and log warning
        log_entry = {
            "column": col_name,
            "missing_count": int(series.isna().sum()),
            "missing_pct": round((series.isna().sum() / len(series)) * 100.0, 2),
            "method": "preserve_date_parse_error",
            "reason": f"Date parsing failed for format '{detected_format}'; preserved raw values.",
            "replacement_value": None,
            "reversible": True,
            "warning": True,
        }
        return series, log_entry


def handle_all_dates(
    df: pd.DataFrame,
    date_columns_info: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Process all date columns in DataFrame according to DIO date metadata.
    """
    cleaned_df = df.copy()
    cleaning_logs: list[dict[str, Any]] = []

    for d_meta in date_columns_info:
        col_name = d_meta.get("column")
        if col_name in cleaned_df.columns:
            normalized_series, log_entry = normalize_date_column(cleaned_df[col_name], d_meta)
            cleaned_df[col_name] = normalized_series
            cleaning_logs.append(log_entry)

    return cleaned_df, cleaning_logs
