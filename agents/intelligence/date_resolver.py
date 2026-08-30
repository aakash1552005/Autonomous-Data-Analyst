"""
agents/intelligence/date_resolver.py
====================================
Tiered deterministic date format resolver.
Never silently guesses ambiguous dates; marks confidence and user confirmation requirements explicitly.
"""

from __future__ import annotations

import re
from typing import Any
import pandas as pd


DATE_REGEX = re.compile(
    r"^\s*(\d{1,4})([./\-])(\d{1,4})\2(\d{1,4})(?:\s+\d{1,2}:\d{1,2}(?::\d{1,2})?)?\s*$"
)

# Common country-to-date-format mapping
LOCALE_DATE_MAP = {
    "us": "MM/DD/YYYY",
    "usa": "MM/DD/YYYY",
    "united states": "MM/DD/YYYY",
    "uk": "DD/MM/YYYY",
    "united kingdom": "DD/MM/YYYY",
    "great britain": "DD/MM/YYYY",
    "india": "DD/MM/YYYY",
    "in": "DD/MM/YYYY",
    "australia": "DD/MM/YYYY",
    "au": "DD/MM/YYYY",
    "canada": "YYYY-MM-DD",
    "germany": "DD.MM.YYYY",
    "de": "DD.MM.YYYY",
    "france": "DD/MM/YYYY",
    "fr": "DD/MM/YYYY",
}


def resolve_date_column(
    series: pd.Series,
    col_name: str,
    already_resolved_formats: dict[str, str] | None = None,
    dataset_df: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """
    Analyze a Series and resolve its date format using deterministic tiered rules.
    Returns date metadata dict or None if the column is not a date.
    """
    already_resolved_formats = already_resolved_formats or {}
    non_null = series.dropna().astype(str).str.strip()
    if len(non_null) == 0:
        return None

    # Check ISO datetime or pre-parsed datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return {
            "column": col_name,
            "detected_format": "YYYY-MM-DD",
            "confidence": 1.0,
            "evidence": ["native datetime dtype"],
            "needs_user_confirmation": False,
        }

    parsed_tuples: list[tuple[int, int, int, str]] = []
    for val in non_null.head(100):
        match = DATE_REGEX.match(val)
        if match:
            p1, sep, p2, p3 = int(match.group(1)), match.group(2), int(match.group(3)), int(match.group(4))
            parsed_tuples.append((p1, p2, p3, sep))

    if not parsed_tuples or (len(parsed_tuples) / min(len(non_null), 100)) < 0.60:
        return None  # Not a date column

    sep = parsed_tuples[0][3]

    # --- Tier 0: ISO Format (YYYY-MM-DD or YYYY/MM/DD) ---
    iso_matches = [t for t in parsed_tuples if t[0] >= 1000]
    if len(iso_matches) == len(parsed_tuples):
        return {
            "column": col_name,
            "detected_format": f"YYYY{sep}MM{sep}DD",
            "confidence": 1.0,
            "evidence": ["four-digit year in first position"],
            "needs_user_confirmation": False,
        }

    # --- Tier 1: Deterministic Day > 12 Disambiguation ---
    has_day_first = False
    has_day_second = False

    for p1, p2, p3, _ in parsed_tuples:
        # e.g. p1=25, p2=10 -> 25 > 12 -> day is first
        if p1 > 12 and p2 <= 12:
            has_day_first = True
        elif p2 > 12 and p1 <= 12:
            has_day_second = True

    if has_day_first and not has_day_second:
        return {
            "column": col_name,
            "detected_format": f"DD{sep}MM{sep}YYYY",
            "confidence": 1.0,
            "evidence": ["day value > 12 found in first position"],
            "needs_user_confirmation": False,
        }

    if has_day_second and not has_day_first:
        return {
            "column": col_name,
            "detected_format": f"MM{sep}DD{sep}YYYY",
            "confidence": 1.0,
            "evidence": ["day value > 12 found in second position"],
            "needs_user_confirmation": False,
        }

    # --- Tier 2: Check other already-resolved date columns in same dataset ---
    for other_col, other_fmt in already_resolved_formats.items():
        if other_col != col_name and other_fmt != "ambiguous":
            return {
                "column": col_name,
                "detected_format": other_fmt,
                "confidence": 0.75,
                "evidence": [f"matched format of resolved column '{other_col}'"],
                "needs_user_confirmation": False,
            }

    # --- Tier 3: Check Country / Locale Column in Dataset ---
    if dataset_df is not None:
        for c in dataset_df.columns:
            if str(c).lower() in ("country", "locale", "nation", "location"):
                locale_vals = dataset_df[c].dropna().astype(str).str.lower().head(10)
                for loc in locale_vals:
                    if loc in LOCALE_DATE_MAP:
                        target_fmt = LOCALE_DATE_MAP[loc]
                        return {
                            "column": col_name,
                            "detected_format": target_fmt,
                            "confidence": 0.70,
                            "evidence": [f"inferred from country/locale column '{c}' value '{loc}'"],
                            "needs_user_confirmation": False,
                        }

    # --- Tier 4: Fallback — Mark Ambiguous (NEVER SILENTLY GUESS) ---
    return {
        "column": col_name,
        "detected_format": "ambiguous",
        "confidence": 0.50,
        "evidence": ["all day and month components <= 12; no disambiguating signal"],
        "needs_user_confirmation": True,
    }


def resolve_all_dates(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Scan all columns in a DataFrame and resolve date formats.
    """
    resolved_columns: list[dict[str, Any]] = []
    known_formats: dict[str, str] = {}

    for col in df.columns:
        result = resolve_date_column(df[col], str(col), already_resolved_formats=known_formats, dataset_df=df)
        if result is not None:
            resolved_columns.append(result)
            if result["detected_format"] != "ambiguous":
                known_formats[str(col)] = result["detected_format"]

    return resolved_columns
