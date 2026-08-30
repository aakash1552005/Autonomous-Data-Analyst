"""
agents/intelligence/quality_scorer.py
=====================================
Deterministic dataset quality scorer.
Evaluates missing values, duplicate rows, ambiguous dates, and low-confidence column types.
Computes a composite score [0-100] and human-readable quality issues list.
"""

from __future__ import annotations

from typing import Any


def calculate_quality_score(
    columns_info: list[dict[str, Any]],
    date_columns_info: list[dict[str, Any]],
    duplicate_row_pct: float = 0.0,
) -> dict[str, Any]:
    """
    Calculate composite data quality score and compiled issue list.
    
    Formula:
    score = 100 - (avg_null_pct * 30)
                - (duplicate_row_pct * 20)
                - (n_ambiguous_dates / n_date_cols * 15 if n_date_cols > 0 else 0)
                - (n_low_confidence_cols / n_cols * 15 if n_cols > 0 else 0)
    """
    issues: list[str] = []
    n_cols = len(columns_info)

    # 1. Null percentage deductions
    if n_cols > 0:
        avg_null_pct = sum(col.get("null_pct", 0.0) for col in columns_info) / n_cols
    else:
        avg_null_pct = 0.0

    null_deduction = avg_null_pct * 30.0

    for col in columns_info:
        col_null_pct = col.get("null_pct", 0.0)
        col_name = col.get("name", "")
        if col_null_pct >= 0.05:
            pct_display = round(col_null_pct * 100, 1)
            issues.append(f"{pct_display}% null values in column '{col_name}'")

    # 2. Duplicate rows deduction
    dup_deduction = duplicate_row_pct * 20.0
    if duplicate_row_pct > 0.0:
        dup_display = round(duplicate_row_pct * 100, 1)
        issues.append(f"Duplicate rows detected ({dup_display}% of dataset)")

    # 3. Ambiguous dates deduction
    n_date_cols = len(date_columns_info)
    ambiguous_date_count = 0
    for d_col in date_columns_info:
        if d_col.get("needs_user_confirmation") or d_col.get("detected_format") == "ambiguous":
            ambiguous_date_count += 1
            issues.append(f"Ambiguous date format in column '{d_col.get('column')}' (requires confirmation)")

    if n_date_cols > 0:
        date_deduction = (ambiguous_date_count / n_date_cols) * 15.0
    else:
        date_deduction = 0.0

    # 4. Low-confidence columns deduction (< 0.70 confidence)
    low_conf_count = 0
    for col in columns_info:
        conf = col.get("confidence", 1.0)
        if conf < 0.70:
            low_conf_count += 1
            issues.append(f"Low-confidence semantic label in column '{col.get('name')}' (confidence: {conf})")

    if n_cols > 0:
        low_conf_deduction = (low_conf_count / n_cols) * 15.0
    else:
        low_conf_deduction = 0.0

    # Final score calculation
    raw_score = 100.0 - null_deduction - dup_deduction - date_deduction - low_conf_deduction
    final_score = int(round(max(0.0, min(100.0, raw_score))))

    return {
        "score": final_score,
        "issues": issues,
        "metrics": {
            "avg_null_pct": round(avg_null_pct, 4),
            "duplicate_row_pct": round(duplicate_row_pct, 4),
            "ambiguous_date_count": ambiguous_date_count,
            "low_confidence_column_count": low_conf_count,
        },
    }
