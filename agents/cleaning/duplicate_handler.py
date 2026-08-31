"""
agents/cleaning/duplicate_handler.py
====================================
Deterministic duplicate row handler.
Detects exact duplicate rows, separates them into a sidecar DataFrame,
and preserves original row indices for lossless round-trip reconstruction.
"""

from __future__ import annotations

from typing import Any
import pandas as pd


def handle_duplicate_rows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    """
    Separate exact duplicate rows from DataFrame while maintaining original indices.
    Returns: (cleaned_df, removed_rows_df, log_entry)
    """
    dup_mask = df.duplicated(keep="first")
    n_duplicates = int(dup_mask.sum())

    if n_duplicates == 0:
        empty_removed = pd.DataFrame(columns=list(df.columns) + ["_orig_row_index"])
        return df.copy(), empty_removed, None

    # Preserve removed rows with original index preserved
    removed_rows = df[dup_mask].copy()
    removed_rows["_orig_row_index"] = removed_rows.index

    # Cleaned rows retain first occurrence
    cleaned_df = df[~dup_mask].copy()

    log_entry = {
        "column": "ALL_COLUMNS",
        "missing_count": 0,
        "missing_pct": 0.0,
        "method": "drop_duplicates",
        "reason": f"Exact duplicate rows detected ({n_duplicates} rows removed, preserved in sidecar)",
        "replacement_value": None,
        "reversible": True,
        "details": {
            "duplicate_count": n_duplicates,
            "duplicate_pct": round((n_duplicates / len(df)) * 100.0, 2),
            "removed_indices": removed_rows.index.tolist(),
        },
    }

    return cleaned_df, removed_rows, log_entry
