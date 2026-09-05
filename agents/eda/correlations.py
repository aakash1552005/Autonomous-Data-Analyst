"""
agents/eda/correlations.py
==========================
Deterministic correlation analysis for numeric columns.
Computes Pearson and Spearman correlation matrices and identifies top feature associations.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def compute_correlations(
    df: pd.DataFrame,
    columns_info: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Compute Pearson and Spearman correlation matrices and top correlated pairs.
    """
    col_type_map = {c["name"]: c.get("dtype_inferred", "string") for c in (columns_info or [])}
    pii_cols = {c["name"] for c in (columns_info or []) if c.get("is_pii", False)}
    identifier_cols = {c["name"] for c in (columns_info or []) if c.get("semantic_label") == "identifier"}
    numeric_cols: list[str] = []

    for col in df.columns:
        if col in pii_cols or col in identifier_cols:
            continue
        col_type = col_type_map.get(col, "string")
        if (col_type in ("int", "float") or pd.api.types.is_numeric_dtype(df[col])) and not pd.api.types.is_bool_dtype(df[col]):
            # Verify has non-null variation
            clean_s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(clean_s) > 1 and clean_s.std() > 0:
                numeric_cols.append(col)

    if len(numeric_cols) < 2:
        return {
            "pearson": {},
            "spearman": {},
            "top_correlations": [],
            "numeric_columns_analyzed": numeric_cols,
        }

    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Pearson Matrix
    pearson_matrix = numeric_df.corr(method="pearson").round(4).to_dict()
    # Spearman Matrix
    spearman_matrix = numeric_df.corr(method="spearman").round(4).to_dict()

    # Extract non-diagonal top pairs
    top_pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    corr_df = numeric_df.corr(method="pearson")
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:
                pair_key = (col1, col2)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    val = corr_df.loc[col1, col2]
                    if not np.isnan(val):
                        top_pairs.append({
                            "col1": col1,
                            "col2": col2,
                            "pearson_r": round(float(val), 4),
                            "abs_r": round(float(abs(val)), 4),
                        })

    # Sort descending by absolute correlation strength
    top_pairs.sort(key=lambda x: x["abs_r"], reverse=True)

    return {
        "pearson": pearson_matrix,
        "spearman": spearman_matrix,
        "top_correlations": top_pairs,
        "numeric_columns_analyzed": numeric_cols,
    }
