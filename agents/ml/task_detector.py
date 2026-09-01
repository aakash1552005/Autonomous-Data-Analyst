"""
agents/ml/task_detector.py
==========================
Deterministic task detection and target column identification from DIO metadata.
Determines whether supervised classification, regression, or no ML is appropriate.
"""

from __future__ import annotations

from typing import Any
import pandas as pd


def detect_ml_task(
    df: pd.DataFrame,
    columns_info: list[dict[str, Any]] | None = None,
    preferred_target: str | None = None,
) -> dict[str, Any]:
    """
    Deterministically identify target column and determine ML task type.
    """
    col_dict = {c["name"]: c for c in (columns_info or [])}
    
    # 1. Target candidate selection
    selected_target: str | None = None
    target_reason: str = ""

    if preferred_target and preferred_target in df.columns:
        c_meta = col_dict.get(preferred_target, {})
        if not c_meta.get("is_pii", False) and c_meta.get("semantic_label") != "identifier":
            selected_target = preferred_target
            target_reason = f"Explicit user/configuration preferred target '{preferred_target}'"

    if selected_target is None:
        # Check explicit target candidates from Intelligence Agent
        candidates: list[str] = []
        for col in df.columns:
            meta = col_dict.get(col, {})
            # Reject PII and identifiers
            if meta.get("is_pii", False) or meta.get("semantic_label") == "identifier":
                continue
            if meta.get("is_target_candidate", False) or meta.get("semantic_label") in ("target_label", "target_value"):
                candidates.append(col)

        if len(candidates) == 1:
            selected_target = candidates[0]
            target_reason = f"Identified unambiguous target candidate '{selected_target}' from Intelligence DIO metadata"
        elif len(candidates) > 1:
            # Deterministic selection priority:
            # 1. Column marked with semantic_label target_label / target_value
            explicit_labels = [c for c in candidates if col_dict.get(c, {}).get("semantic_label") in ("target_label", "target_value")]
            if explicit_labels:
                selected_target = explicit_labels[0]
                target_reason = f"Selected semantic target label '{selected_target}'"
            else:
                # 2. Target keyword patterns in column name
                target_keywords = ["default", "churn", "target", "label", "status", "outcome", "fraud", "class", "response", "purchased", "approved"]
                matched_keyword = [c for c in candidates if any(kw in c.lower() for kw in target_keywords)]
                if matched_keyword:
                    selected_target = matched_keyword[0]
                    target_reason = f"Selected semantic outcome candidate '{selected_target}' matching target keywords"
                else:
                    # 3. Discrete binary/categorical candidates before high-cardinality continuous metrics
                    discrete_cands = [c for c in candidates if df[c].dropna().nunique() <= 10]
                    if discrete_cands:
                        selected_target = discrete_cands[0]
                        target_reason = f"Selected discrete candidate '{selected_target}'"
                    else:
                        selected_target = candidates[0]
                        target_reason = f"Selected first deterministic candidate '{selected_target}' among {candidates}"

    if selected_target is None or selected_target not in df.columns:
        return {
            "task_type": "none",
            "target_column": None,
            "reason": "No valid target candidate identified in dataset",
            "status": "unsupported",
        }

    # 2. Determine Task Type from target data
    s_target = df[selected_target].dropna()
    if len(s_target) == 0:
        return {
            "task_type": "none",
            "target_column": selected_target,
            "reason": f"Target column '{selected_target}' contains only missing values",
            "status": "invalid_target",
        }

    n_unique = int(s_target.nunique())
    meta = col_dict.get(selected_target, {})
    dtype_inferred = meta.get("dtype_inferred", "string")

    if n_unique <= 1:
        return {
            "task_type": "none",
            "target_column": selected_target,
            "reason": f"Target column '{selected_target}' is constant with only {n_unique} unique value",
            "status": "constant_target",
        }

    if dtype_inferred in ("category", "bool", "string") or (dtype_inferred == "int" and n_unique <= 10):
        task_type = "classification"
        subtype = "binary" if n_unique == 2 else "multiclass"
        reason = f"Target '{selected_target}' has {n_unique} discrete categories ({subtype} classification)"
    else:
        task_type = "regression"
        subtype = "continuous"
        reason = f"Target '{selected_target}' has {n_unique} continuous numeric values (regression)"

    return {
        "task_type": task_type,
        "subtype": subtype,
        "target_column": selected_target,
        "target_reason": target_reason,
        "reason": reason,
        "n_unique_target": n_unique,
        "status": "ready",
    }
