"""
agents/ml/leakage_detector.py
=============================
Data and target leakage prevention.
Enforces exclusion of target, identifiers, PII, high-correlation/high-MI features,
and semantic post-outcome artifacts.
"""

from __future__ import annotations

import re
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


POST_OUTCOME_PATTERNS = [
    r"cancel.*date",
    r"cancellation",
    r"discharge.*date",
    r"outcome.*note",
    r"resolved.*flag",
    r"resolution.*date",
    r"churn.*date",
    r"termination.*date",
    r"exit.*date",
]


def detect_and_exclude_leakage(
    df: pd.DataFrame,
    target_col: str,
    columns_info: list[dict[str, Any]] | None = None,
    leakage_threshold: float = 0.95,
) -> dict[str, Any]:
    """
    Perform thorough data leakage analysis and produce safe feature matrix columns.
    """
    col_dict = {c["name"]: c for c in (columns_info or [])}
    usable_features: list[str] = []
    excluded_features: list[dict[str, Any]] = []
    leakage_checks: list[dict[str, Any]] = []

    s_target = df[target_col]
    target_is_numeric = pd.api.types.is_numeric_dtype(s_target) and not pd.api.types.is_bool_dtype(s_target)

    for col in df.columns:
        if col == target_col:
            # 1. Direct Target Exclusion
            excluded_features.append({
                "column": col,
                "reason": "target_column",
                "exclusion_status": "excluded",
            })
            leakage_checks.append({
                "column": col,
                "detection_method": "target_identity",
                "specific_reason": "Target column cannot be included as a feature",
                "threshold_used": None,
                "measured_value": 1.0,
                "exclusion_status": "excluded",
            })
            continue

        meta = col_dict.get(col, {})
        s_feat = df[col]

        # 2. PII Exclusion
        if meta.get("is_pii", False):
            excluded_features.append({
                "column": col,
                "reason": "pii_protection",
                "exclusion_status": "excluded",
            })
            leakage_checks.append({
                "column": col,
                "detection_method": "pii_metadata",
                "specific_reason": f"Column contains sensitive PII ({meta.get('pii_type', 'pii')})",
                "threshold_used": None,
                "measured_value": None,
                "exclusion_status": "excluded",
            })
            continue

        # 3. Identifier Exclusion
        if meta.get("semantic_label") == "identifier":
            excluded_features.append({
                "column": col,
                "reason": "identifier_exclusion",
                "exclusion_status": "excluded",
            })
            leakage_checks.append({
                "column": col,
                "detection_method": "identifier_metadata",
                "specific_reason": "Identifier column has zero generalizable predictive value",
                "threshold_used": None,
                "measured_value": None,
                "exclusion_status": "excluded",
            })
            continue

        # 4. Semantic Post-Outcome Pattern Match
        is_post_outcome_name = any(re.search(pat, col.lower()) for pat in POST_OUTCOME_PATTERNS)
        if is_post_outcome_name:
            excluded_features.append({
                "column": col,
                "reason": "post_outcome_name_match",
                "exclusion_status": "excluded",
            })
            leakage_checks.append({
                "column": col,
                "detection_method": "name_pattern_heuristic",
                "specific_reason": f"Column name '{col}' matches known post-outcome temporal pattern",
                "threshold_used": None,
                "measured_value": None,
                "exclusion_status": "excluded",
            })
            continue

        # 5. Near-Perfect Correlation / Mutual Information Leakage Check
        is_leaky = False
        measured_metric = 0.0
        method_used = "none"

        try:
            if target_is_numeric and pd.api.types.is_numeric_dtype(s_feat) and not pd.api.types.is_bool_dtype(s_feat):
                method_used = "pearson_correlation"
                valid_mask = s_feat.notna() & s_target.notna()
                if valid_mask.sum() > 2 and s_feat[valid_mask].std() > 0 and s_target[valid_mask].std() > 0:
                    r_val = float(abs(np.corrcoef(s_feat[valid_mask], s_target[valid_mask])[0, 1]))
                    measured_metric = round(r_val, 4)
                    if r_val >= leakage_threshold:
                        is_leaky = True
            else:
                method_used = "normalized_mutual_info"
                valid_mask = s_feat.notna() & s_target.notna()
                if valid_mask.sum() > 2:
                    nmi_val = float(normalized_mutual_info_score(s_feat[valid_mask].astype(str), s_target[valid_mask].astype(str)))
                    measured_metric = round(nmi_val, 4)
                    if nmi_val >= leakage_threshold:
                        is_leaky = True
        except Exception:
            pass

        if is_leaky:
            excluded_features.append({
                "column": col,
                "reason": f"high_{method_used}_leakage",
                "exclusion_status": "excluded",
            })
            leakage_checks.append({
                "column": col,
                "detection_method": method_used,
                "specific_reason": f"Measured metric {measured_metric} exceeds leakage threshold of {leakage_threshold}",
                "threshold_used": leakage_threshold,
                "measured_value": measured_metric,
                "exclusion_status": "excluded",
            })
            continue

        # Feature passed all leakage checks
        usable_features.append(col)
        leakage_checks.append({
            "column": col,
            "detection_method": method_used if method_used != "none" else "standard_check",
            "specific_reason": "Feature passed all leakage and safety checks",
            "threshold_used": leakage_threshold if method_used != "none" else None,
            "measured_value": measured_metric if method_used != "none" else None,
            "exclusion_status": "retained",
        })

    return {
        "usable_features": usable_features,
        "excluded_features": excluded_features,
        "leakage_checks": leakage_checks,
        "has_sufficient_features": len(usable_features) > 0,
    }
