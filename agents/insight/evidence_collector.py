"""
agents/insight/evidence_collector.py
====================================
Collects, structures, and indexes grounded numerical facts from the Dataset Intelligence Object (DIO).
Extracts verifiable floats, integers, and proportions across quality, summary_stats, correlations, and ml sections.
"""

from __future__ import annotations

import re
from typing import Any


SENSITIVE_SEMANTIC_LABELS = {
    "identifier",
    "person_name",
    "email",
    "phone",
    "credit_card",
    "ssn_national_id",
    "address",
}

PII_COL_NAME_REGEX = re.compile(
    r"(?i)^(full_?name|first_?name|last_?name|customer_?name|patient_?name|employee_?name|user_?name|client_?name|fname|lname|name|email|e_mail|mail|phone|mobile|cell|telephone|contact_num|contact_no|ssn|social_security|national_id|aadhaar|pan_card|tax_id|credit_card|card_num|cc_num|debit_card|card_no|address|street|addr|street_address|home_address|postal_code|zip_code|zipcode|customer_?id|user_?id|client_?id|patient_?id|account_?id|employee_?id|ssn_?id)$"
)


def get_excluded_pii_columns(dio: dict[str, Any]) -> set[str]:
    """
    Identify columns that must be strictly excluded from LLM evidence due to PII,
    identifier status, or sensitive identity information.
    Inspects dio["columns"] metadata and enforces defense-in-depth header patterns.
    """
    excluded: set[str] = set()
    columns_info = dio.get("columns", [])

    for c in columns_info:
        if not isinstance(c, dict):
            continue
        c_name = str(c.get("name", "")).strip()
        if not c_name:
            continue

        # 1. Flagged as is_pii == True
        if c.get("is_pii", False):
            excluded.add(c_name)
            continue

        # 2. Identified with sensitive semantic labels
        sem_label = str(c.get("semantic_label", "")).lower().strip()
        if sem_label in SENSITIVE_SEMANTIC_LABELS:
            excluded.add(c_name)
            continue

        # 3. Specific pii_type assigned
        pii_type = str(c.get("pii_type", "")).lower().strip()
        if pii_type and pii_type != "none":
            excluded.add(c_name)
            continue

        # 4. Defense-in-depth column name pattern
        if PII_COL_NAME_REGEX.match(c_name):
            excluded.add(c_name)

    return excluded


def extract_all_dio_numbers(dio: dict[str, Any]) -> tuple[set[int], set[float]]:
    """
    Extract grounded integers and floats separately to prevent cross-type grounding errors
    (e.g. preventing a 99.9% hallucinated rate from matching 100 integer rows).
    Strictly excludes numbers derived from PII and identifier columns.
    """
    grounded_ints: set[int] = set()
    grounded_floats: set[float] = set()
    excluded_cols = get_excluded_pii_columns(dio)

    def add_val(val: Any) -> None:
        if val is None or isinstance(val, bool):
            return
        if isinstance(val, int):
            grounded_ints.add(val)
        elif isinstance(val, float):
            if val.is_integer():
                grounded_ints.add(int(val))
            else:
                grounded_floats.add(val)

    # 1. Ingestion metadata
    ingestion = dio.get("ingestion", {})
    add_val(ingestion.get("n_rows"))
    add_val(ingestion.get("n_columns"))

    # 2. Quality assessment
    quality = dio.get("quality", {})
    add_val(quality.get("score"))

    # 3. EDA Summary Statistics (excluding PII / identifier columns)
    eda = dio.get("eda", {})
    summary_stats = eda.get("summary_stats", {})

    numeric_stats = summary_stats.get("numeric", {})
    for col_name, stats in numeric_stats.items():
        if col_name in excluded_cols:
            continue
        if isinstance(stats, dict):
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    add_val(v)

    categorical_stats = summary_stats.get("categorical", {})
    for col_name, stats in categorical_stats.items():
        if col_name in excluded_cols:
            continue
        if isinstance(stats, dict):
            add_val(stats.get("unique_count"))
            top_counts = stats.get("top_categories", {})
            if isinstance(top_counts, dict):
                for cat_val, count in top_counts.items():
                    add_val(count)
                    n_rows = ingestion.get("n_rows", 0)
                    if n_rows and isinstance(count, (int, float)):
                        pct = (count / n_rows) * 100.0
                        grounded_floats.add(round(pct, 1))
                        grounded_floats.add(round(pct, 2))

    # Flat summary_stats fallback
    for col_name, stats in summary_stats.items():
        if col_name in ("numeric", "categorical") or not isinstance(stats, dict):
            continue
        if col_name in excluded_cols:
            continue
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                add_val(v)

    # 4. EDA Correlations (Exclude self-correlation diagonal 1.0 and PII columns)
    correlations = eda.get("correlations", {})
    for matrix_key in ("pearson", "spearman"):
        matrix = correlations.get(matrix_key, {})
        if isinstance(matrix, dict):
            for c1, row_vals in matrix.items():
                if c1 in excluded_cols:
                    continue
                if isinstance(row_vals, dict):
                    for c2, val in row_vals.items():
                        if c2 in excluded_cols:
                            continue
                        if c1 != c2 and isinstance(val, (int, float)):
                            add_val(val)

    top_corrs = correlations.get("top_correlations", [])
    if isinstance(top_corrs, list):
        for item in top_corrs:
            if isinstance(item, dict):
                c1 = item.get("col1")
                c2 = item.get("col2")
                if c1 in excluded_cols or c2 in excluded_cols:
                    continue
                if isinstance(item.get("pearson"), (int, float)):
                    add_val(item.get("pearson"))
                if isinstance(item.get("spearman"), (int, float)):
                    add_val(item.get("spearman"))

    # 5. Machine Learning Results
    ml = dio.get("ml", {})
    if ml.get("status") == "trained":
        split = ml.get("split", {})
        add_val(split.get("train_rows"))
        add_val(split.get("test_rows"))
        add_val(split.get("train_pct"))
        add_val(split.get("test_pct"))

        metrics = ml.get("metrics", {})
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    add_val(v)

        feat_imp = ml.get("feature_importance", {})
        if isinstance(feat_imp, dict):
            for feat, imp in feat_imp.items():
                if feat not in excluded_cols and isinstance(imp, (int, float)):
                    add_val(imp)

    return grounded_ints, grounded_floats


def collect_evidence_summary(dio: dict[str, Any]) -> dict[str, Any]:
    """
    Format a clean, structured dictionary of high-level evidence for prompting the LLM.
    Strictly excludes raw rows, PII, and identifier details.
    """
    ingestion = dio.get("ingestion", {})
    quality = dio.get("quality", {})
    domain = dio.get("domain_guess", {}).get("domain", "generic")

    eda = dio.get("eda", {})
    summary_stats = eda.get("summary_stats", {})
    correlations = eda.get("correlations", {})
    top_corrs = correlations.get("top_correlations", [])[:5]

    ml = dio.get("ml", {})
    excluded_cols = get_excluded_pii_columns(dio)

    evidence: dict[str, Any] = {
        "dataset_profile": {
            "domain": domain,
            "rows": ingestion.get("n_rows", 0),
            "columns": ingestion.get("n_columns", 0),
            "quality_score": quality.get("score", 100),
            "quality_issues": quality.get("issues", []),
        },
        "numeric_summaries": {},
        "categorical_summaries": {},
        "correlations": [],
        "machine_learning": {},
    }

    # 1. Numeric summaries (nested format)
    for col, stats in summary_stats.get("numeric", {}).items():
        if col in excluded_cols:
            continue
        if isinstance(stats, dict):
            evidence["numeric_summaries"][col] = {
                "mean": stats.get("mean"),
                "median": stats.get("median"),
                "std": stats.get("std"),
                "min": stats.get("min"),
                "max": stats.get("max"),
            }

    # 2. Categorical summaries (nested format)
    for col, stats in summary_stats.get("categorical", {}).items():
        if col in excluded_cols:
            continue
        if isinstance(stats, dict):
            evidence["categorical_summaries"][col] = {
                "unique_count": stats.get("unique_count"),
                "top_categories": stats.get("top_categories"),
            }

    # 3. Flat summary_stats format support (real EDA pipeline structure)
    for col, stats in summary_stats.items():
        if col in ("numeric", "categorical") or not isinstance(stats, dict):
            continue
        if col in excluded_cols:
            continue
        col_type = stats.get("type")
        if col_type == "numeric" and col not in evidence["numeric_summaries"]:
            evidence["numeric_summaries"][col] = {
                "mean": stats.get("mean"),
                "median": stats.get("median"),
                "std": stats.get("std"),
                "min": stats.get("min"),
                "max": stats.get("max"),
            }
        elif col_type == "categorical" and col not in evidence["categorical_summaries"]:
            evidence["categorical_summaries"][col] = {
                "unique_count": stats.get("unique_count"),
                "top_categories": stats.get("top_categories"),
            }

    # 4. Top correlations (exclude PII/identifier columns)
    for c in top_corrs:
        c1 = c.get("col1")
        c2 = c.get("col2")
        if c1 in excluded_cols or c2 in excluded_cols:
            continue
        evidence["correlations"].append({
            "col1": c1,
            "col2": c2,
            "pearson": c.get("pearson"),
            "spearman": c.get("spearman"),
        })

    # 5. ML results if trained (exclude PII columns from features)
    if ml.get("status") == "trained":
        filtered_top_features = [
            (feat, imp) for feat, imp in ml.get("feature_importance", {}).items()
            if feat not in excluded_cols
        ][:5]
        evidence["machine_learning"] = {
            "status": "trained",
            "task_type": ml.get("task_type"),
            "target_column": ml.get("target_column"),
            "selected_model": ml.get("selected_model"),
            "metrics": ml.get("metrics"),
            "top_features": filtered_top_features,
        }
    else:
        evidence["machine_learning"] = {
            "status": ml.get("status", "skipped"),
            "reason": ml.get("reason", "Not trained / Insufficient data"),
        }

    return evidence
