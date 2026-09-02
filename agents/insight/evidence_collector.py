"""
agents/insight/evidence_collector.py
====================================
Collects, structures, and indexes grounded numerical facts from the Dataset Intelligence Object (DIO).
Extracts verifiable floats, integers, and proportions across quality, summary_stats, correlations, and ml sections.
"""

from __future__ import annotations

from typing import Any


def extract_all_dio_numbers(dio: dict[str, Any]) -> tuple[set[int], set[float]]:
    """
    Extract grounded integers and floats separately to prevent cross-type grounding errors
    (e.g. preventing a 99.9% hallucinated rate from matching 100 integer rows).
    """
    grounded_ints: set[int] = set()
    grounded_floats: set[float] = set()

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

    # 3. EDA Summary Statistics
    eda = dio.get("eda", {})
    summary_stats = eda.get("summary_stats", {})

    numeric_stats = summary_stats.get("numeric", {})
    for col_name, stats in numeric_stats.items():
        if isinstance(stats, dict):
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    add_val(v)

    categorical_stats = summary_stats.get("categorical", {})
    for col_name, stats in categorical_stats.items():
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

    # 4. EDA Correlations (Exclude self-correlation diagonal 1.0)
    correlations = eda.get("correlations", {})
    for matrix_key in ("pearson", "spearman"):
        matrix = correlations.get(matrix_key, {})
        if isinstance(matrix, dict):
            for c1, row_vals in matrix.items():
                if isinstance(row_vals, dict):
                    for c2, val in row_vals.items():
                        if c1 != c2 and isinstance(val, (int, float)):
                            add_val(val)

    top_corrs = correlations.get("top_correlations", [])
    if isinstance(top_corrs, list):
        for item in top_corrs:
            if isinstance(item, dict):
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
                if isinstance(imp, (int, float)):
                    add_val(imp)

    return grounded_ints, grounded_floats


def collect_evidence_summary(dio: dict[str, Any]) -> dict[str, Any]:
    """
    Format a clean, structured dictionary of high-level evidence for prompting the LLM.
    Strictly excludes raw rows, PII, and unnecessary low-level details.
    """
    ingestion = dio.get("ingestion", {})
    quality = dio.get("quality", {})
    domain = dio.get("domain_guess", {}).get("domain", "generic")

    eda = dio.get("eda", {})
    summary_stats = eda.get("summary_stats", {})
    correlations = eda.get("correlations", {})
    top_corrs = correlations.get("top_correlations", [])[:5]

    ml = dio.get("ml", {})

    evidence = {
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

    # Numeric summaries
    for col, stats in summary_stats.get("numeric", {}).items():
        if isinstance(stats, dict):
            evidence["numeric_summaries"][col] = {
                "mean": stats.get("mean"),
                "median": stats.get("median"),
                "std": stats.get("std"),
                "min": stats.get("min"),
                "max": stats.get("max"),
            }

    # Categorical summaries
    for col, stats in summary_stats.get("categorical", {}).items():
        if isinstance(stats, dict):
            evidence["categorical_summaries"][col] = {
                "unique_count": stats.get("unique_count"),
                "top_categories": stats.get("top_categories"),
            }

    # Top correlations
    for c in top_corrs:
        evidence["correlations"].append({
            "col1": c.get("col1"),
            "col2": c.get("col2"),
            "pearson": c.get("pearson"),
            "spearman": c.get("spearman"),
        })

    # ML results if trained
    if ml.get("status") == "trained":
        evidence["machine_learning"] = {
            "status": "trained",
            "task_type": ml.get("task_type"),
            "target_column": ml.get("target_column"),
            "selected_model": ml.get("selected_model"),
            "metrics": ml.get("metrics"),
            "top_features": list(ml.get("feature_importance", {}).items())[:5],
        }
    else:
        evidence["machine_learning"] = {
            "status": ml.get("status", "skipped"),
            "reason": ml.get("reason", "Not trained / Insufficient data"),
        }

    return evidence
