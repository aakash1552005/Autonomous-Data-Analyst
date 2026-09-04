"""
agents/report/data_extractor.py
===============================
Safe, non-mutating data extraction and formatting for Report Agent.
Extracts structured facts from the Dataset Intelligence Object (DIO)
without reading raw DataFrame rows or exposing PII / identifier values.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any


def extract_report_payload(dio: Any) -> dict[str, Any]:
    """
    Extract a sanitized, formatted data payload from DIO for report generation.
    Strictly read-only: does NOT mutate any DIO field.
    
    Guarantees:
    - Never extracts raw DataFrame rows.
    - Masks and flags all PII and identifier columns.
    - Gracefully handles missing upstream sections or incomplete analysis.
    """
    # 1. Core Identification & Metadata
    file_name = str(dio.get("file_name", "") or "Dataset")
    dataset_id = str(dio.get("dataset_id", "") or "unknown_dataset")
    dataset_hash = str(dio.get("dataset_hash", "") or "")
    schema_version = str(dio.get("schema_version", "") or "1.0.0")
    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 2. Ingestion Profile
    ingestion_data = dio.get("ingestion", {}) or {}
    n_rows = int(ingestion_data.get("n_rows", 0))
    n_columns = int(ingestion_data.get("n_columns", 0))
    file_type = str(ingestion_data.get("file_type", "csv")).upper()
    encoding = str(ingestion_data.get("encoding", "utf-8"))

    # Fallback for rows/cols if ingestion section was empty
    columns_raw = dio.get("columns", []) or []
    if n_columns == 0:
        n_columns = len(columns_raw)

    # 3. Domain Classification
    domain_data = dio.get("domain_guess", {}) or {}
    domain = str(domain_data.get("domain", "generic")).capitalize()
    domain_confidence = float(domain_data.get("confidence", 0.0))

    # 4. Data Quality
    quality_data = dio.get("quality", {}) or {}
    quality_score = int(quality_data.get("score", 0))
    quality_issues = list(quality_data.get("issues", []) or [])

    # 5. Column Architecture & PII Safe Catalog
    # Never render actual data values for any column, especially PII/identifiers
    columns_catalog: list[dict[str, Any]] = []
    pii_columns_detected: list[str] = []

    for col in columns_raw:
        c_name = str(col.get("name", "unnamed"))
        c_type = str(col.get("logical_type") or col.get("dtype") or "unknown")
        c_nulls = int(col.get("null_count", 0))
        c_null_pct = float(col.get("null_pct", 0.0))
        c_unique = int(col.get("unique_count") or col.get("cardinality") or 0)
        is_pii = bool(col.get("is_pii", False) or col.get("pii_type"))
        is_id = bool(col.get("is_identifier", False))
        sem_label = str(col.get("semantic_label", "generic_feature"))

        if is_pii:
            pii_columns_detected.append(c_name)
            status_badge = f"[Protected PII: {col.get('pii_type', 'sensitive')}]"
        elif is_id:
            status_badge = "[Protected Identifier]"
        else:
            status_badge = "Standard"

        columns_catalog.append({
            "name": c_name,
            "logical_type": c_type,
            "null_count": c_nulls,
            "null_pct": round(c_null_pct, 2),
            "unique_count": c_unique,
            "is_pii": is_pii,
            "is_identifier": is_id,
            "semantic_label": sem_label,
            "status_badge": status_badge,
        })

    # 6. Cleaning Log Summary
    cleaning_log = dio.get("cleaning_log", []) or []
    duplicates_removed = 0
    imputation_count = 0
    cleaning_operations: list[str] = []

    for entry in cleaning_log:
        action = entry.get("action", "")
        col_target = entry.get("column") or entry.get("details", "")
        if action == "drop_duplicates" or "duplicate" in action.lower():
            count = entry.get("count") or entry.get("removed_rows_count", 0)
            duplicates_removed += int(count)
            cleaning_operations.append(f"Removed {count} exact duplicate rows.")
        elif "imput" in action.lower():
            imputation_count += 1
            method = entry.get("method", "replacement")
            val = entry.get("replacement_value")
            # Ensure replacement value doesn't expose PII if column is PII
            if any(col.get("name") == entry.get("column") and (col.get("is_pii") or col.get("is_identifier")) for col in columns_raw):
                val = "[MASKED_PII]"
            cleaning_operations.append(f"Imputed column '{entry.get('column', '')}' via {method} (val={val}).")
        elif action == "coerce_type":
            cleaning_operations.append(f"Coerced '{entry.get('column', '')}' to {entry.get('target_type', '')}.")
        elif action:
            cleaning_operations.append(f"{action.replace('_', ' ').capitalize()}: {col_target}")

    # 7. Exploratory Data Analysis (EDA)
    eda_data = dio.get("eda", {}) or {}
    raw_stats = eda_data.get("summary_stats", {}) or {}
    summary_stats: list[dict[str, Any]] = []

    for col_name, stats in raw_stats.items():
        if not isinstance(stats, dict):
            continue
        # Check if column is PII or identifier — do not display min/max if it could expose sensitive bounds
        is_sensitive = any(c["name"] == col_name and (c["is_pii"] or c["is_identifier"]) for c in columns_catalog)
        if is_sensitive:
            continue

        s_entry = {
            "column": col_name,
            "mean": f"{float(stats['mean']):.2f}" if stats.get("mean") is not None else "N/A",
            "std": f"{float(stats['std']):.2f}" if stats.get("std") is not None else "N/A",
            "min": f"{float(stats['min']):.2f}" if stats.get("min") is not None else "N/A",
            "median": f"{float(stats['median']):.2f}" if stats.get("median") is not None else "N/A",
            "max": f"{float(stats['max']):.2f}" if stats.get("max") is not None else "N/A",
        }
        summary_stats.append(s_entry)

    # Top correlations
    correlations_data = eda_data.get("correlations", {}) or {}
    top_corrs: list[dict[str, Any]] = []
    raw_corr_pairs = correlations_data.get("top_correlations", []) or correlations_data.get("top_features", []) or []
    
    if isinstance(raw_corr_pairs, list):
        for pair in raw_corr_pairs:
            if isinstance(pair, dict):
                col_a = pair.get("feature_a") or pair.get("column_1") or pair.get("col1")
                col_b = pair.get("feature_b") or pair.get("column_2") or pair.get("col2")
                r_val = pair.get("pearson_correlation") or pair.get("correlation") or pair.get("r")
                if col_a and col_b and r_val is not None:
                    top_corrs.append({
                        "feature_a": str(col_a),
                        "feature_b": str(col_b),
                        "correlation": round(float(r_val), 3),
                    })

    # Available chart paths (verified valid image files)
    chart_paths: list[str] = []
    candidate_charts = (dio.get("artifacts", {}) or {}).get("chart_paths", []) or eda_data.get("charts", []) or []
    for cp in candidate_charts:
        p = Path(cp)
        if p.is_file() and p.stat().st_size > 0:
            try:
                from PIL import Image as PILImage
                with PILImage.open(p) as im:
                    im.verify()
                chart_paths.append(str(p))
            except Exception:
                pass

    # 8. Machine Learning Results
    ml_data = dio.get("ml", {}) or {}
    ml_status = str(ml_data.get("status", "none")).lower()
    ml_task_type = str(ml_data.get("task_type", "none")).capitalize()
    ml_target = ml_data.get("target_column")
    ml_best_model = ml_data.get("best_model") or (ml_data.get("selected_model", {}).get("model_name") if isinstance(ml_data.get("selected_model"), dict) else None)
    ml_reason = ml_data.get("reason", "")
    ml_warnings = list(ml_data.get("verification_warnings", []) or [])
    
    ml_metrics: dict[str, str] = {}
    raw_metrics = ml_data.get("metrics", {}) or {}
    for m_name, m_val in raw_metrics.items():
        if isinstance(m_val, (int, float)):
            ml_metrics[m_name.replace("_", " ").title()] = f"{float(m_val):.4f}"
        elif isinstance(m_val, str):
            ml_metrics[m_name.replace("_", " ").title()] = m_val

    feature_importances: list[tuple[str, float]] = []
    raw_fi = ml_data.get("feature_importance", {}) or {}
    if isinstance(raw_fi, dict):
        sorted_fi = sorted(raw_fi.items(), key=lambda x: abs(float(x[1])), reverse=True)
        feature_importances = [(k, round(float(v), 4)) for k, v in sorted_fi[:8]]

    # 9. Key Business Insights
    insights_raw = dio.get("insights", []) or []
    insights_list: list[dict[str, Any]] = []
    recommendations_list: list[str] = []

    for ins in insights_raw:
        if isinstance(ins, dict):
            ins_id = ins.get("id", "ins_000")
            category = str(ins.get("category", "General")).title()
            text = ins.get("text", "")
            conf = float(ins.get("confidence", 0.90))
            evidence = ins.get("evidence", "")
            rec = ins.get("recommendation", "")
            grounded_nums = ins.get("grounded_numbers", [])

            insights_list.append({
                "id": ins_id,
                "category": category,
                "text": text,
                "confidence": round(conf, 2),
                "evidence": evidence,
                "recommendation": rec,
                "grounded_numbers": grounded_nums,
            })
            if rec and rec not in recommendations_list:
                recommendations_list.append(rec)
        elif isinstance(ins, str):
            insights_list.append({
                "id": "ins_legacy",
                "category": "General",
                "text": ins,
                "confidence": 0.85,
                "evidence": "DIO statistical analysis",
                "recommendation": "",
                "grounded_numbers": [],
            })

    # 10. Limitations and Warnings
    limitations: list[str] = []
    for issue in quality_issues:
        limitations.append(f"Data Quality Issue: {issue}")

    # Check date ambiguities
    date_cols = dio.get("date_columns", []) or []
    for dc in date_cols:
        if dc.get("status") == "ambiguous" or dc.get("needs_confirmation"):
            limitations.append(f"Temporal Ambiguity: Date column '{dc.get('name')}' contained ambiguous formats and was preserved in raw format.")

    if ml_status in ("insufficient_data", "unsupported", "leakage_detected"):
        limitations.append(f"Machine Learning Limitation: {ml_reason or f'Status {ml_status}'}")

    if not insights_list:
        limitations.append("Business Insights: Limited numerical signal was available in the dataset to extract high-confidence business insights.")

    return {
        "metadata": {
            "dataset_id": dataset_id,
            "file_name": file_name,
            "dataset_hash": dataset_hash,
            "schema_version": schema_version,
            "generated_at": generated_at,
        },
        "ingestion": {
            "n_rows": n_rows,
            "n_columns": n_columns,
            "file_type": file_type,
            "encoding": encoding,
        },
        "domain": {
            "name": domain,
            "confidence": round(domain_confidence, 2),
        },
        "quality": {
            "score": quality_score,
            "issues": quality_issues,
        },
        "columns": columns_catalog,
        "pii_columns": pii_columns_detected,
        "cleaning": {
            "duplicates_removed": duplicates_removed,
            "imputation_count": imputation_count,
            "operations": cleaning_operations,
        },
        "eda": {
            "summary_stats": summary_stats,
            "top_correlations": top_corrs,
            "chart_paths": chart_paths,
        },
        "ml": {
            "status": ml_status,
            "task_type": ml_task_type,
            "target_column": ml_target,
            "best_model": ml_best_model,
            "metrics": ml_metrics,
            "feature_importances": feature_importances,
            "reason": ml_reason,
            "warnings": ml_warnings,
        },
        "insights": insights_list,
        "recommendations": recommendations_list,
        "limitations": limitations,
        "progress": dict(dio.get("progress", {}) or {}),
    }
