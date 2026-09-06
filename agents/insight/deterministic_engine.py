"""
agents/insight/deterministic_engine.py
======================================
Deterministic, rule-based insight generation engine.
Produces 100% reproducible, grounded insights from computed DIO statistics
without requiring an LLM. Respects the evidence availability floor (never fabricates).
"""

from __future__ import annotations

from typing import Any
from agents.insight.hallucination_guard import extract_numbers_from_text
from agents.insight.evidence_collector import get_excluded_pii_columns


def generate_deterministic_insights(
    dio: dict[str, Any],
    max_insights: int = 6,
) -> list[dict[str, Any]]:
    """
    Generate structured insights deterministically from DIO evidence.
    Prioritizes highest-signal statistical findings across ML, Correlations,
    Distributions, and Data Quality.
    Strictly excludes any PII or identifier columns.
    """
    candidates: list[dict[str, Any]] = []
    excluded_cols = get_excluded_pii_columns(dio)

    ingestion = dio.get("ingestion", {})
    n_rows = ingestion.get("n_rows", 0)
    n_cols = ingestion.get("n_columns", 0)
    quality = dio.get("quality", {})
    q_score = quality.get("score", 100)
    domain = dio.get("domain_guess", {}).get("domain", "generic")

    eda = dio.get("eda", {})
    summary_stats = eda.get("summary_stats", {})
    correlations = eda.get("correlations", {})
    ml = dio.get("ml", {})

    # 1. Machine Learning Performance & Key Drivers (Highest Priority if trained)
    if ml.get("status") == "trained":
        task = ml.get("task_type", "classification")
        target = ml.get("target_column", "target")
        model = ml.get("selected_model", "model")
        metrics = ml.get("metrics", {})
        feat_imp = ml.get("feature_importance", {})

        if task == "classification":
            f1 = metrics.get("f1", 0.0)
            acc = metrics.get("accuracy", 0.0)
            candidates.append({
                "category": "machine_learning",
                "text": f"Machine learning classification model ({model}) achieved an F1 score of {f1:.2f} (Accuracy: {acc * 100.0:.1f}%) predicting target column '{target}'.",
                "confidence": 0.85,
                "evidence": "dio.ml.metrics",
                "recommendation": f"Deploy the {model} model pipeline for automated prediction on new incoming records.",
            })
        else:
            rmse = metrics.get("rmse", 0.0)
            r2 = metrics.get("r2", 0.0)
            candidates.append({
                "category": "machine_learning",
                "text": f"Machine learning regression model ({model}) achieved an RMSE of {rmse:.2f} with an R² score of {r2:.2f} predicting target column '{target}'.",
                "confidence": 0.85,
                "evidence": "dio.ml.metrics",
                "recommendation": f"Use {model} baseline forecasts for resource planning and variance monitoring.",
            })

        if feat_imp:
            top_feat, top_val = list(feat_imp.items())[0]
            candidates.append({
                "category": "machine_learning",
                "text": f"Primary predictive driver is '{top_feat}' with a relative feature importance of {top_val * 100.0:.1f}%.",
                "confidence": 0.80,
                "evidence": f"dio.ml.feature_importance.{top_feat}",
                "recommendation": f"Focus operational interventions primarily on managing variations in '{top_feat}'.",
            })

    # 2. Strongest Correlation Insights
    top_corrs = correlations.get("top_correlations", [])
    if top_corrs:
        for corr in top_corrs:
            c1 = corr.get("col1")
            c2 = corr.get("col2")
            if c1 in excluded_cols or c2 in excluded_cols:
                continue
            p_val = corr.get("pearson")
            if p_val is not None and abs(p_val) >= 0.20:
                direction = "positive" if p_val > 0 else "negative"
                strength = "strong" if abs(p_val) >= 0.70 else ("moderate" if abs(p_val) >= 0.40 else "mild")
                candidates.append({
                    "category": "correlation",
                    "text": f"Detected a {strength} {direction} linear correlation of {p_val:.2f} between '{c1}' and '{c2}'.",
                    "confidence": 0.90,
                    "evidence": f"dio.eda.correlations.pearson.{c1}.{c2}",
                    "recommendation": f"Evaluate potential causal dependencies between '{c1}' and '{c2}' in operational workflows.",
                })
            if len([c for c in candidates if c["category"] == "correlation"]) >= 2:
                break

    # 3. Numeric Distribution Insights (Mean, Median, Dispersion)
    numeric_stats = summary_stats.get("numeric", {})
    if numeric_stats:
        # Pick top 2 numeric non-PII columns with highest variance/mean
        sorted_num = sorted(
            [item for item in numeric_stats.items() if item[0] not in excluded_cols],
            key=lambda item: item[1].get("std", 0.0) if isinstance(item[1], dict) else 0.0,
            reverse=True,
        )
        for col_name, stats in sorted_num[:2]:
            if isinstance(stats, dict):
                mean_val = stats.get("mean", 0.0)
                med_val = stats.get("median", 0.0)
                min_val = stats.get("min", 0.0)
                max_val = stats.get("max", 0.0)
                candidates.append({
                    "category": "distribution",
                    "text": f"Column '{col_name}' averages {mean_val:.2f} (median: {med_val:.2f}) spanning a range from {min_val:.2f} to {max_val:.2f}.",
                    "confidence": 0.95,
                    "evidence": f"dio.eda.summary_stats.numeric.{col_name}",
                    "recommendation": f"Monitor distributions of '{col_name}' to establish threshold alerts for anomalous deviations.",
                })

    # 4. Categorical Distribution Insights
    categorical_stats = summary_stats.get("categorical", {})
    if categorical_stats:
        valid_cats = [item for item in categorical_stats.items() if item[0] not in excluded_cols]
        for col_name, stats in valid_cats[:2]:
            if isinstance(stats, dict):
                u_cnt = stats.get("unique_count", 0)
                top_cats = stats.get("top_categories", {})
                if top_cats and n_rows > 0:
                    top_cat, count = list(top_cats.items())[0]
                    cat_pct = (count / n_rows) * 100.0
                    candidates.append({
                        "category": "distribution",
                        "text": f"Column '{col_name}' has {u_cnt} unique categories, led by '{top_cat}' with {count} occurrences ({cat_pct:.1f}% of total).",
                        "confidence": 0.95,
                        "evidence": f"dio.eda.summary_stats.categorical.{col_name}",
                        "recommendation": f"Segment reporting by '{col_name}' to tailor strategies to the dominant '{top_cat}' segment.",
                    })

    # 5. Data Quality & Profile Overview Insight
    if n_rows > 0 and n_cols > 0:
        candidates.append({
            "category": "quality",
            "text": f"Analyzed dataset of {n_rows} rows and {n_cols} columns with an overall data quality score of {q_score}/100.",
            "confidence": 1.0,
            "evidence": "dio.quality.score",
            "recommendation": "Maintain standardized data ingestion pipelines to preserve data quality and prevent schema drift.",
        })

    # Deduplicate and cap to max_insights
    unique_candidates: list[dict[str, Any]] = []
    seen_texts: set[str] = set()

    for c in candidates:
        if c["text"] not in seen_texts:
            seen_texts.add(c["text"])
            c["grounded_numbers"] = extract_numbers_from_text(c["text"])
            unique_candidates.append(c)
        if len(unique_candidates) >= max_insights:
            break

    # Attach deterministic IDs
    for idx, c in enumerate(unique_candidates, start=1):
        c["id"] = f"ins_{idx:03d}"

    return unique_candidates
