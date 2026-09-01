"""
agents/eda/chart_generator.py
=============================
Deterministic Plotly + Kaleido chart generator for Exploratory Data Analysis.
Enforces strict rule-based chart selection (ZERO LLM), configurable max chart limits,
graceful degradation on narrow datasets, and static PNG artifact rendering.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Default visual theme
PLOTLY_THEME = "plotly_white"


def sanitize_filename(name: str) -> str:
    """Clean chart identifier for filesystem safety."""
    clean = re.sub(r"[^\w\-]", "_", name.strip().lower())
    return clean.strip("_") or "chart"


def select_charts_deterministically(
    df: pd.DataFrame,
    columns_info: list[dict[str, Any]] | None = None,
    date_columns_info: list[dict[str, Any]] | None = None,
    correlations_info: dict[str, Any] | None = None,
    max_charts: int = 6,
) -> list[dict[str, Any]]:
    """
    Apply the fixed deterministic chart-selection rule table.
    Returns ordered list of chart generation plans up to `max_charts`.
    """
    col_type_map = {c["name"]: c.get("dtype_inferred", "string") for c in (columns_info or [])}
    pii_cols = {c["name"] for c in (columns_info or []) if c.get("is_pii", False)}
    identifier_cols = {c["name"] for c in (columns_info or []) if c.get("semantic_label") == "identifier"}

    # Ambiguous date columns requiring confirmation must NEVER become time-series axes
    date_cols = [
        d["column"]
        for d in (date_columns_info or [])
        if d["column"] in df.columns and not d.get("needs_user_confirmation", False)
    ]

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        if col in pii_cols or col in identifier_cols:
            continue  # Exclude PII and identifiers from visualization charts

        col_type = col_type_map.get(col, "string")
        if (col_type in ("int", "float") or pd.api.types.is_numeric_dtype(df[col])) and not pd.api.types.is_bool_dtype(df[col]):
            clean_s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(clean_s) > 1 and clean_s.std() > 0:
                numeric_cols.append(col)
        elif col not in date_cols:
            n_unique = df[col].dropna().nunique()
            if 1 <= n_unique <= 15:
                categorical_cols.append(col)

    chart_plans: list[dict[str, Any]] = []

    # Rule 1: Correlation Heatmap (Requires >= 3 numeric columns)
    if len(numeric_cols) >= 3 and len(chart_plans) < max_charts:
        chart_plans.append({
            "chart_id": "01_correlation_heatmap",
            "chart_type": "correlation_heatmap",
            "title": "Feature Correlation Heatmap",
            "columns": numeric_cols,
        })

    # Rule 2: Time Series / Trend over Time (Requires date column + numeric column)
    if date_cols and numeric_cols and len(chart_plans) < max_charts:
        primary_date = date_cols[0]
        primary_num = numeric_cols[0]
        chart_plans.append({
            "chart_id": f"02_trend_{sanitize_filename(primary_num)}_over_time",
            "chart_type": "line_trend",
            "title": f"Trend of {primary_num} Over Time",
            "x_col": primary_date,
            "y_col": primary_num,
        })

    # Rule 3: Scatter Plot (Top correlated numeric pair, or first two numeric columns)
    if len(numeric_cols) >= 2 and len(chart_plans) < max_charts:
        top_pairs = (correlations_info or {}).get("top_correlations", [])
        if top_pairs:
            x_col = top_pairs[0]["col1"]
            y_col = top_pairs[0]["col2"]
        else:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]

        chart_plans.append({
            "chart_id": f"03_scatter_{sanitize_filename(x_col)}_vs_{sanitize_filename(y_col)}",
            "chart_type": "scatter",
            "title": f"Relationship: {x_col} vs {y_col}",
            "x_col": x_col,
            "y_col": y_col,
        })

    # Rule 4: Histograms / Numeric Distributions
    for num_col in numeric_cols:
        if len(chart_plans) >= max_charts:
            break
        chart_plans.append({
            "chart_id": f"04_distribution_{sanitize_filename(num_col)}",
            "chart_type": "histogram",
            "title": f"Distribution of {num_col}",
            "column": num_col,
        })

    # Rule 5: Categorical Frequency Bar Charts (<= 15 unique values)
    for cat_col in categorical_cols:
        if len(chart_plans) >= max_charts:
            break
        chart_plans.append({
            "chart_id": f"05_bar_{sanitize_filename(cat_col)}_counts",
            "chart_type": "categorical_bar",
            "title": f"Frequency Breakdown: {cat_col}",
            "column": cat_col,
        })

    # Rule 6: Box Plots (Numeric Spread)
    if numeric_cols and len(chart_plans) < max_charts:
        box_num = numeric_cols[0]
        box_cat = categorical_cols[0] if categorical_cols else None
        chart_plans.append({
            "chart_id": f"06_boxplot_{sanitize_filename(box_num)}",
            "chart_type": "boxplot",
            "title": f"Boxplot of {box_num}" + (f" by {box_cat}" if box_cat else ""),
            "y_col": box_num,
            "x_col": box_cat,
        })

    return chart_plans[:max_charts]


def render_chart_to_png(
    df: pd.DataFrame,
    plan: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    """
    Render a single chart plan to a PNG file using Plotly and Kaleido.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{plan['chart_id']}.png"
    chart_type = plan["chart_type"]

    fig: go.Figure | None = None

    try:
        if chart_type == "correlation_heatmap":
            cols = plan["columns"]
            corr_matrix = df[cols].apply(pd.to_numeric, errors="coerce").corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title=plan["title"],
                template=PLOTLY_THEME,
            )

        elif chart_type == "line_trend":
            x_col = plan["x_col"]
            y_col = plan["y_col"]
            plot_df = df[[x_col, y_col]].dropna().copy()
            # Sort by date for clean timeline
            plot_df = plot_df.sort_values(by=x_col)
            fig = px.line(
                plot_df,
                x=x_col,
                y=y_col,
                title=plan["title"],
                template=PLOTLY_THEME,
                markers=True,
            )

        elif chart_type == "scatter":
            x_col = plan["x_col"]
            y_col = plan["y_col"]
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=plan["title"],
                template=PLOTLY_THEME,
            )

        elif chart_type == "histogram":
            col = plan["column"]
            fig = px.histogram(
                df,
                x=col,
                title=plan["title"],
                template=PLOTLY_THEME,
            )

        elif chart_type == "categorical_bar":
            col = plan["column"]
            counts = df[col].astype(str).value_counts().reset_index()
            counts.columns = [col, "count"]
            fig = px.bar(
                counts,
                x=col,
                y="count",
                title=plan["title"],
                template=PLOTLY_THEME,
                color="count",
                color_continuous_scale="Blues",
            )

        elif chart_type == "boxplot":
            y_col = plan["y_col"]
            x_col = plan.get("x_col")
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                title=plan["title"],
                template=PLOTLY_THEME,
            )

        if fig is not None:
            fig.update_layout(
                margin=dict(l=40, r=40, t=50, b=40),
                height=500,
                width=800,
            )
            # Render to PNG via Kaleido
            fig.write_image(str(out_path))
            return out_path

    except Exception as e:
        # Graceful fallback: return None without crashing EDA pipeline
        return None

    return None


def generate_all_eda_charts(
    df: pd.DataFrame,
    columns_info: list[dict[str, Any]] | None = None,
    date_columns_info: list[dict[str, Any]] | None = None,
    correlations_info: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    max_charts: int = 6,
) -> list[str]:
    """
    Coordinate deterministic chart planning and static PNG artifact generation.
    Returns list of absolute PNG artifact paths.
    """
    if output_dir is None:
        output_dir = Path("runs/default/artifacts/charts").resolve()

    plans = select_charts_deterministically(
        df=df,
        columns_info=columns_info,
        date_columns_info=date_columns_info,
        correlations_info=correlations_info,
        max_charts=max_charts,
    )

    rendered_paths: list[str] = []
    for plan in plans:
        png_path = render_chart_to_png(df, plan, output_dir)
        if png_path is not None and png_path.is_file():
            rendered_paths.append(str(png_path.resolve()))

    return rendered_paths
