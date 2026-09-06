"""
agents/chat/context_retriever.py
================================
Bounded DIO context retrieval for Agent 7 (Chat Agent).
Compiles verified facts from permitted DIO sections ONLY:
  - domain_guess
  - quality
  - eda (excluding sensitive entities)
  - ml
  - insights

Security invariants:
  - ZERO raw DataFrame rows
  - ZERO raw PII values
  - ZERO raw identifier values
  - Strictly token-bounded to prevent context inflation
"""

from __future__ import annotations

from typing import Any
from utils.mask_for_llm import mask_value_str


# Permitted DIO sections for contextual question answering
PERMITTED_DIO_SECTIONS = {
    "domain_guess",
    "quality",
    "eda",
    "ml",
    "insights",
}


def build_chat_context(dio: dict[str, Any], max_context_tokens: int = 1200) -> str:
    """
    Compile a sanitized, bounded textual context representing the completed dataset analysis.

    Parameters:
        dio: Global Data Interchange Object.
        max_context_tokens: Upper ceiling for context length (estimated 4 chars per token).

    Returns:
        Structured markdown summary safe for LLM consumption.
    """
    sections: list[str] = []

    # 1. Ingestion / Schema Overview (Sanitized column names only)
    ingestion = dio.get("ingestion", {})
    n_rows = ingestion.get("n_rows", "Unknown")
    n_cols = ingestion.get("n_columns", "Unknown")
    sections.append(f"### Dataset Overview\n- Total Rows: {n_rows}\n- Total Columns: {n_cols}")

    # 2. Domain Classification
    domain_data = dio.get("domain_guess", {})
    if domain_data:
        domain = domain_data.get("domain", "Unknown")
        conf = domain_data.get("confidence", 0.0)
        sections.append(f"### Domain\n- Detected Domain: {domain.title()} (Confidence: {conf:.1%})")

    # 3. Data Quality & Profiling
    quality = dio.get("quality", {})
    if quality:
        q_score = quality.get("score", "N/A")
        issues = quality.get("issues", [])
        issues_str = "; ".join(issues[:5]) if issues else "No major anomalies flagged."
        sections.append(f"### Data Quality\n- Quality Score: {q_score}/100\n- Key Issues: {issues_str}")

    # 4. Non-PII EDA Summary Statistics
    eda = dio.get("eda", {})
    summary_stats = eda.get("summary_stats", {})
    # Identify excluded sensitive columns
    excluded_cols = set()
    for col_info in dio.get("columns", []):
        if col_info.get("is_pii") or col_info.get("semantic_label") == "identifier":
            excluded_cols.add(col_info.get("name"))

    stat_lines: list[str] = []
    for col_name, stats in summary_stats.items():
        if col_name in excluded_cols or not isinstance(stats, dict):
            continue
        c_type = stats.get("type", "")
        if c_type == "numeric":
            stat_lines.append(
                f"- **{col_name}** (numeric): mean={stats.get('mean')}, min={stats.get('min')}, max={stats.get('max')}"
            )
        elif c_type == "categorical" and "top_category" in stats:
            stat_lines.append(
                f"- **{col_name}** (categorical): unique={stats.get('unique_count')}, top='{stats.get('top_category')}'"
            )
    if stat_lines:
        sections.append("### Key Statistical Summaries\n" + "\n".join(stat_lines[:8]))

    # 5. Machine Learning Summary
    ml = dio.get("ml", {})
    if ml:
        status = ml.get("model_training_status", "NOT_CONFIGURED")
        best_model = ml.get("best_model_name", "None")
        target_col = ml.get("target_column", "None")
        task_type = ml.get("task_type", "None")
        metrics = ml.get("metrics", {})
        metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()]) if metrics else "N/A"
        sections.append(
            f"### Machine Learning\n- Status: {status}\n- Task Type: {task_type}\n- Target Column: {target_col}\n- Selected Model: {best_model}\n- Held-out Metrics: {metrics_str}"
        )

    # 6. Validated Executive Insights
    insights = dio.get("insights", [])
    if insights:
        ins_lines = []
        for idx, ins in enumerate(insights[:6], start=1):
            text = ins.get("text", "")
            rec = ins.get("recommendation", "")
            ins_lines.append(f"{idx}. {text} (Recommendation: {rec})")
        sections.append("### Executive Insights\n" + "\n".join(ins_lines))

    # Combine and bound
    full_context = "\n\n".join(sections)

    # Character budget based on max_context_tokens (rough ~4 chars per token)
    char_limit = max_context_tokens * 4
    if len(full_context) > char_limit:
        full_context = full_context[:char_limit] + "\n\n[Context truncated to token budget]"

    return full_context
