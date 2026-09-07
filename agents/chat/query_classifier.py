"""
agents/chat/query_classifier.py
===============================
Deterministic query classification for Agent 7 (Chat Agent).
Extracts intent and maps queries to one of the 11 whitelisted operations:
  1. mean
  2. sum
  3. count
  4. min
  5. max
  6. value_counts
  7. groupby_mean
  8. median
  9. std
  10. variance
  11. groupby_sum
Zero LLM dependencies. Classification is 100% deterministic and schema-bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass
class ClassifiedQuery:
    """Represents the deterministic classification of a user query."""
    operation: str  # "mean", "sum", "count", "min", "max", "value_counts", "groupby_mean", or "unclassified"
    column: str | None = None
    group_column: str | None = None
    raw_query: str = ""


# Supported safe operations in Tier 1 (V1.1 Increment 2: 11 operations)
SAFE_OPERATIONS = {
    "mean",
    "sum",
    "count",
    "min",
    "max",
    "value_counts",
    "groupby_mean",
    "median",
    "std",
    "variance",
    "groupby_sum",
}


def _find_column_in_text(text: str, columns: list[str]) -> str | None:
    """
    Search for a column name in the given text.
    Matches case-insensitively, prioritizing longer column names to prevent substring shadowing
    (e.g., 'total_amount' before 'amount').
    """
    text_lower = text.lower()
    # Sort columns by length descending
    sorted_cols = sorted(columns, key=len, reverse=True)

    for col in sorted_cols:
        col_lower = col.lower()
        # Direct word boundary or quoted boundary match
        pattern = rf"(?:\b|['\"]){re.escape(col_lower)}(?:\b|['\"])"
        if re.search(pattern, text_lower):
            return col

    return None


def classify_query(
    query: str,
    columns: list[str],
    safe_operations: set[str] | list[str] | None = None,
) -> ClassifiedQuery:
    """
    Deterministically classify a user query into a safe pandas operation.

    Parameters:
        query: Raw query string from user.
        columns: List of valid column names in the active dataset.
        safe_operations: Set of permitted operations from configuration.
    """
    allowed_ops = set(safe_operations) if safe_operations else SAFE_OPERATIONS
    cleaned_query = query.strip()
    q_lower = cleaned_query.lower()

    if not cleaned_query:
        return ClassifiedQuery(operation="unclassified", raw_query=query)

    # --- 1. Check GroupBy Mean ---
    # Patterns: "average/mean {num_col} by/per/grouped by/for each {cat_col}"
    if "groupby_mean" in allowed_ops:
        gb_patterns = [
            r"(?:what is the\s+)?(?:mean|average|avg)\s+(?:of\s+)?(?P<num>.+?)\s+(?:grouped by|by|per|for each)\s+(?P<grp>.+)",
            r"(?:group by|groupby)\s+(?P<grp>.+?)\s+(?:and calculate|and find|with)?\s*(?:the\s+)?(?:mean|average|avg)\s+(?:of\s+)?(?P<num>.+)",
        ]
        for pat in gb_patterns:
            m = re.search(pat, q_lower)
            if m:
                num_text = m.group("num").strip()
                grp_text = m.group("grp").strip()
                num_col = _find_column_in_text(num_text, columns)
                grp_col = _find_column_in_text(grp_text, columns)
                if num_col and grp_col and num_col != grp_col:
                    return ClassifiedQuery(
                        operation="groupby_mean",
                        column=num_col,
                        group_column=grp_col,
                        raw_query=query,
                    )

    # --- 1b. Check GroupBy Sum ---
    # Patterns: "total/sum {num_col} by/per/grouped by/for each {cat_col}"
    if "groupby_sum" in allowed_ops:
        gb_sum_patterns = [
            r"(?:what is the\s+)?(?:sum|total)\s+(?:of\s+)?(?P<num>.+?)\s+(?:grouped by|by|per|for each)\s+(?P<grp>.+)",
            r"(?:group by|groupby)\s+(?P<grp>.+?)\s+(?:and calculate|and find|with)?\s*(?:the\s+)?(?:sum|total)\s+(?:of\s+)?(?P<num>.+)",
        ]
        for pat in gb_sum_patterns:
            m = re.search(pat, q_lower)
            if m:
                num_text = m.group("num").strip()
                grp_text = m.group("grp").strip()
                num_col = _find_column_in_text(num_text, columns)
                grp_col = _find_column_in_text(grp_text, columns)
                if num_col and grp_col and num_col != grp_col:
                    return ClassifiedQuery(
                        operation="groupby_sum",
                        column=num_col,
                        group_column=grp_col,
                        raw_query=query,
                    )

    # --- 2. Value Counts / Distribution ---
    if "value_counts" in allowed_ops:
        vc_patterns = [
            r"(?:value\s*counts?|distribution|breakdown|frequencies|frequency|categories)\s+(?:of|for|in)?\s*(.+)",
            r"(?:what are the\s+)?(?:top|unique)?\s*categories\s+(?:in|of)\s+(.+)",
            r"(?:show\s+)?(?:the\s+)?(?:most\s+common|most\s+frequent|top\s+values?|frequent)\s+(?:of\s+|in\s+)?(.+)",
            r"(?:how is\s+)?(?P<col>.+?)\s+distributed",
        ]
        for pat in vc_patterns:
            m = re.search(pat, q_lower)
            if m:
                target_text = m.group(1).strip()
                col = _find_column_in_text(target_text, columns)
                if col:
                    return ClassifiedQuery(operation="value_counts", column=col, raw_query=query)

    # --- 3. Median ---
    if "median" in allowed_ops:
        median_patterns = [
            r"(?:what is the\s+)?(?:median)\s+(?:of\s+)?(.+)",
            r"(?:calculate\s+)?(?:the\s+)?(?:median)\s+(?:of\s+)?(.+)",
            r"(?P<col>.+?)\s+(?:median)\b",
        ]
        for pat in median_patterns:
            m = re.search(pat, q_lower)
            if m:
                target_text = m.group(1).strip()
                col = _find_column_in_text(target_text, columns)
                if col:
                    return ClassifiedQuery(operation="median", column=col, raw_query=query)

    # --- 4. Standard Deviation ---
    if "std" in allowed_ops:
        std_patterns = [
            r"(?:what is the\s+)?(?:standard\s+deviation|std\s+dev|stddev|stdev|std)\s+(?:of\s+)?(.+)",
            r"(?:calculate\s+)?(?:the\s+)?(?:standard\s+deviation|std\s+dev|stddev|stdev|std)\s+(?:of\s+)?(.+)",
            r"(?P<col>.+?)\s+(?:standard\s+deviation|std\s+dev|stddev|stdev|std)\b",
        ]
        for pat in std_patterns:
            m = re.search(pat, q_lower)
            if m:
                target_text = m.group(1).strip()
                col = _find_column_in_text(target_text, columns)
                if col:
                    return ClassifiedQuery(operation="std", column=col, raw_query=query)

    # --- 5. Variance ---
    if "variance" in allowed_ops:
        var_patterns = [
            r"(?:what is the\s+)?(?:variance|var)\s+(?:of\s+)?(.+)",
            r"(?:calculate\s+)?(?:the\s+)?(?:variance|var)\s+(?:of\s+)?(.+)",
            r"(?P<col>.+?)\s+(?:variance|var)\b",
        ]
        for pat in var_patterns:
            m = re.search(pat, q_lower)
            if m:
                target_text = m.group(1).strip()
                col = _find_column_in_text(target_text, columns)
                if col:
                    return ClassifiedQuery(operation="variance", column=col, raw_query=query)

    # --- 6. Mean / Average ---
    if "mean" in allowed_ops:
        mean_patterns = [
            r"(?:what is the\s+)?(?:mean|average|avg)\s+(?:of\s+)?(.+)",
            r"(?:calculate\s+)?(?:the\s+)?(?:mean|average|avg)\s+(?:of\s+)?(.+)",
            r"(?P<col>.+?)\s+(?:mean|average|avg)\b",
        ]
        for pat in mean_patterns:
            m = re.search(pat, q_lower)
            if m:
                target_text = m.group(1).strip()
                col = _find_column_in_text(target_text, columns)
                if col:
                    return ClassifiedQuery(operation="mean", column=col, raw_query=query)

    # --- 7. Sum / Total ---
    if "sum" in allowed_ops:
        sum_patterns = [
            r"(?:what is the\s+)?(?:sum|total)\s+(?:of\s+)?(.+)",
            r"(?:calculate\s+)?(?:the\s+)?(?:sum|total)\s+(?:of\s+)?(.+)",
            r"(?P<col>.+?)\s+(?:sum|total)\b",
        ]
        for pat in sum_patterns:
            m = re.search(pat, q_lower)
            if m:
                target_text = m.group(1).strip()
                col = _find_column_in_text(target_text, columns)
                if col:
                    return ClassifiedQuery(operation="sum", column=col, raw_query=query)

    # --- 8. Minimum ---
    if "min" in allowed_ops:
        min_patterns = [
            r"(?:what is the\s+)?(?:min|minimum|lowest|smallest)\s+(?:value\s+of\s+|of\s+)?(.+)",
            r"(?:find\s+)?(?:the\s+)?(?:min|minimum|lowest|smallest)\s+(?:of\s+)?(.+)",
            r"(?P<col>.+?)\s+(?:min|minimum|lowest|smallest)\b",
        ]
        for pat in min_patterns:
            m = re.search(pat, q_lower)
            if m:
                target_text = m.group(1).strip()
                col = _find_column_in_text(target_text, columns)
                if col:
                    return ClassifiedQuery(operation="min", column=col, raw_query=query)

    # --- 9. Maximum ---
    if "max" in allowed_ops:
        max_patterns = [
            r"(?:what is the\s+)?(?:max|maximum|highest|largest)\s+(?:value\s+of\s+|of\s+)?(.+)",
            r"(?:find\s+)?(?:the\s+)?(?:max|maximum|highest|largest)\s+(?:of\s+)?(.+)",
            r"(?P<col>.+?)\s+(?:max|maximum|highest|largest)\b",
        ]
        for pat in max_patterns:
            m = re.search(pat, q_lower)
            if m:
                target_text = m.group(1).strip()
                col = _find_column_in_text(target_text, columns)
                if col:
                    return ClassifiedQuery(operation="max", column=col, raw_query=query)

    # --- 10. Count ---
    if "count" in allowed_ops:
        # Row count special cases
        if re.search(r"\b(?:how many rows|total rows|row count|number of rows|count of rows)\b", q_lower):
            return ClassifiedQuery(operation="count", column=None, raw_query=query)

        count_patterns = [
            r"(?:what is the\s+)?(?:count|number)\s+(?:of\s+)?(.+)",
            r"how many\s+(?P<target>.+)",
        ]
        for pat in count_patterns:
            m = re.search(pat, q_lower)
            if m:
                target_text = m.group(1).strip()
                col = _find_column_in_text(target_text, columns)
                if col:
                    return ClassifiedQuery(operation="count", column=col, raw_query=query)

    return ClassifiedQuery(operation="unclassified", raw_query=query)
