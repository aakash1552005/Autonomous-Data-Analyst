"""
agents/insight/hallucination_guard.py
=====================================
Strict numerical hallucination guard for InsightAgent.
Extracts numbers from generated text and verifies that all cited figures
are grounded against computed DIO statistics within a configurable tolerance.
"""

from __future__ import annotations

import math
import re
from typing import Any


# Regex to extract numbers (including floats, percentages, commas in integers like 1,000)
NUMBER_REGEX = re.compile(r"[-+]?\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")


def extract_numbers_from_text(text: str) -> list[float]:
    """
    Extract all numerical values (floats, ints, percentages) from a text string.
    Normalizes percentages and thousands separators.
    """
    matches = NUMBER_REGEX.findall(text)
    extracted: list[float] = []

    for m in matches:
        clean = m.replace(",", "").strip()
        if clean.endswith("%"):
            clean = clean[:-1].strip()

        try:
            val = float(clean)
            if not (math.isnan(val) or math.isinf(val)):
                extracted.append(val)
        except ValueError:
            continue

    return extracted


def is_number_grounded(
    num: float,
    grounded_ints: set[int],
    grounded_floats: set[float],
    tolerance: float = 0.05,
) -> bool:
    """
    Check if a specific number matches grounded integers or floats within tolerance.
    """
    # Check exact match in ints or floats
    if float(num).is_integer() and int(num) in grounded_ints:
        return True

    if num in grounded_floats:
        return True

    # Check match against grounded floats with tolerance
    for gf in grounded_floats:
        abs_diff = abs(num - gf)
        rel_diff = abs_diff / abs(gf) if gf != 0 else abs_diff

        if abs_diff <= 0.02 or rel_diff <= tolerance:
            return True

        # Percentage scaling check for proportions (e.g. gf=0.81 -> num=81.0)
        if 0.0 < gf <= 1.0:
            gf_pct = gf * 100.0
            abs_pct_diff = abs(num - gf_pct)
            rel_pct_diff = abs_pct_diff / gf_pct
            if abs_pct_diff <= 0.5 or rel_pct_diff <= tolerance:
                return True

    # Check if num is a percentage representation of an integer proportion or score
    for gi in grounded_ints:
        # If integer represents a score or percentage (e.g. 95), allow small rounding
        if abs(num - gi) <= 0.01:
            return True

    return False


def verify_insight_grounding(
    insight: dict[str, Any],
    grounded_ints: set[int],
    grounded_floats: set[float],
    tolerance: float = 0.05,
) -> tuple[bool, list[float], str]:
    """
    Verify that an insight is fully grounded in the provided DIO numbers.

    Rules:
    1. Insight must contain at least one quantifiable number (qualitative claims without numbers are rejected).
    2. Every extracted number in insight['text'] must be grounded in grounded_ints or grounded_floats.
    3. If any number fails grounding, the insight is rejected.

    Returns:
        (is_grounded, extracted_numbers, rejection_reason)
    """
    text = insight.get("text", "").strip()
    if not text:
        return False, [], "Empty insight text"

    extracted = extract_numbers_from_text(text)
    if not extracted:
        return False, [], "Insight contains zero numbers (unquantifiable claim)"

    ungrounded_nums: list[float] = []
    for n in extracted:
        if not is_number_grounded(n, grounded_ints, grounded_floats, tolerance=tolerance):
            ungrounded_nums.append(n)

    if ungrounded_nums:
        return (
            False,
            extracted,
            f"Ungrounded numerical claim(s) detected: {ungrounded_nums}",
        )

    return True, extracted, ""


VALID_INSIGHT_CATEGORIES = {"distribution", "correlation", "quality", "machine_learning"}


def validate_insight_schema(candidate: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate that an LLM-generated insight candidate strictly complies with the 7-field contract.
    Ensures category, text, confidence, evidence, and recommendation are present, validly typed,
    and meet domain constraints.
    Returns (is_valid, rejection_reason).
    """
    if not isinstance(candidate, dict):
        return False, "Candidate is not a dictionary"

    # 1. category: required, string, in valid categories
    category = candidate.get("category")
    if not isinstance(category, str) or not category.strip():
        return False, "Missing or empty 'category' field"
    if category.strip().lower() not in VALID_INSIGHT_CATEGORIES:
        return False, f"Invalid 'category' '{category}'. Must be one of: {sorted(VALID_INSIGHT_CATEGORIES)}"

    # 2. text: required, non-empty string
    text = candidate.get("text")
    if not isinstance(text, str) or not text.strip():
        return False, "Missing or empty 'text' field"

    # 3. confidence: required, numeric (int or float, not bool), between 0.0 and 1.0
    confidence = candidate.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return False, f"Invalid 'confidence' value: {confidence!r}. Must be a float between 0.0 and 1.0"
    if not (0.0 <= float(confidence) <= 1.0):
        return False, f"Out-of-range 'confidence' value: {confidence}. Must be between 0.0 and 1.0"

    # 4. evidence: required, non-empty string
    evidence = candidate.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        return False, "Missing or empty 'evidence' field"

    # 5. recommendation: required, non-empty string
    recommendation = candidate.get("recommendation")
    if not isinstance(recommendation, str) or not recommendation.strip():
        return False, "Missing or empty 'recommendation' field"

    return True, ""
