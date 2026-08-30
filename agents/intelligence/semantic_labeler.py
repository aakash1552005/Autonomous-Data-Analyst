"""
agents/intelligence/semantic_labeler.py
=======================================
Tiered semantic column labeler.
Tier 1: Deterministic dictionary rules on column names (confidence 0.90)
Tier 2: Value-pattern detection on sample data (confidence 0.75-0.85)
Tier 3: LLM fallback with strict PII masking (confidence 0.60, bounded to 1 call per column)
"""

from __future__ import annotations

import re
from typing import Any
import pandas as pd

from llm.base import LLMProvider, LLMTokenBudgetExceededError
from utils.mask_for_llm import mask_sample_values


# --- Tier 1 Rule Dictionary (Regex -> Label) ---
NAME_RULES: list[tuple[re.Pattern, str]] = [
    # Identifiers
    (re.compile(r"(?i)^(id|uuid|guid|_id|id_|key|code|sku|order_id|customer_id|patient_id|user_id|emp_id|transaction_id)$"), "identifier"),
    (re.compile(r"(?i)(_id|id_|account_num|account_no|acc_num|acc_no|tracking_num)"), "identifier"),
    # PII & Contact
    (re.compile(r"(?i)(email|e_mail|mail)"), "email"),
    (re.compile(r"(?i)(phone|mobile|cell|telephone|fax)"), "phone"),
    (re.compile(r"(?i)^(full_?name|first_?name|last_?name|customer_?name|patient_?name|client_?name|user_?name|name)$"), "person_name"),
    (re.compile(r"(?i)(address|street|zip|postal|city|state|country|region|lat|latitude|lon|longitude)"), "geographic"),
    (re.compile(r"(?i)(dob|birth|birthday|date_of_birth)"), "date_of_birth"),
    # Financial & Currency
    (re.compile(r"(?i)(revenue|sales|income|salary|wage|price|cost|amount|amt|spend|budget|balance|fare|fee|total_amt|gross|profit)"), "currency_amount"),
    (re.compile(r"(?i)(account_balance|loan_amount|credit_score|debt|mortgage)"), "financial_metric"),
    # Targets & Outcomes
    (re.compile(r"(?i)^(churn|churned|is_churn|target|label|outcome|class|status|approved|default|defaulted|fraud|is_fraud|purchased|converted)$"), "target_label"),
    # Demographics & Metrics
    (re.compile(r"(?i)^(age|years_old|age_years)$"), "age"),
    (re.compile(r"(?i)(gender|sex)"), "gender"),
    (re.compile(r"(?i)(quantity|qty|units|count|volume)"), "quantity"),
    (re.compile(r"(?i)(discount|discount_pct|rate|tax|percentage|pct|margin)"), "percentage"),
    (re.compile(r"(?i)(rating|score|stars|review_score|satisfaction)"), "score"),
    # Healthcare
    (re.compile(r"(?i)(blood_pressure|bp|diagnosis|heart_rate|pulse|treatment|glucose|insulin|bmi|symptom)"), "health_metric"),
]

# Patterns for Tier 2 Value Sniffing
CURRENCY_SYMBOL_REGEX = re.compile(r"[$€£₹¥]")
EMAIL_VAL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PHONE_VAL_REGEX = re.compile(r"^(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$")
UUID_VAL_REGEX = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


class SemanticLabeler:
    """
    Tiered semantic labeler classifying column semantics.
    """

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        self.llm_provider = llm_provider

    def label_column(
        self,
        col_name: str,
        series: pd.Series,
        is_pii: bool = False,
        pii_type: str = "none",
    ) -> tuple[str, float, str]:
        """
        Label a column using Tier 1 (Rules), Tier 2 (Value Patterns), or Tier 3 (LLM Fallback).
        Returns (semantic_label: str, confidence: float, method_used: str).
        """
        clean_name = str(col_name).strip()
        non_null = series.dropna()

        # --- Tier 1: Rules on Column Name ---
        for pattern, label in NAME_RULES:
            if pattern.search(clean_name):
                return label, 0.90, "rule"

        # --- Tier 2: Pattern Detection on Values ---
        if len(non_null) > 0:
            sample_str = non_null.astype(str).str.strip().head(40)
            n_sample = len(sample_str)

            if n_sample > 0:
                # UUID / Identifier
                uuid_matches = sum(1 for v in sample_str if UUID_VAL_REGEX.match(v))
                if uuid_matches / n_sample >= 0.70:
                    return "identifier", 0.85, "pattern"

                # Email values
                email_matches = sum(1 for v in sample_str if EMAIL_VAL_REGEX.match(v))
                if email_matches / n_sample >= 0.70:
                    return "email", 0.85, "pattern"

                # Phone values
                phone_matches = sum(1 for v in sample_str if PHONE_VAL_REGEX.match(v))
                if phone_matches / n_sample >= 0.70:
                    return "phone", 0.85, "pattern"

                # Currency symbols
                curr_matches = sum(1 for v in sample_str if CURRENCY_SYMBOL_REGEX.search(v))
                if curr_matches / n_sample >= 0.60:
                    return "currency_amount", 0.80, "pattern"

            # Age check on numeric series
            if pd.api.types.is_numeric_dtype(series):
                num_vals = pd.to_numeric(non_null, errors="coerce").dropna()
                if len(num_vals) > 0 and num_vals.min() >= 0 and num_vals.max() <= 120 and (num_vals % 1 == 0).all():
                    if "age" in clean_name.lower():
                        return "age", 0.90, "rule"
                    if num_vals.mean() > 10 and num_vals.mean() < 70:
                        return "age", 0.75, "pattern"

        # --- Tier 3: LLM Fallback (Strictly Bounded) ---
        if self.llm_provider is not None:
            return self._llm_fallback_label(clean_name, non_null, is_pii=is_pii, pii_type=pii_type)

        return "unknown", 0.40, "rule"

    def _llm_fallback_label(
        self,
        col_name: str,
        series: pd.Series,
        is_pii: bool = False,
        pii_type: str = "none",
    ) -> tuple[str, float, str]:
        """
        Execute bounded Tier 3 LLM labeling with masked sample values.
        """
        masked_samples = mask_sample_values(series.head(10), is_pii=is_pii, pii_type=pii_type, max_samples=8)
        samples_str = ", ".join(f"'{s}'" for s in masked_samples) if masked_samples else "none"

        prompt = (
            f"Column name: {col_name}. Sample values: {samples_str}. "
            "In one or two lowercase words, what does this column represent in a business dataset? "
            "Respond with ONLY the short label."
        )

        try:
            raw_response = self.llm_provider.complete(prompt, max_tokens=50, temperature=0.0)
            clean_label = re.sub(r"[^\w\s]", "", raw_response).strip().lower()
            clean_label = "_".join(clean_label.split()[:2])
            if clean_label:
                return clean_label, 0.60, "llm"
        except LLMTokenBudgetExceededError:
            # Respect token governor cleanly without failing
            return "unknown", 0.40, "fallback_budget_exhausted"
        except Exception:
            # Fall back gracefully on network or provider error
            return "unknown", 0.40, "fallback_llm_error"

        return "unknown", 0.40, "llm"
