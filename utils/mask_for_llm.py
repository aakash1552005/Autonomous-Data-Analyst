"""
utils/mask_for_llm.py
=====================
Mandatory PII masking security boundary.
Ensures that sensitive PII values (emails, phones, national IDs, credit cards, names)
are never included in any LLM prompt string.
"""

from __future__ import annotations

import re
from typing import Any, Iterable
import pandas as pd


# Regex patterns used for proactive masking
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def mask_value_str(val: Any, is_pii: bool = False, pii_type: str = "pii") -> str:
    """
    Mask a single value if marked as PII or if matching sensitive PII patterns.
    """
    if pd.isna(val) or val is None:
        return "<NULL>"

    text = str(val).strip()
    if is_pii:
        return f"[REDACTED_{pii_type.upper() or 'PII'}]"

    # Secondary pattern-based safety net
    if EMAIL_REGEX.search(text):
        return "[REDACTED_EMAIL]"
    if PHONE_REGEX.search(text):
        return "[REDACTED_PHONE]"
    if CREDIT_CARD_REGEX.search(text):
        return "[REDACTED_CREDIT_CARD]"
    if SSN_REGEX.search(text):
        return "[REDACTED_SSN]"

    return text


def mask_sample_values(
    values: Iterable[Any],
    is_pii: bool = False,
    pii_type: str = "pii",
    max_samples: int = 10,
) -> list[str]:
    """
    Produce up to `max_samples` sanitized strings suitable for an LLM prompt.
    """
    samples: list[str] = []
    for val in values:
        if pd.isna(val) or val is None or str(val).strip() == "":
            continue
        masked = mask_value_str(val, is_pii=is_pii, pii_type=pii_type)
        samples.append(masked)
        if len(samples) >= max_samples:
            break
    return samples


def mask_dataframe_for_llm(
    df: pd.DataFrame,
    pii_columns: list[str] | dict[str, str],
) -> pd.DataFrame:
    """
    Return a copy of the dataframe with all PII columns masked.
    """
    df_masked = df.copy()
    if isinstance(pii_columns, dict):
        for col, pii_type in pii_columns.items():
            if col in df_masked.columns:
                df_masked[col] = f"[REDACTED_{pii_type.upper()}]"
    else:
        for col in pii_columns:
            if col in df_masked.columns:
                df_masked[col] = "[REDACTED_PII]"
    return df_masked


def verify_no_pii_in_prompt(prompt: str, pii_values: Iterable[str]) -> bool:
    """
    Security assertion helper: confirms no sensitive raw value appears in the prompt.
    Returns True if clean, False if any raw PII value is detected in the prompt text.
    """
    for val in pii_values:
        if not val or len(str(val).strip()) < 3:
            continue
        val_str = str(val).strip()
        if val_str in prompt:
            return False
    return True
