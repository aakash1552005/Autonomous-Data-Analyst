"""
agents/intelligence/pii_detector.py
===================================
Deterministic PII (Personally Identifiable Information) detector.
Detects emails, phone numbers, national IDs, credit cards, full names, and addresses.
Must run before any LLM prompt construction.
"""

from __future__ import annotations

import re
from typing import Any
import pandas as pd


EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PHONE_REGEX = re.compile(r"^(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$")
CREDIT_CARD_REGEX = re.compile(r"^(?:\d{4}[-\s]?){3}\d{4}$")
SSN_REGEX = re.compile(r"^\d{3}-\d{2}-\d{4}$")
AADHAAR_REGEX = re.compile(r"^\d{4}[-\s]?\d{4}[-\s]?\d{4}$")

NAME_COL_REGEX = re.compile(
    r"(?i)^(full_?name|first_?name|last_?name|customer_?name|patient_?name|employee_?name|user_?name|client_?name|fname|lname|name)$"
)
EMAIL_COL_REGEX = re.compile(r"(?i)(email|e_mail|mail_address|mail)")
PHONE_COL_REGEX = re.compile(r"(?i)(phone|mobile|cell|telephone|contact_num|contact_no)")
SSN_COL_REGEX = re.compile(r"(?i)(ssn|social_security|national_id|aadhaar|pan_card|tax_id)")
CC_COL_REGEX = re.compile(r"(?i)(credit_card|card_num|cc_num|debit_card|card_no)")
ADDRESS_COL_REGEX = re.compile(r"(?i)(address|street|addr|street_address|home_address|postal_code|zip_code|zipcode)")


def detect_column_pii(col_name: str, series: pd.Series) -> tuple[bool, str]:
    """
    Evaluate whether a column contains Personally Identifiable Information (PII).
    Returns (is_pii: bool, pii_type: str).
    """
    clean_name = str(col_name).strip()
    non_null = series.dropna().astype(str).str.strip()
    sample = non_null.head(50)
    n_sample = len(sample)

    # 1. Header Name Inspection
    if EMAIL_COL_REGEX.search(clean_name):
        return True, "email"
    if PHONE_COL_REGEX.search(clean_name):
        return True, "phone"
    if SSN_COL_REGEX.search(clean_name):
        return True, "ssn_national_id"
    if CC_COL_REGEX.search(clean_name):
        return True, "credit_card"
    if NAME_COL_REGEX.search(clean_name):
        return True, "name"
    if ADDRESS_COL_REGEX.search(clean_name):
        return True, "address"

    # 2. Value Pattern Inspection
    if n_sample > 0:
        email_matches = sum(1 for v in sample if EMAIL_REGEX.match(v))
        if email_matches / n_sample >= 0.50:
            return True, "email"

        phone_matches = sum(1 for v in sample if PHONE_REGEX.match(v))
        if phone_matches / n_sample >= 0.50:
            return True, "phone"

        cc_matches = sum(1 for v in sample if CREDIT_CARD_REGEX.match(v))
        if cc_matches / n_sample >= 0.50:
            return True, "credit_card"

        ssn_matches = sum(1 for v in sample if SSN_REGEX.match(v) or AADHAAR_REGEX.match(v))
        if ssn_matches / n_sample >= 0.50:
            return True, "ssn_national_id"

    return False, "none"
