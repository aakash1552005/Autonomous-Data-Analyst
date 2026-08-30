"""
tests/test_pii_detector.py
==========================
Tests for deterministic PII detection, masking utilities, and leak prevention.
Includes structural PII-to-LLM gate verification asserting call_count == 0.
"""

import pandas as pd
from agents.intelligence.pii_detector import detect_column_pii
from agents.intelligence.semantic_labeler import SemanticLabeler
from llm.base import LLMProvider, LLMResponse, TokenGovernor
from utils.mask_for_llm import mask_sample_values, mask_dataframe_for_llm, verify_no_pii_in_prompt


class MockTrackingLLM(LLMProvider):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def _call_provider(self, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            text="mock_label",
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
            model="mock",
            provider="mock",
        )


def test_pii_email_detection():
    # Header name
    is_pii, ptype = detect_column_pii("cust_email_address", pd.Series(["user1@example.com", "user2@domain.org"]))
    assert is_pii is True
    assert ptype == "email"

    # Value patterns only
    is_pii_val, ptype_val = detect_column_pii("contact_info", pd.Series(["john.doe@gmail.com", "jane@company.io"]))
    assert is_pii_val is True
    assert ptype_val == "email"


def test_pii_phone_detection():
    is_pii, ptype = detect_column_pii("mobile_number", pd.Series(["(555) 123-4567", "555-987-6543"]))
    assert is_pii is True
    assert ptype == "phone"


def test_pii_national_id_ssn():
    is_pii, ptype = detect_column_pii("ssn_number", pd.Series(["123-45-6789", "987-65-4321"]))
    assert is_pii is True
    assert ptype == "ssn_national_id"


def test_pii_credit_card_detection():
    is_pii, ptype = detect_column_pii("cc_num", pd.Series(["4532-1234-5678-9012", "5412-3456-7890-1234"]))
    assert is_pii is True
    assert ptype == "credit_card"


def test_pii_name_and_address_detection():
    is_pii_name, ptype_name = detect_column_pii("full_name", pd.Series(["Alice Smith", "Bob Johnson"]))
    assert is_pii_name is True
    assert ptype_name == "name"

    is_pii_addr, ptype_addr = detect_column_pii("street_address", pd.Series(["123 Main St", "456 Oak Ave"]))
    assert is_pii_addr is True
    assert ptype_addr == "address"


def test_mask_for_llm_prevents_pii_leakage():
    raw_emails = ["sensitive_ceo@private.com", "confidential@secret.org"]
    masked = mask_sample_values(raw_emails, is_pii=True, pii_type="email")

    assert all("[REDACTED_EMAIL]" in m for m in masked)
    assert "sensitive_ceo@private.com" not in str(masked)

    # Prompt safety verification helper
    constructed_prompt = f"Analyze these sample items: {', '.join(masked)}"
    assert verify_no_pii_in_prompt(constructed_prompt, raw_emails) is True

    # Intentionally leaked prompt fails check
    leaked_prompt = f"Analyze this user: sensitive_ceo@private.com"
    assert verify_no_pii_in_prompt(leaked_prompt, raw_emails) is False


def test_mask_dataframe_for_llm():
    df = pd.DataFrame({
        "customer_name": ["John Doe", "Jane Smith"],
        "email": ["john@example.com", "jane@example.com"],
        "revenue": [500.0, 750.0],
    })
    masked_df = mask_dataframe_for_llm(df, pii_columns={"customer_name": "name", "email": "email"})

    assert (masked_df["customer_name"] == "[REDACTED_NAME]").all()
    assert (masked_df["email"] == "[REDACTED_EMAIL]").all()
    assert (masked_df["revenue"] == [500.0, 750.0]).all()


def test_structural_pii_gate_blocks_llm_fallback():
    """
    MANDATORY SECURITY TEST:
    Explicitly assert that a column flagged is_pii=True NEVER invokes Tier 3 LLM fallback,
    and mock_llm.call_count remains exactly 0.
    """
    mock_llm = MockTrackingLLM()
    labeler = SemanticLabeler(llm_provider=mock_llm)

    # Obscure header name that won't match Tier 1 rules or Tier 2 patterns, but marked is_pii=True
    pii_series = pd.Series(["secret_value_1", "secret_value_2", "secret_value_3"])
    label, conf, method = labeler.label_column(
        col_name="obscure_pii_field",
        series=pii_series,
        is_pii=True,
        pii_type="custom_pii",
    )

    # Must be blocked and must NOT make any LLM call
    assert method == "pii_blocked"
    assert mock_llm.call_count == 0
