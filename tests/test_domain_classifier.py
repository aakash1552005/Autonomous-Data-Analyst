"""
tests/test_domain_classifier.py
===============================
Tests for deterministic weighted domain classification (Retail, Healthcare, Finance, Generic).
"""

from agents.intelligence.domain_classifier import classify_domain


def test_domain_healthcare():
    columns_info = [
        {"name": "patient_id", "semantic_label": "identifier"},
        {"name": "blood_pressure", "semantic_label": "health_metric"},
        {"name": "diagnosis", "semantic_label": "health_metric"},
        {"name": "glucose_level", "semantic_label": "health_metric"},
    ]
    res = classify_domain(columns_info)
    assert res["domain"] == "healthcare"
    assert res["confidence"] >= 0.70


def test_domain_retail():
    columns_info = [
        {"name": "order_id", "semantic_label": "identifier"},
        {"name": "sku", "semantic_label": "sku"},
        {"name": "unit_price", "semantic_label": "currency_amount"},
        {"name": "quantity", "semantic_label": "quantity"},
        {"name": "discount", "semantic_label": "percentage"},
    ]
    res = classify_domain(columns_info)
    assert res["domain"] == "retail"
    assert res["confidence"] >= 0.70


def test_domain_finance():
    columns_info = [
        {"name": "transaction_id", "semantic_label": "identifier"},
        {"name": "account_balance", "semantic_label": "financial_metric"},
        {"name": "loan_amount", "semantic_label": "financial_metric"},
        {"name": "credit_score", "semantic_label": "score"},
        {"name": "default_status", "semantic_label": "target_label"},
    ]
    res = classify_domain(columns_info)
    assert res["domain"] == "finance"
    assert res["confidence"] >= 0.70


def test_domain_generic_fallback():
    columns_info = [
        {"name": "col_a", "semantic_label": "string"},
        {"name": "col_b", "semantic_label": "int"},
        {"name": "col_c", "semantic_label": "float"},
    ]
    res = classify_domain(columns_info)
    assert res["domain"] == "generic"
    assert res["confidence"] == 0.50
