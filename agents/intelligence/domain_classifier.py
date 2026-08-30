"""
agents/intelligence/domain_classifier.py
========================================
Deterministic domain classifier based on weighted semantic label voting.
Classifies dataset domain into retail, healthcare, finance, or generic.
"""

from __future__ import annotations

from typing import Any


DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "retail": {
        "revenue", "sales", "sku", "customer_id", "order_date", "price",
        "quantity", "discount", "product", "store", "inventory", "cart",
        "item", "unit_price", "orders", "purchased",
    },
    "healthcare": {
        "blood_pressure", "diagnosis", "patient_id", "heart_rate", "pulse",
        "glucose", "insulin", "bmi", "treatment", "symptom", "dosage",
        "disease", "medical", "doctor", "hospital", "health_metric",
    },
    "finance": {
        "account_balance", "transaction_id", "loan_amount", "interest_rate",
        "credit_score", "default", "financial_metric", "balance", "debt",
        "mortgage", "credit_limit", "assets", "liabilities", "loan_status",
    },
}


def classify_domain(columns_info: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Perform weighted voting across semantic labels and column names to determine domain.
    Returns: {"domain": str, "confidence": float}
    """
    domain_scores: dict[str, float] = {"retail": 0.0, "healthcare": 0.0, "finance": 0.0}

    for col in columns_info:
        name = str(col.get("name", "")).lower().strip()
        label = str(col.get("semantic_label", "")).lower().strip()

        for domain, keywords in DOMAIN_KEYWORDS.items():
            # Check semantic label match (higher weight: +2.0)
            if label in keywords:
                domain_scores[domain] += 2.0
            # Check column name substring match (+1.0)
            elif any(kw in name for kw in keywords):
                domain_scores[domain] += 1.0

    best_domain = max(domain_scores, key=domain_scores.get)
    max_score = domain_scores[best_domain]
    total_score = sum(domain_scores.values())

    if max_score >= 2.0 and total_score > 0:
        # Confidence scaled by proportion of dominance
        confidence = min(0.95, max(0.70, round(max_score / total_score, 2)))
        return {
            "domain": best_domain,
            "confidence": confidence,
        }

    return {
        "domain": "generic",
        "confidence": 0.50,
    }
