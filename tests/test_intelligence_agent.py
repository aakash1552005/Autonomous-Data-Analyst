"""
tests/test_intelligence_agent.py
================================
Integration tests for Agent 1 (IntelligenceAgent).
Verifies full pipeline orchestration, DIO section population, decision provenance, and metrics.
"""

import pandas as pd
from agents.intelligence.intelligence_agent import IntelligenceAgent
from core.dio import DIO
from core.base_agent import ProgressState


def test_intelligence_agent_full_run():
    df = pd.DataFrame({
        "order_id": [101, 102, 103, 104],
        "customer_email": ["c1@shop.com", "c2@shop.com", "c3@shop.com", "c4@shop.com"],
        "order_date": ["25/01/2024", "14/02/2024", "03/03/2024", "18/04/2024"],
        "price": [19.99, 45.00, 12.50, 99.00],
        "quantity": [1, 2, 1, 3],
        "is_returned": [0, 0, 1, 0],
    })

    dio = DIO.create_empty(file_name="orders.csv", dataset_hash="hash12345")
    agent = IntelligenceAgent()
    returned_df, updated_dio = agent.run(df, dio)

    # 1. State check
    assert updated_dio["progress"]["intelligence"] == ProgressState.FINISHED.value

    # 2. DIO Columns check
    cols = {c["name"]: c for c in updated_dio["columns"]}
    assert cols["order_id"]["semantic_label"] == "identifier"
    assert cols["customer_email"]["is_pii"] is True
    assert cols["customer_email"]["pii_type"] == "email"
    assert cols["price"]["semantic_label"] == "currency_amount"
    assert cols["quantity"]["semantic_label"] == "quantity"

    # 3. Date resolution
    assert len(updated_dio["date_columns"]) == 1
    assert updated_dio["date_columns"][0]["detected_format"] == "DD/MM/YYYY"

    # 4. Domain classification
    assert updated_dio["domain_guess"]["domain"] == "retail"
    assert updated_dio["domain_guess"]["confidence"] >= 0.70

    # 5. Quality scoring
    assert updated_dio["quality"]["score"] >= 90

    # 6. Provenance & Metrics
    assert len(updated_dio["decision_log"]) > 0
    assert len(updated_dio["agent_metrics"]) == 1
    assert updated_dio["agent_metrics"][0]["agent"] == "intelligence"
    assert updated_dio["agent_metrics"][0]["runtime_seconds"] > 0
