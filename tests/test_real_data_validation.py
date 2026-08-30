"""
tests/test_real_data_validation.py
==================================
Real Data Validation suite for Phase 3 Intelligence Agent.
Evaluates 5 distinct messy datasets spanning retail, healthcare, finance,
mixed anomalies, and ambiguous dates with sensitive PII fields.
Saves DIO validation outputs to tests/fixtures/phase3_validation/.
"""

import json
from pathlib import Path
import pytest
from core.data_router import DataRouter
from agents.intelligence.intelligence_agent import IntelligenceAgent
from core.dio import DIO
from utils.mask_for_llm import verify_no_pii_in_prompt


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "phase3_validation"


def save_validation_dio(dio: DIO, dataset_name: str) -> None:
    """Save pruned DIO output for validation inspection."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_file = FIXTURES_DIR / f"{dataset_name}_dio_output.json"
    
    validation_payload = {
        "dataset_name": dataset_name,
        "dataset_hash": dio.dataset_hash,
        "ingestion": dio.ingestion,
        "columns": dio.columns,
        "date_columns": dio.date_columns,
        "domain_guess": dio.domain_guess,
        "quality": dio.quality,
        "agent_metrics": dio.agent_metrics,
        "decision_log": dio.decision_log,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(validation_payload, f, indent=2, default=str)


def test_dataset_1_retail_sales(tmp_path: Path):
    file_path = DATA_DIR / "retail_sales.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)

    agent = IntelligenceAgent()
    df, dio = agent.run(df, dio)

    assert dio["domain_guess"]["domain"] == "retail"
    assert dio["domain_guess"]["confidence"] >= 0.70

    # Date resolution check
    date_col = next(d for d in dio["date_columns"] if d["column"] == "order_date")
    assert date_col["detected_format"] == "DD/MM/YYYY"
    assert date_col["confidence"] == 1.0

    save_validation_dio(dio, "retail_sales")

    # Serialization round trip
    json_str = dio.to_json()
    reconstructed = DIO.from_json(json_str)
    assert reconstructed.to_dict() == dio.to_dict()


def test_dataset_2_healthcare_patients(tmp_path: Path):
    file_path = DATA_DIR / "healthcare_patients.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)

    agent = IntelligenceAgent()
    df, dio = agent.run(df, dio)

    assert dio["domain_guess"]["domain"] == "healthcare"
    assert dio["domain_guess"]["confidence"] >= 0.70

    # ISO dates resolved
    admission_date = next(d for d in dio["date_columns"] if d["column"] == "admission_date")
    assert admission_date["detected_format"] == "YYYY-MM-DD"

    # Quality score handles missing glucose and discharge_date
    assert dio["quality"]["score"] >= 80
    assert any("null values" in iss for iss in dio["quality"]["issues"])

    save_validation_dio(dio, "healthcare_patients")


def test_dataset_3_financial_loans(tmp_path: Path):
    file_path = DATA_DIR / "financial_loans.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)

    agent = IntelligenceAgent()
    df, dio = agent.run(df, dio)

    assert dio["domain_guess"]["domain"] == "finance"
    assert dio["domain_guess"]["confidence"] >= 0.70

    # Target candidate heuristic check
    target_col = next(c for c in dio["columns"] if c["name"] == "default_status")
    assert target_col["is_target_candidate"] is True

    save_validation_dio(dio, "financial_loans")


def test_dataset_4_mixed_messy_data(tmp_path: Path):
    file_path = DATA_DIR / "mixed_messy_data.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)

    agent = IntelligenceAgent()
    df, dio = agent.run(df, dio)

    # Duplicates detected and penalized in quality score
    assert any("Duplicate rows detected" in iss for iss in dio["quality"]["issues"])
    assert dio["quality"]["score"] < 95

    save_validation_dio(dio, "mixed_messy_data")


def test_dataset_5_ambiguous_dates_and_pii(tmp_path: Path):
    file_path = DATA_DIR / "ambiguous_dates_pii.csv"
    router = DataRouter()
    df, dio, run_dir = router.ingest(file_path, base_runs_dir=tmp_path)

    agent = IntelligenceAgent()
    df, dio = agent.run(df, dio)

    # 1. PII detection
    cols = {c["name"]: c for c in dio["columns"]}
    assert cols["full_name"]["is_pii"] is True
    assert cols["email"]["is_pii"] is True
    assert cols["phone_number"]["is_pii"] is True
    assert cols["credit_card"]["is_pii"] is True
    assert cols["amount"]["is_pii"] is False

    # 2. Date Ambiguity (all components <= 12)
    sub_date = next(d for d in dio["date_columns"] if d["column"] == "subscription_date")
    assert sub_date["detected_format"] == "ambiguous"
    assert sub_date["confidence"] == 0.50
    assert sub_date["needs_user_confirmation"] is True

    # 3. PII Leakage Verification
    raw_sensitive_values = [
        "johndoe@email.com",
        "4532-1122-3344-5566",
        "Johnathan Doe",
        "(555) 234-5678",
    ]
    log_str = str(dio["decision_log"])
    assert verify_no_pii_in_prompt(log_str, raw_sensitive_values) is True

    save_validation_dio(dio, "ambiguous_dates_pii")
