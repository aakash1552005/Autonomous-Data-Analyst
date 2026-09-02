"""
tests/test_phase7_real_data_validation.py
=========================================
Real data validation suite for Phase 7 (Insight & Narrative Agent).
Runs the full multi-agent pipeline:
  DataRouter -> IntelligenceAgent -> CleaningAgent -> EDAAgent -> MLAgent -> InsightAgent
across all 5 sample datasets and verifies grounded insights in dio["insights"].
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from core.data_router import DataRouter
from agents.intelligence.intelligence_agent import IntelligenceAgent
from agents.cleaning.cleaning_agent import CleaningAgent
from agents.eda.eda_agent import EDAAgent
from agents.ml.ml_agent import MLAgent
from agents.insight.insight_agent import InsightAgent


DATASETS = [
    "retail_sales.csv",
    "healthcare_patients.csv",
    "financial_loans.csv",
    "mixed_messy_data.csv",
    "ambiguous_dates_pii.csv",
]


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_phase7_real_data_pipeline_validation(dataset_name: str, tmp_path: Path):
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "sample" / dataset_name
    assert csv_path.exists(), f"Sample dataset not found: {csv_path}"

    fixtures_dir = project_root / "tests" / "fixtures" / "phase7_validation"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest
    router = DataRouter()
    df, dio, run_dir = router.ingest(csv_path, base_runs_dir=tmp_path)

    # 2. Intelligence
    intel_agent = IntelligenceAgent()
    df, dio = intel_agent.run(df, dio)

    # 3. Cleaning
    clean_agent = CleaningAgent()
    cleaned_df, dio = clean_agent.run(df, dio, run_dir=run_dir)

    # 4. EDA
    eda_agent = EDAAgent()
    cleaned_df, dio = eda_agent.run(cleaned_df, dio, run_dir=run_dir)

    # 5. ML
    ml_agent = MLAgent()
    cleaned_df, dio = ml_agent.run(cleaned_df, dio, run_dir=run_dir)

    # 6. Insight
    insight_agent = InsightAgent()
    cleaned_df, dio = insight_agent.run(cleaned_df, dio, run_dir=run_dir)

    # Assertions
    assert dio["progress"]["insight"] == "FINISHED"
    assert isinstance(dio["insights"], list)
    assert len(dio["insights"]) >= 1

    for ins in dio["insights"]:
        assert "id" in ins
        assert "category" in ins
        assert "text" in ins
        assert "confidence" in ins
        assert "evidence" in ins
        assert "recommendation" in ins
        assert "grounded_numbers" in ins
        assert len(ins["grounded_numbers"]) >= 1

    # Save validation snapshot
    output_fixture = fixtures_dir / f"{csv_path.stem}_phase7_dio.json"
    with open(output_fixture, "w", encoding="utf-8") as f:
        f.write(dio.to_json(indent=2))

    assert output_fixture.is_file()
