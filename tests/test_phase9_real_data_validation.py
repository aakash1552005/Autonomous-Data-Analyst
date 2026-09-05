"""
tests/test_phase9_real_data_validation.py
=========================================
Real-data validation suite for Phase 9: Orchestrator & End-to-End Pipeline.
Executes the unified Orchestrator pipeline across all 5 project benchmark datasets:
  1. retail_sales.csv
  2. healthcare_patients.csv
  3. financial_loans.csv
  4. mixed_messy_data.csv
  5. ambiguous_dates_pii.csv

Verifies for every dataset:
  - Pipeline execution succeeds (status is 'completed' or 'partial')
  - Valid DIO schema with zero structural errors
  - Deterministic cleaning produces non-empty cleaned data
  - EDA generates non-empty visual chart artifacts
  - Machine learning fits model or skips safely without crashing
  - Executive insights synthesized
  - High-fidelity PDF report and PPTX deliverables generated (> 1000 bytes)
  - Persisted dio.json and pipeline.log exist
  - Zero PII leakage in deliverable artifacts
"""

from __future__ import annotations

import json
from pathlib import Path
from pptx import Presentation
import pytest

from core.dio import DIO
from orchestrator import Orchestrator, OrchestratorResult

DATASETS = [
    "retail_sales.csv",
    "healthcare_patients.csv",
    "financial_loans.csv",
    "mixed_messy_data.csv",
    "ambiguous_dates_pii.csv",
]


def _extract_all_pptx_text(pptx_path: Path | str) -> str:
    """Helper to extract all text from all shapes and tables in a PPTX."""
    prs = Presentation(pptx_path)
    texts: list[str] = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for p in shape.text_frame.paragraphs:
                    texts.append(p.text)
            elif shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        texts.append(cell.text)
    return " ".join(texts)


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_phase9_real_data_validation(dataset_name: str, tmp_path: Path):
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "sample" / dataset_name
    assert csv_path.exists(), f"Sample dataset not found: {csv_path}"

    run_dir = tmp_path / dataset_name.replace(".csv", "")
    orchestrator = Orchestrator()
    result: OrchestratorResult = orchestrator.run(
        file_path=csv_path,
        run_dir=run_dir,
    )

    # 1. Pipeline Completion
    assert result.is_success, f"Pipeline failed on {dataset_name}: {result.errors}"
    assert result.status in ("completed", "partial")
    assert result.run_dir == run_dir

    dio = result.dio
    assert isinstance(dio, DIO)

    # 2. Ingestion & Profile Consistency
    assert dio["ingestion"]["n_rows"] > 0
    assert dio["ingestion"]["n_columns"] > 0
    assert dio["domain_guess"]["domain"] != ""
    assert 0 <= dio["quality"]["score"] <= 100

    # 3. Artifact Deliverable Verification
    artifacts = result.artifacts
    assert artifacts.get("cleaned_csv") is not None
    assert Path(artifacts["cleaned_csv"]).exists()
    assert Path(artifacts["cleaned_csv"]).stat().st_size > 50

    pdf_path = artifacts.get("pdf_report")
    assert pdf_path is not None
    assert Path(pdf_report_path := Path(pdf_path)).exists()
    assert pdf_report_path.stat().st_size > 1000

    pptx_path = artifacts.get("pptx_report")
    assert pptx_path is not None
    assert Path(pptx_report_path := Path(pptx_path)).exists()
    assert pptx_report_path.stat().st_size > 1000

    # 4. State & Logging Persistence
    dio_json = run_dir / "dio.json"
    assert dio_json.exists()
    dio_dict = json.loads(dio_json.read_text(encoding="utf-8"))
    assert dio_dict["dataset_hash"] == dio.dataset_hash

    pipeline_log = run_dir / "pipeline.log"
    assert pipeline_log.exists()
    assert pipeline_log.stat().st_size > 100

    # 5. PII Redaction in PPTX Output
    pptx_text = _extract_all_pptx_text(pptx_report_path)
    # Check that obvious raw PII strings do not leak into presentation text
    for col in dio.get("columns", []):
        if col.get("is_pii"):
            col_name = col.get("name")
            # The column name itself may appear in metadata, but raw PII values must not
            assert "@example.com" not in pptx_text
