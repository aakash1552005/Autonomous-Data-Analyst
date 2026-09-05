"""
tests/test_phase9_e2e.py
========================
End-to-End Pipeline Integration Test for Phase 9.
Executes the full Orchestrator pipeline with all real agents on retail_sales.csv.
Validates:
  1. Full pipeline execution returns status == "completed"
  2. DIO schema validation passes with 0 issues
  3. All filesystem artifacts are generated and non-empty:
     - Cleaned CSV
     - Visualization charts
     - Executive PDF report
     - Executive PPTX deck
     - Persisted dio.json
     - Run pipeline.log
  4. Per-stage execution timings and statuses are recorded
  5. Progress callbacks fire continuously up to 100%
"""

import json
from pathlib import Path
import pytest

from core.dio import DIO
from orchestrator import Orchestrator, OrchestratorResult


def test_phase9_full_pipeline_e2e(tmp_path: Path):
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "sample" / "retail_sales.csv"
    assert csv_path.exists(), f"Sample dataset not found at {csv_path}"

    run_dir = tmp_path / "e2e_run"
    progress_history: list[tuple[str, float, str]] = []

    def progress_cb(stage: str, pct: float, msg: str) -> None:
        progress_history.append((stage, pct, msg))

    orchestrator = Orchestrator(progress_callback=progress_cb)
    result: OrchestratorResult = orchestrator.run(
        file_path=csv_path,
        run_dir=run_dir,
        preferred_target="Revenue",
    )

    # 1. Pipeline Completion
    assert result.status == "completed"
    assert result.is_success is True
    assert result.run_dir == run_dir
    assert result.df is not None

    # 2. DIO Validation
    dio = result.dio
    assert isinstance(dio, DIO)
    val_issues = dio.validate()
    assert val_issues == [], f"DIO validation failed with issues: {val_issues}"

    # Verify DIO sections
    assert dio["ingestion"]["n_rows"] > 0
    assert dio["ingestion"]["n_columns"] > 0
    assert dio["domain_guess"]["domain"] in ("retail", "ecommerce", "sales")
    assert dio["quality"]["score"] > 0
    assert len(dio["columns"]) > 0
    assert len(dio["cleaning_log"]) > 0
    assert len(dio["insights"]) >= 1

    # 3. Artifact Filesystem Verification
    artifacts = result.artifacts

    # Cleaned CSV
    cleaned_csv_path = artifacts.get("cleaned_csv")
    assert cleaned_csv_path is not None
    assert Path(cleaned_csv_path).exists()
    assert Path(cleaned_csv_path).stat().st_size > 50

    # EDA Charts
    chart_paths = artifacts.get("chart_paths", [])
    assert len(chart_paths) >= 1
    for cp in chart_paths:
        assert Path(cp).exists()
        assert Path(cp).stat().st_size > 500

    # PDF Report
    pdf_report_path = artifacts.get("pdf_report")
    assert pdf_report_path is not None
    assert Path(pdf_report_path).exists()
    assert Path(pdf_report_path).stat().st_size > 1000

    # PPTX Presentation
    pptx_report_path = artifacts.get("pptx_report")
    assert pptx_report_path is not None
    assert Path(pptx_report_path).exists()
    assert Path(pptx_report_path).stat().st_size > 1000

    # Persisted DIO JSON
    dio_json_path = run_dir / "dio.json"
    assert dio_json_path.exists()
    saved_dio = json.loads(dio_json_path.read_text(encoding="utf-8"))
    assert saved_dio["dataset_hash"] == dio.dataset_hash
    assert saved_dio["schema_version"] == "1.0.0"

    # Pipeline Log
    pipeline_log_path = run_dir / "pipeline.log"
    assert pipeline_log_path.exists()
    assert pipeline_log_path.stat().st_size > 100

    # 4. Stage Timings and Statuses
    expected_stages = ["validation", "intelligence", "cleaning", "eda", "ml", "insight", "report"]
    for stg in expected_stages:
        assert stg in result.stage_timings
        assert result.stage_timings[stg] >= 0.0
        assert result.stage_statuses[stg] in ("completed", "skipped")

    assert "total_pipeline" in result.stage_timings
    assert result.stage_timings["total_pipeline"] > 0

    # 5. Progress Callbacks
    assert len(progress_history) >= 7
    last_call = progress_history[-1]
    assert last_call[1] == 1.0  # Finished at 100%
