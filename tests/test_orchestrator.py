"""
tests/test_orchestrator.py
==========================
Unit and behavioral tests for Phase 9 Orchestrator.
Validates:
  1. Pipeline stage execution order
  2. DIO propagation across all stages
  3. Progress callback invocation with stage names and increasing percentages
  4. Fatal failure handling (Validation failure, Cleaning failure)
  5. Recoverable failure handling (Intelligence failure, EDA failure, ML failure, Report failure)
  6. Explicit failure tracking without fabricated fallbacks (Intelligence recoverable handling)
  7. ML skipping when no eligible target exists
  8. Run directory and DIO json persistence
  9. Run isolation between multiple sequential pipeline executions
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from core.base_agent import BaseAgent, ProgressState
from core.config import AppConfig, load_config
from core.dio import DIO
from orchestrator import Orchestrator, OrchestratorResult, PIPELINE_STAGES


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(
        "id,age,salary,department,signup_date\n"
        "1,25,50000,Engineering,2021-01-15\n"
        "2,30,60000,Marketing,2021-02-20\n"
        "3,35,75000,Engineering,2021-03-10\n"
        "4,40,90000,Executive,2021-04-05\n"
        "5,28,55000,Marketing,2021-05-12\n",
        encoding="utf-8",
    )
    return csv_file


@pytest.fixture
def mock_agents():
    """Create lightweight mock agents for fast deterministic testing."""
    class MockIntelligence(BaseAgent):
        name = "intelligence"
        def run(self, df, dio):
            dio["progress"][self.name] = ProgressState.FINISHED.value
            dio["domain_guess"] = {"domain": "corporate", "confidence": 0.95}
            dio["columns"] = [{"name": c, "dtype": str(df[c].dtype), "is_pii": False} for c in df.columns]
            return df, dio

    class MockCleaning(BaseAgent):
        name = "cleaning"
        def run(self, df, dio, run_dir=None):
            dio["progress"][self.name] = ProgressState.FINISHED.value
            dio["cleaning_log"].append({"step": "deduplication", "action": "checked"})
            return df, dio

    class MockEDA(BaseAgent):
        name = "eda"
        def run(self, df, dio, run_dir=None):
            dio["progress"][self.name] = ProgressState.FINISHED.value
            dio["eda"]["summary_stats"] = {"age": {"mean": 31.6}}
            return df, dio

    class MockML(BaseAgent):
        name = "ml"
        def run(self, df, dio, run_dir=None, preferred_target=None):
            dio["progress"][self.name] = ProgressState.FINISHED.value
            dio["ml"]["problem_type"] = "regression"
            dio["ml"]["target_column"] = preferred_target or "salary"
            dio["ml"]["best_model"] = "RandomForest"
            return df, dio

    class MockInsight(BaseAgent):
        name = "insight"
        def run(self, df, dio, run_dir=None):
            dio["progress"][self.name] = ProgressState.FINISHED.value
            dio["insights"].append({
                "category": "Compensation",
                "insight": "Engineering salary correlates strongly with age.",
                "evidence": ["Age 25: 50k", "Age 40: 90k"],
                "recommendation": "Review salary bands across departments.",
                "confidence": 0.92,
            })
            return df, dio

    class MockReport(BaseAgent):
        name = "report"
        def run(self, df, dio, run_dir=None):
            dio["progress"][self.name] = ProgressState.FINISHED.value
            if run_dir:
                pdf = Path(run_dir) / "report.pdf"
                pdf.write_bytes(b"%PDF-1.4 dummy")
                dio["artifacts"]["pdf_report"] = str(pdf)
            return df, dio

    return {
        "intelligence": MockIntelligence(),
        "cleaning": MockCleaning(),
        "eda": MockEDA(),
        "ml": MockML(),
        "insight": MockInsight(),
        "report": MockReport(),
    }


def test_orchestrator_stage_definitions():
    """Verify all 7 standard pipeline stages are defined with correct fatal flags."""
    stage_names = [s.name for s in PIPELINE_STAGES]
    assert stage_names == ["validation", "intelligence", "cleaning", "eda", "ml", "insight", "report"]
    fatal_stages = [s.name for s in PIPELINE_STAGES if s.is_fatal]
    assert fatal_stages == ["validation", "cleaning"]


def test_orchestrator_successful_execution(sample_csv, tmp_path, mock_agents):
    """Verify end-to-end execution through all 7 stages with mock agents."""
    run_dir = tmp_path / "test_run"
    progress_calls = []

    def callback(stage, pct, msg):
        progress_calls.append((stage, pct, msg))

    orchestrator = Orchestrator(agents=mock_agents, progress_callback=callback)
    result = orchestrator.run(sample_csv, run_dir=run_dir, preferred_target="salary")

    assert result.status == "completed"
    assert result.is_success is True
    assert result.dio is not None
    assert result.run_dir == run_dir

    # Check stage statuses
    assert result.stage_statuses["validation"] == "completed"
    assert result.stage_statuses["intelligence"] == "completed"
    assert result.stage_statuses["cleaning"] == "completed"
    assert result.stage_statuses["eda"] == "completed"
    assert result.stage_statuses["ml"] == "completed"
    assert result.stage_statuses["insight"] == "completed"
    assert result.stage_statuses["report"] == "completed"

    # Check progress callback was triggered across stages
    recorded_stages = [call[0] for call in progress_calls]
    assert "validation" in recorded_stages
    assert "intelligence" in recorded_stages
    assert "cleaning" in recorded_stages
    assert "eda" in recorded_stages
    assert "ml" in recorded_stages
    assert "insight" in recorded_stages
    assert "report" in recorded_stages

    # Verify DIO json persistence
    dio_file = run_dir / "dio.json"
    assert dio_file.exists()
    saved_dio = json.loads(dio_file.read_text(encoding="utf-8"))
    assert saved_dio["dataset_hash"] != ""
    assert saved_dio["domain_guess"]["domain"] == "corporate"


def test_orchestrator_fatal_validation_failure(tmp_path):
    """Verify invalid file causes fatal halt at validation stage."""
    empty_file = tmp_path / "corrupt.csv"
    empty_file.write_text("", encoding="utf-8")  # Empty file

    orchestrator = Orchestrator()
    result = orchestrator.run(empty_file)

    assert result.status == "failed"
    assert result.stage_statuses["validation"] == "failed"
    assert len(result.errors) >= 1
    assert result.errors[0]["stage"] == "validation"
    assert result.stage_statuses["intelligence"] == "pending"
    assert result.stage_statuses["cleaning"] == "pending"


def test_orchestrator_fatal_cleaning_failure(sample_csv, tmp_path, mock_agents):
    """Verify failure during cleaning halts the pipeline immediately (fatal stage)."""
    class FailingCleaning(BaseAgent):
        name = "cleaning"
        def run(self, df, dio, run_dir=None):
            raise RuntimeError("Catastrophic cleaning failure: corrupt column types")

    mock_agents["cleaning"] = FailingCleaning()
    orchestrator = Orchestrator(agents=mock_agents)
    result = orchestrator.run(sample_csv, run_dir=tmp_path / "clean_fail")

    assert result.status == "failed"
    assert result.stage_statuses["cleaning"] == "failed"
    assert result.stage_statuses["eda"] == "pending"
    assert result.stage_statuses["ml"] == "pending"
    assert any(e["stage"] == "cleaning" for e in result.errors)


def test_orchestrator_intelligence_recoverable_failure(sample_csv, tmp_path, mock_agents):
    """
    Mandatory Test: Intelligence failure is RECOVERABLE, does NOT fabricate fallbacks,
    records failure explicitly in DIO/telemetry, marks stage failed/partial,
    never hides original exception, and sets overall pipeline status to 'partial'.
    """
    class FailingIntelligence(BaseAgent):
        name = "intelligence"
        def run(self, df, dio):
            raise ValueError("Ollama connection timeout during semantic labeling")

    mock_agents["intelligence"] = FailingIntelligence()
    orchestrator = Orchestrator(agents=mock_agents)
    result = orchestrator.run(sample_csv, run_dir=tmp_path / "intel_fail")

    # Pipeline continued past Intelligence through Cleaning, EDA, ML, Insight, Report
    assert result.status == "partial"
    assert result.stage_statuses["intelligence"] == "failed"
    assert result.stage_statuses["cleaning"] == "completed"
    assert result.stage_statuses["eda"] == "completed"
    assert result.stage_statuses["report"] == "completed"

    # Explicit failure recorded in DIO without swallowing
    dio = result.dio
    assert dio["progress"]["intelligence"] == ProgressState.FAILED.value
    intel_errors = [e for e in dio["errors"] if e["stage"] == "intelligence"]
    assert len(intel_errors) >= 1
    assert "Ollama connection timeout" in intel_errors[0]["error"]


def test_orchestrator_recoverable_eda_failure(sample_csv, tmp_path, mock_agents):
    """Verify EDA failure does not crash pipeline and marks status as partial."""
    class FailingEDA(BaseAgent):
        name = "eda"
        def run(self, df, dio, run_dir=None):
            raise RuntimeError("Plotly kaleido export crashed")

    mock_agents["eda"] = FailingEDA()
    orchestrator = Orchestrator(agents=mock_agents)
    result = orchestrator.run(sample_csv, run_dir=tmp_path / "eda_fail")

    assert result.status == "partial"
    assert result.stage_statuses["eda"] == "failed"
    assert result.stage_statuses["ml"] == "completed"
    assert result.stage_statuses["insight"] == "completed"
    assert result.stage_statuses["report"] == "completed"


def test_orchestrator_ml_skipped_behavior(sample_csv, tmp_path, mock_agents):
    """Verify ML agent skipping sets stage_status to 'skipped' and preserves completed status."""
    class SkippingML(BaseAgent):
        name = "ml"
        def run(self, df, dio, run_dir=None, preferred_target=None):
            dio["progress"][self.name] = ProgressState.SKIPPED.value
            dio["ml"]["problem_type"] = "none"
            return df, dio

    mock_agents["ml"] = SkippingML()
    orchestrator = Orchestrator(agents=mock_agents)
    result = orchestrator.run(sample_csv, run_dir=tmp_path / "ml_skip")

    assert result.status == "completed"
    assert result.stage_statuses["ml"] == "skipped"


def test_orchestrator_run_isolation(sample_csv, tmp_path, mock_agents):
    """Verify multiple runs on the same orchestrator instance have clean isolated state."""
    orchestrator = Orchestrator(agents=mock_agents)

    run_dir1 = tmp_path / "run1"
    res1 = orchestrator.run(sample_csv, run_dir=run_dir1)

    run_dir2 = tmp_path / "run2"
    res2 = orchestrator.run(sample_csv, run_dir=run_dir2)

    assert res1.run_dir != res2.run_dir
    assert res1.dio.dataset_id != res2.dio.dataset_id
    assert (run_dir1 / "dio.json").exists()
    assert (run_dir2 / "dio.json").exists()


# =============================================================================
# ADDITIONAL FAILURE-INJECTION AUDIT TESTS (Item 3)
# =============================================================================

def test_orchestrator_fatal_unsupported_file_type(tmp_path):
    """Verify unsupported file extensions (.txt, .json, .parquet) cause immediate fatal halt."""
    unsupported_file = tmp_path / "data.json"
    unsupported_file.write_text('{"key": "value"}', encoding="utf-8")

    orchestrator = Orchestrator()
    result = orchestrator.run(unsupported_file)

    assert result.status == "failed"
    assert result.stage_statuses["validation"] == "failed"
    assert len(result.errors) >= 1
    assert "validation" in result.errors[0]["stage"]


def test_orchestrator_fatal_corrupt_unreadable_upload(tmp_path):
    """Verify unparseable corrupt bytes with .csv extension cause immediate fatal halt."""
    corrupt_file = tmp_path / "corrupt_data.csv"
    # Write invalid binary bytes that fail text decoding and CSV structure
    corrupt_file.write_bytes(b"\x00\xff\xfe\x00\x12\x34\x56\x78" * 20)

    orchestrator = Orchestrator()
    result = orchestrator.run(corrupt_file)

    assert result.status == "failed"
    assert result.stage_statuses["validation"] == "failed"
    assert len(result.errors) >= 1


def test_orchestrator_fatal_zero_row_dataset(tmp_path):
    """Verify header-only dataset with 0 data rows causes immediate fatal halt."""
    zero_row_file = tmp_path / "zero_rows.csv"
    zero_row_file.write_text("colA,colB,colC\n", encoding="utf-8")

    orchestrator = Orchestrator()
    result = orchestrator.run(zero_row_file)

    assert result.status == "failed"
    assert result.stage_statuses["validation"] == "failed"
    assert any("no data rows" in e.get("error", "").lower() or "zero" in e.get("error", "").lower() for e in result.errors)


def test_orchestrator_fatal_zero_column_dataset(tmp_path):
    """Verify empty/blank file with no columns causes immediate fatal halt."""
    zero_col_file = tmp_path / "zero_cols.csv"
    zero_col_file.write_text("\n\n\n", encoding="utf-8")

    orchestrator = Orchestrator()
    result = orchestrator.run(zero_col_file)

    assert result.status == "failed"
    assert result.stage_statuses["validation"] == "failed"


def test_orchestrator_fatal_nonexistent_file(tmp_path):
    """Verify non-existent file path causes immediate fatal halt."""
    nonexistent = tmp_path / "does_not_exist.csv"

    orchestrator = Orchestrator()
    result = orchestrator.run(nonexistent)

    assert result.status == "failed"
    assert result.stage_statuses["validation"] == "failed"


def test_orchestrator_recoverable_ml_failure(sample_csv, tmp_path, mock_agents):
    """Verify ML agent exception is recorded in DIO, pipeline status is 'partial', and downstream continues."""
    class FailingML(BaseAgent):
        name = "ml"
        def run(self, df, dio, run_dir=None, preferred_target=None):
            raise RuntimeError("Model training failed: matrix singular")

    mock_agents["ml"] = FailingML()
    orchestrator = Orchestrator(agents=mock_agents)
    result = orchestrator.run(sample_csv, run_dir=tmp_path / "ml_fail")

    assert result.status == "partial"
    assert result.stage_statuses["ml"] == "failed"
    # Downstream stages continued
    assert result.stage_statuses["insight"] == "completed"
    assert result.stage_statuses["report"] == "completed"
    # Recorded in DIO errors
    assert any(e.get("stage") == "ml" for e in result.dio["errors"])


def test_orchestrator_recoverable_insight_failure(sample_csv, tmp_path, mock_agents):
    """Verify Insight agent exception is recorded in DIO, pipeline status is 'partial', and Report continues."""
    class FailingInsight(BaseAgent):
        name = "insight"
        def run(self, df, dio, run_dir=None):
            raise RuntimeError("Insight generator failed: LLM token budget exhausted")

    mock_agents["insight"] = FailingInsight()
    orchestrator = Orchestrator(agents=mock_agents)
    result = orchestrator.run(sample_csv, run_dir=tmp_path / "insight_fail")

    assert result.status == "partial"
    assert result.stage_statuses["insight"] == "failed"
    assert result.stage_statuses["report"] == "completed"
    assert any(e.get("stage") == "insight" for e in result.dio["errors"])


def test_orchestrator_recoverable_report_failure(sample_csv, tmp_path, mock_agents):
    """Verify Report agent exception is recorded in DIO and pipeline status is 'partial'."""
    class FailingReport(BaseAgent):
        name = "report"
        def run(self, df, dio, run_dir=None):
            raise RuntimeError("Report generator failed: disk full")

    mock_agents["report"] = FailingReport()
    orchestrator = Orchestrator(agents=mock_agents)
    result = orchestrator.run(sample_csv, run_dir=tmp_path / "report_fail")

    assert result.status == "partial"
    assert result.stage_statuses["report"] == "failed"
    assert any(e.get("stage") == "report" for e in result.dio["errors"])


# =============================================================================
# DIO OWNERSHIP & IMMUTABILITY AUDIT (Item 4)
# =============================================================================

def test_orchestrator_dio_ownership_and_immutability(sample_csv, tmp_path, mock_agents):
    """
    Verify orchestrator only mutates its allowed sections (progress, errors, telemetry, artifacts)
    and does NOT overwrite or mutate agent-owned analytical sections.
    """
    import copy

    captured_snapshots = {}

    class TrackingIntelligence(BaseAgent):
        name = "intelligence"
        def run(self, df, dio):
            dio["domain_guess"] = {"domain": "finance", "confidence": 0.99}
            dio["quality"] = {"score": 98, "issues": ["minor missing"]}
            dio["columns"] = [{"name": "id", "dtype": "int64", "semantic_type": "identifier"}]
            captured_snapshots["intelligence"] = {
                "domain_guess": copy.deepcopy(dio["domain_guess"]),
                "quality": copy.deepcopy(dio["quality"]),
                "columns": copy.deepcopy(dio["columns"]),
            }
            return df, dio

    class TrackingEDA(BaseAgent):
        name = "eda"
        def run(self, df, dio, run_dir=None):
            dio["eda"] = {"summary_stats": {"metric_a": {"mean": 42.0}}}
            captured_snapshots["eda"] = copy.deepcopy(dio["eda"])
            return df, dio

    class TrackingML(BaseAgent):
        name = "ml"
        def run(self, df, dio, run_dir=None, preferred_target=None):
            dio["ml"] = {"problem_type": "classification", "best_model": "XGBoost"}
            captured_snapshots["ml"] = copy.deepcopy(dio["ml"])
            return df, dio

    class TrackingInsight(BaseAgent):
        name = "insight"
        def run(self, df, dio, run_dir=None):
            dio["insights"] = [{"category": "Revenue", "insight": "Revenue grew by 20%"}]
            captured_snapshots["insight"] = copy.deepcopy(dio["insights"])
            return df, dio

    mock_agents["intelligence"] = TrackingIntelligence()
    mock_agents["eda"] = TrackingEDA()
    mock_agents["ml"] = TrackingML()
    mock_agents["insight"] = TrackingInsight()

    orchestrator = Orchestrator(agents=mock_agents)
    result = orchestrator.run(sample_csv, run_dir=tmp_path / "immutability_run")

    assert result.status == "completed"
    final_dio = result.dio

    # Verify agent-owned sections were strictly preserved and not overwritten by orchestrator
    assert final_dio["domain_guess"] == captured_snapshots["intelligence"]["domain_guess"]
    assert final_dio["quality"] == captured_snapshots["intelligence"]["quality"]
    assert final_dio["columns"] == captured_snapshots["intelligence"]["columns"]
    assert final_dio["eda"] == captured_snapshots["eda"]
    assert final_dio["ml"] == captured_snapshots["ml"]
    assert final_dio["insights"] == captured_snapshots["insight"]
