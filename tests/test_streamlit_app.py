"""
tests/test_streamlit_app.py
===========================
Tests for Phase 9 Streamlit application logic.
Validates:
  1. app module import and core function availability
  2. safe_preview_dataframe masks PII columns correctly
  3. Dashboard rendering handles complete, partial, and failed states
  4. Session state isolation and reset behavior
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

import app
from core.dio import DIO
from orchestrator import OrchestratorResult


def test_app_imports():
    """Verify app.py module and primary entrypoints exist."""
    assert hasattr(app, "main")
    assert hasattr(app, "init_page_config")
    assert hasattr(app, "safe_preview_dataframe")
    assert hasattr(app, "render_dashboard")


def test_safe_preview_dataframe_pii_masking():
    """Verify safe_preview_dataframe masks columns identified as PII in DIO."""
    df = pd.DataFrame({
        "name": ["Alice Smith", "Bob Jones"],
        "email": ["alice@example.com", "bob@example.com"],
        "age": [28, 34],
        "salary": [75000, 85000],
    })

    dio = DIO.create_empty("test.csv")
    dio["columns"] = [
        {"name": "name", "is_pii": True, "pii_type": "name"},
        {"name": "email", "is_pii": True, "pii_type": "email"},
        {"name": "age", "is_pii": False},
        {"name": "salary", "is_pii": False},
    ]

    preview = app.safe_preview_dataframe(df, dio, n_rows=2)

    # PII columns must be redacted
    for val in preview["name"]:
        assert "[REDACTED" in val
    for val in preview["email"]:
        assert "[REDACTED" in val

    # Non-PII columns must be preserved
    assert list(preview["age"]) == [28, 34]
    assert list(preview["salary"]) == [75000, 85000]


@patch("streamlit.columns")
@patch("streamlit.metric")
@patch("streamlit.tabs")
@patch("streamlit.markdown")
def test_render_dashboard_handles_complete_result(mock_md, mock_tabs, mock_metric, mock_cols):
    """Verify render_dashboard does not raise on a valid OrchestratorResult."""
    # Setup mock tabs and dynamic columns
    tab_mocks = [MagicMock() for _ in range(7)]
    mock_tabs.return_value = tab_mocks
    mock_cols.side_effect = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]

    dio = DIO.create_empty("sales.csv")
    dio["ingestion"] = {"n_rows": 100, "n_columns": 5, "file_type": "csv"}
    dio["domain_guess"] = {"domain": "retail", "confidence": 0.9}
    dio["quality"] = {"score": 95, "issues": []}

    result = OrchestratorResult(
        status="completed",
        run_dir=Path("runs/test_run"),
        dio=dio,
        df=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
        stage_timings={"total_pipeline": 4.5},
        stage_statuses={"validation": "completed"},
    )

    # Should execute without uncaught exceptions
    app.render_dashboard(result, "sales.csv")
    assert mock_metric.called


@patch("streamlit.error")
def test_render_dashboard_handles_none_dio(mock_err):
    """Verify render_dashboard gracefully alerts when DIO is None (fatal early failure)."""
    result = OrchestratorResult(
        status="failed",
        run_dir=None,
        dio=None,
        df=None,
        errors=[{"stage": "validation", "error": "Invalid format"}],
    )

    app.render_dashboard(result, "corrupted.txt")
    assert mock_err.called


# =============================================================================
# STREAMLIT SESSION-STATE & REPORT INTEGRATION AUDIT (Items 5 & 6)
# =============================================================================

def test_streamlit_session_state_prevents_pipeline_reexecution():
    """
    Verify that when pipeline_result exists in st.session_state,
    subsequent Streamlit actions/reruns do not execute Orchestrator.run() a second time.
    """
    dio = DIO.create_empty("test.csv")
    existing_result = OrchestratorResult(
        status="completed",
        run_dir=Path("runs/cached_run"),
        dio=dio,
        df=pd.DataFrame({"a": [1]}),
    )

    fake_session = {"pipeline_result": existing_result, "file_name": "test.csv"}

    with patch("streamlit.session_state", fake_session), \
         patch("orchestrator.Orchestrator.run") as mock_run, \
         patch("streamlit.sidebar"), \
         patch("app.render_dashboard") as mock_render, \
         patch("streamlit.button", return_value=False):
        # Simulate Streamlit rerun where user did not click 'Analyze Dataset'
        app.main()
        # Orchestrator must NOT have been called
        assert not mock_run.called
        # Dashboard must have been rendered from session_state
        assert mock_render.called


def test_streamlit_consumes_existing_report_artifacts_without_duplication():
    """
    Verify app.py does not contain duplicate report generation logic
    (does not instantiate ReportLab or python-pptx doc builders), but purely
    consumes the pre-compiled artifact filepaths from dio["artifacts"].
    """
    import inspect

    app_source = inspect.getsource(app)

    # Must NOT import ReportLab builders in app.py
    assert "SimpleDocTemplate" not in app_source
    assert "reportlab.platypus" not in app_source
    assert "from reportlab" not in app_source

    # Must NOT import python-pptx Presentation builders in app.py
    assert "from pptx import Presentation" not in app_source
    assert "import pptx" not in app_source

    # Must download existing artifact paths directly
    assert "artifacts.get(\"pdf_report\")" in app_source
    assert "artifacts.get(\"pptx_report\")" in app_source
