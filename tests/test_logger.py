"""
tests/test_logger.py
====================
Tests for structured logging, file handler creation, and sensitive token redaction.
"""

from pathlib import Path
from core.logger import setup_run_logger, get_logger, SensitiveDataFilter


def test_setup_run_logger_creates_file(tmp_path: Path):
    run_id = "test_run_123"
    logger = setup_run_logger(run_id=run_id, run_dir=tmp_path)
    logger.info("Pipeline started successfully.")

    log_file = tmp_path / "pipeline.log"
    assert log_file.is_file()
    content = log_file.read_text(encoding="utf-8")
    assert "Pipeline started successfully." in content
    assert "test_run_123" in content


def test_logger_redaction_filter(tmp_path: Path):
    run_id = "redaction_test"
    logger = setup_run_logger(run_id=run_id, run_dir=tmp_path)

    logger.info("Connecting with api_key: 'super_secret_api_key_12345'")
    logger.info("OpenAI token sk-1234567890abcdef1234567890abcdef")

    log_file = tmp_path / "pipeline.log"
    content = log_file.read_text(encoding="utf-8")

    assert "super_secret_api_key_12345" not in content
    assert "REDACTED" in content


def test_get_logger_fallback():
    logger = get_logger("test_module", run_id="run_abc")
    assert logger is not None
    logger.info("Message through get_logger")
