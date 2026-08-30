"""
core/logger.py
==============
Structured logging infrastructure supporting console and per-run file output.
Includes redaction filter to prevent API keys and credentials from being logged.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

# Sensitive pattern tuples: (compiled regex, replacement template)
SENSITIVE_PATTERNS = [
    (
        re.compile(r"(?i)(api[_-]?key|secret|password|token|bearer)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?"),
        r"\1: [REDACTED]",
    ),
    (
        re.compile(r"sk-[a-zA-Z0-9]{20,48}"),
        r"[REDACTED_API_KEY]",
    ),
]


class SensitiveDataFilter(logging.Filter):
    """Filter that masks sensitive tokens, credentials, and API keys."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            msg = record.msg
            for pattern, replacement in SENSITIVE_PATTERNS:
                msg = pattern.sub(replacement, msg)
            record.msg = msg
        return True


def setup_run_logger(
    run_id: str,
    run_dir: Path | str | None = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure structured logging for a pipeline run.
    Attaches a console stream handler and a per-run `pipeline.log` file handler.
    """
    logger = logging.getLogger("autonomous_data_analyst")
    logger.setLevel(log_level)
    logger.propagate = False

    # Clear existing handlers to prevent duplicate lines across runs
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [run_id=%(run_id)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    redaction_filter = SensitiveDataFilter()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(redaction_filter)
    logger.addHandler(console_handler)

    # File Handler (if run_dir provided)
    if run_dir is not None:
        log_dir = Path(run_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "pipeline.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redaction_filter)
        logger.addHandler(file_handler)

    # Extra context adapter
    return logging.LoggerAdapter(logger, {"run_id": run_id})  # type: ignore[return-value]


def get_logger(name: str = "autonomous_data_analyst", run_id: str = "global") -> logging.LoggerAdapter:
    """
    Retrieve a logger adapter pre-populated with run_id.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s]: %(message)s")
        )
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logging.LoggerAdapter(logger, {"run_id": run_id})
