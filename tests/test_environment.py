"""
tests/test_environment.py
=========================
Phase 0 environment verification tests.

Verifies:
- Python version >= 3.11
- All required packages can be imported
- Project directory structure exists
- Configuration files are present

These tests do NOT test agent behavior — that comes in later phases.
"""

import sys
import os
import importlib
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Project root — two levels up from this file (tests/ → project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPythonVersion:
    """Verify the Python runtime meets the minimum version requirement."""

    def test_python_major_version(self):
        assert sys.version_info.major == 3, "Python 3 is required"

    def test_python_minor_version(self):
        assert sys.version_info.minor >= 11, (
            f"Python 3.11+ is required, got 3.{sys.version_info.minor}"
        )


class TestRequiredImports:
    """Verify every core dependency can be imported."""

    REQUIRED_PACKAGES = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("openpyxl", "openpyxl"),
        ("dateutil", "python-dateutil"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "xgboost"),
        ("joblib", "joblib"),
        ("plotly", "plotly"),
        ("kaleido", "kaleido"),
        ("reportlab", "reportlab"),
        ("pptx", "python-pptx"),
        ("streamlit", "streamlit"),
        ("yaml", "pyyaml"),
        ("dotenv", "python-dotenv"),
        ("requests", "requests"),
        ("pytest", "pytest"),
    ]

    @pytest.mark.parametrize(
        "import_name,pip_name",
        REQUIRED_PACKAGES,
        ids=[p[1] for p in REQUIRED_PACKAGES],
    )
    def test_import(self, import_name: str, pip_name: str):
        """Each core dependency must be importable."""
        try:
            importlib.import_module(import_name)
        except ImportError:
            pytest.fail(
                f"Cannot import '{import_name}'. "
                f"Install it with: pip install {pip_name}"
            )


class TestProjectStructure:
    """Verify the project directory structure exists."""

    REQUIRED_DIRS = [
        "core",
        "security",
        "llm",
        "agents",
        "agents/intelligence",
        "agents/cleaning",
        "agents/eda",
        "agents/ml",
        "agents/insight",
        "agents/dashboard",
        "agents/report",
        "agents/chat",
        "utils",
        "tests",
        "data",
        "runs",
    ]

    REQUIRED_FILES = [
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "README.md",
    ]

    @pytest.mark.parametrize("directory", REQUIRED_DIRS)
    def test_directory_exists(self, directory: str):
        dir_path = PROJECT_ROOT / directory
        assert dir_path.is_dir(), f"Missing required directory: {directory}"

    @pytest.mark.parametrize("filename", REQUIRED_FILES)
    def test_file_exists(self, filename: str):
        file_path = PROJECT_ROOT / filename
        assert file_path.is_file(), f"Missing required file: {filename}"


class TestNoSecretsCommitted:
    """Verify that no secrets or sensitive files exist in the project root."""

    def test_no_dotenv_file(self):
        """The .env file (with real secrets) must NOT exist in the repo."""
        env_path = PROJECT_ROOT / ".env"
        # .env is allowed to exist locally but should be in .gitignore.
        # This test verifies .gitignore includes it.
        gitignore_path = PROJECT_ROOT / ".gitignore"
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            assert ".env" in content, ".env must be listed in .gitignore"

    def test_env_example_has_no_real_keys(self):
        """The .env.example must not contain actual API keys."""
        env_example = PROJECT_ROOT / ".env.example"
        if env_example.exists():
            content = env_example.read_text()
            # OPENAI_API_KEY should be blank or a placeholder
            for line in content.splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    value = line.split("=", 1)[1].strip()
                    assert value == "" or value.startswith("<"), (
                        "OPENAI_API_KEY in .env.example must be blank or a placeholder"
                    )
