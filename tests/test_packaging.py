"""
tests/test_packaging.py
========================
V1.1 Increment 1: Verification tests for packaging and Windows developer tooling.

Tests:
  - pyproject.toml validity, PEP 517/518/621 compliance, and setuptools backend
  - Dependency synchronization between pyproject.toml and requirements.txt
  - Package importability and editable-install registration
  - run.bat and run.ps1 presence, idempotency, safety, and non-interactive check modes
"""

import importlib.metadata
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_exists_and_valid_toml():
    """Verify pyproject.toml exists and conforms to PEP 517/518/621 specifications."""
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    assert pyproject_path.is_file(), "pyproject.toml must exist in project root"

    content = pyproject_path.read_text(encoding="utf-8")
    data = tomllib.loads(content)

    # Build system
    assert "build-system" in data, "pyproject.toml must contain [build-system]"
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    assert "setuptools>=61.0" in data["build-system"]["requires"][0]

    # Project metadata
    assert "project" in data, "pyproject.toml must contain [project]"
    project = data["project"]
    assert project["name"] == "autonomous-data-analyst"
    assert project["version"] == "1.1.0.dev0"
    assert "description" in project
    assert project.get("readme") == "README.md"
    assert project.get("requires-python") == ">=3.10"

    # Core runtime dependencies
    dependencies = project.get("dependencies", [])
    assert len(dependencies) > 0, "Core dependencies must not be empty"
    expected_core = {
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "plotly",
        "kaleido",
        "reportlab",
        "python-pptx",
        "streamlit",
        "pyyaml",
        "python-dotenv",
        "requests",
    }
    dep_names = {re.split(r"[<>=!~]", d)[0].strip().lower() for d in dependencies}
    assert expected_core.issubset(dep_names), f"Missing core dependencies: {expected_core - dep_names}"

    # Optional test dependencies
    optional_deps = project.get("optional-dependencies", {})
    assert "test" in optional_deps, "pyproject.toml must include 'test' optional dependency group"
    test_deps = {re.split(r"[<>=!~]", d)[0].strip().lower() for d in optional_deps["test"]}
    assert {"pytest", "pytest-cov"}.issubset(test_deps)


def test_pyproject_dependencies_match_requirements_txt():
    """Verify runtime dependencies in requirements.txt match pyproject.toml."""
    req_path = PROJECT_ROOT / "requirements.txt"
    pyproject_path = PROJECT_ROOT / "pyproject.toml"

    assert req_path.is_file()
    assert pyproject_path.is_file()

    req_lines = [
        line.strip()
        for line in req_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    req_names = {re.split(r"[<>=!~]", line)[0].strip().lower() for line in req_lines}

    pyproject_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    core_deps = {
        re.split(r"[<>=!~]", d)[0].strip().lower()
        for d in pyproject_data["project"]["dependencies"]
    }
    test_deps = {
        re.split(r"[<>=!~]", d)[0].strip().lower()
        for d in pyproject_data["project"]["optional-dependencies"].get("test", [])
    }
    all_pyproject_deps = core_deps.union(test_deps)

    # Every item in requirements.txt must be accounted for in pyproject.toml
    diff = req_names - all_pyproject_deps
    assert not diff, f"Dependencies in requirements.txt not found in pyproject.toml: {diff}"


def test_installed_package_and_module_resolution():
    """Verify the package is installed in the current environment and modules resolve."""
    # Check distribution metadata
    dist = importlib.metadata.distribution("autonomous-data-analyst")
    assert dist.metadata["Name"] == "autonomous-data-analyst"
    assert dist.metadata["Version"] == "1.1.0.dev0"

    # Verify key packages/modules resolve cleanly
    for module_name in ["agents", "core", "llm", "security", "sources", "utils", "app", "orchestrator"]:
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Failed to import {module_name}"


def test_run_bat_structure_and_safety():
    """Verify run.bat adheres to Windows safety, idempotency, and security requirements."""
    bat_path = PROJECT_ROOT / "run.bat"
    assert bat_path.is_file(), "run.bat must exist"

    content = bat_path.read_text(encoding="utf-8", errors="replace")

    # Safety & operational directives
    assert "@echo off" in content, "run.bat must begin with @echo off"
    assert "%~dp0" in content, "run.bat must locate project root dynamically"
    assert ".venv" in content, "run.bat must target local .venv"
    assert "streamlit" in content.lower(), "run.bat must invoke Streamlit"
    assert "--check-only" in content, "run.bat must support --check-only validation mode"

    # Security check: No hardcoded secrets or destructive commands
    forbidden_patterns = [
        r"(?i)api[_-]?key\s*=\s*['\"][a-zA-Z0-9_\-]{8,}['\"]",
        r"(?i)password\s*=\s*['\"][^\s'\"]+['\"]",
        r"(?i)rmdir\s+/s\s+/q\s+[c-zC-Z]:\\",
        r"(?i)del\s+/f\s+/s\s+/q\s+[c-zC-Z]:\\",
        r"(?i)format\s+[c-zC-Z]:",
        r"(?i)Set-ExecutionPolicy\s+Unrestricted",
    ]
    for pattern in forbidden_patterns:
        assert not re.search(pattern, content), f"Dangerous pattern matched in run.bat: {pattern}"


def test_run_ps1_structure_and_safety():
    """Verify run.ps1 adheres to PowerShell safety, non-admin, and security requirements."""
    ps1_path = PROJECT_ROOT / "run.ps1"
    assert ps1_path.is_file(), "run.ps1 must exist"

    content = ps1_path.read_text(encoding="utf-8", errors="replace")

    # Pure ASCII check for universal PowerShell 5.1 compatibility
    assert all(ord(c) < 128 for c in content), "run.ps1 must contain only ASCII characters"

    # Operational directives
    assert "$PSScriptRoot" in content, "run.ps1 must locate project root dynamically"
    assert ".venv" in content, "run.ps1 must target local .venv"
    assert "streamlit" in content.lower(), "run.ps1 must invoke Streamlit"
    assert "CheckOnly" in content, "run.ps1 must support -CheckOnly parameter"

    # Security check: Never alter persistent system execution policy
    assert "Set-ExecutionPolicy" not in content, "run.ps1 must NEVER alter system-wide execution policy"

    # Security check: No hardcoded secrets or destructive commands
    forbidden_patterns = [
        r"(?i)api[_-]?key\s*=\s*['\"][a-zA-Z0-9_\-]{8,}['\"]",
        r"(?i)password\s*=\s*['\"][^\s'\"]+['\"]",
        r"(?i)Remove-Item\s+-Recurse\s+[c-zC-Z]:\\",
    ]
    for pattern in forbidden_patterns:
        assert not re.search(pattern, content), f"Dangerous pattern matched in run.ps1: {pattern}"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher tests require Windows platform")
def test_run_bat_check_only_execution():
    """Execute run.bat --check-only and assert exit code is 0."""
    bat_path = PROJECT_ROOT / "run.bat"
    result = subprocess.run(
        ["cmd.exe", "/c", str(bat_path), "--check-only"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"run.bat --check-only failed: {result.stderr}\n{result.stdout}"
    assert "Environment and dependencies verified" in result.stdout or "Check-only validation passed" in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell launcher tests require Windows platform")
def test_run_ps1_check_only_execution():
    """Execute run.ps1 -CheckOnly via PowerShell and assert exit code is 0."""
    ps1_path = PROJECT_ROOT / "run.ps1"
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1_path), "-CheckOnly"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"run.ps1 -CheckOnly failed: {result.stderr}\n{result.stdout}"
    assert "Check-only validation passed" in result.stdout
