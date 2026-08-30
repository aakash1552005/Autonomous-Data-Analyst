"""
tests/test_persistence.py
==========================
Tests for run directory creation, atomic DIO persistence, and path security.
"""

from pathlib import Path
import pytest
from core.dio import DIO
from core.persistence import (
    create_run_directory,
    save_dio_json,
    load_dio_json,
    resolve_run_path,
    sanitize_filename,
)


def test_sanitize_filename():
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("my sales report (2026).csv") == "my_sales_report__2026_.csv"
    assert sanitize_filename("..//..") == "dataset"


def test_create_run_directory(tmp_path: Path):
    run_dir = create_run_directory(tmp_path, "customer_data.csv")
    assert run_dir.is_dir()
    assert (run_dir / "charts").is_dir()
    assert "customer_data.csv" in run_dir.name


def test_save_and_load_dio_json(tmp_path: Path):
    dio = DIO.create_empty(file_name="sales.csv", dataset_hash="1234567890abcdef")
    dio.quality = {"score": 95, "issues": []}

    dio_file = tmp_path / "dio.json"
    saved_path = save_dio_json(dio, dio_file)
    assert saved_path.is_file()

    loaded_dio = load_dio_json(dio_file)
    assert loaded_dio.file_name == "sales.csv"
    assert loaded_dio.dataset_hash == "1234567890abcdef"
    assert loaded_dio.quality["score"] == 95


def test_resolve_run_path_traversal_guard(tmp_path: Path):
    safe_target = resolve_run_path(tmp_path, "subfolder/file.txt")
    assert str(safe_target).startswith(str(tmp_path.resolve()))

    with pytest.raises(ValueError):
        resolve_run_path(tmp_path, "../../outside_file.txt")
