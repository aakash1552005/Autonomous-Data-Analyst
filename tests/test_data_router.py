"""
tests/test_data_router.py
=========================
Tests for DataRouter ingestion coordination and initial DIO initialization.
"""

from pathlib import Path
import pytest
import pandas as pd
from core.data_router import DataRouter
from security.file_validator import FileValidationError
from core.persistence import load_dio_json


def test_data_router_ingest_valid_csv(tmp_path: Path):
    csv_file = tmp_path / "orders.csv"
    csv_file.write_text("order_id,amount,status\n1001,45.5,delivered\n1002,99.0,pending\n", encoding="utf-8")

    runs_dir = tmp_path / "runs"
    router = DataRouter(max_upload_size_mb=200)
    df, dio, run_dir = router.ingest(csv_file, base_runs_dir=runs_dir)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert len(df.columns) == 3

    assert dio.file_name == "orders.csv"
    assert len(dio.dataset_hash) == 64
    assert dio.ingestion["n_rows"] == 2
    assert dio.ingestion["n_columns"] == 3
    assert dio.ingestion["file_type"] == "csv"

    # Verify DIO was persisted to run_dir
    assert run_dir is not None
    assert run_dir.is_dir()
    dio_file = run_dir / "dio.json"
    assert dio_file.is_file()

    loaded_dio = load_dio_json(dio_file)
    assert loaded_dio.dataset_hash == dio.dataset_hash
    assert loaded_dio.ingestion["n_rows"] == 2


def test_data_router_rejects_invalid_file(tmp_path: Path):
    invalid_file = tmp_path / "bad.csv"
    invalid_file.write_text("", encoding="utf-8")  # empty

    router = DataRouter()
    with pytest.raises(FileValidationError) as exc_info:
        router.ingest(invalid_file)

    assert "File validation failed" in str(exc_info.value)
